from .config import MolConfig

from dataclasses import dataclass, field
from os import cpu_count
import os
from multiprocessing import Process, Queue, Pool
import logging
import math
from functools import partial
from tqdm import tqdm
import lmdb
import pickle
import pandas as pd
from rdkit.Chem import Draw
import subprocess
import click
from typing import List, Tuple, Dict, FrozenSet, Iterator
from rdkit import Chem
import itertools
from collections import defaultdict
from typing import DefaultDict, Optional

from .workers import worker_process_compositions, worker_build_molecule
from .cleave import enumerate_cleavage_products


@dataclass
class ChemicalSpace:
    max_atoms: int = field(default=4)
    atoms: FrozenSet[str] = field(default_factory=lambda: frozenset(["C", "H", "O"]))
    config: MolConfig = field(default_factory=lambda: MolConfig())
    smiles: str = None
    output_dir: str = "chemical_space"
    n_workers: int = field(default_factory=cpu_count)

    _db_paths: Dict[str, str] = field(init=False, default_factory=dict)
    _allowed_valences: Dict[str, Tuple[int, ...]] = field(init=False)
    _heavy_atoms: List[str] = field(init=False)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        db_dir = os.path.join(self.output_dir, "db")
        os.makedirs(db_dir, exist_ok=True)

        self._db_paths = {
            "compositions": os.path.join(db_dir, "compositions_n{n}.lmdb"),
            "molecules": os.path.join(db_dir, "molecules_n{n}.lmdb"),
        }

        pt = Chem.GetPeriodicTable()
        self._heavy_atoms = sorted([atom for atom in self.atoms if atom != "H"])
        self._allowed_valences = {
            symbol: tuple(pt.GetValenceList(symbol)) for symbol in self._heavy_atoms
        }
        logging.info(
            f"ChemicalSpace initialized for {self.atoms} up to {self.max_atoms} heavy atoms."
        )

    def generate_compositions(self):
        click.secho("--- Stage 1: Generating Plausible Compositions ---", bold=True)
        for n in range(1, self.max_atoms + 1):
            self._generate_compositions_for_n(n)

    def build_molecules(self):
        click.secho(
            "\n--- Stage 2: Building Molecules from Compositions ---", bold=True
        )
        for n in range(1, self.max_atoms + 1):
            self._build_molecules_for_n(n)

    def explore(self):
        self.generate_compositions()
        self.build_molecules()
        self.build_cleavage_products_to_db(
            self.smiles, max_cuts=2, include_ring_bonds=True
        )
        click.secho(
            f"\nExploration complete. Results are in '{self.output_dir}'", fg="green"
        )
        self.export_to_csv()

    def _lmdb_writer(self, q: Queue, db_path: str, batch_size: int = 4096):
        # IMPORTANT: be explicit about directory-style LMDB
        env = lmdb.open(
            db_path,
            map_size=10**11,
            subdir=True,  # db_path is a directory
            create=True,
            lock=True,
            writemap=False,  # safer than writemap=True across platforms
            metasync=True,
            sync=True,
            readahead=False,
            meminit=False,
        )

        batch = {}
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                key, val = item
                batch[key] = val

                if len(batch) >= batch_size:
                    with env.begin(write=True) as txn:
                        for k, v in batch.items():
                            txn.put(k, v, overwrite=False)
                    batch.clear()

            # final flush
            if batch:
                with env.begin(write=True) as txn:
                    for k, v in batch.items():
                        txn.put(k, v, overwrite=False)
                batch.clear()

            # force persist
            env.sync()
        finally:
            env.close()

    def _generate_compositions_for_n(self, n_atoms: int):
        """
        Generates and saves plausible compositions for a specific number of heavy atoms.
        """

        db_path = self._db_paths["compositions"].format(n=n_atoms)
        if os.path.exists(db_path):
            click.secho(
                f"Skipping N={n_atoms}, database already exists: {db_path}", fg="green"
            )
            return

        click.secho(f"\nGenerating for N={n_atoms} heavy atoms...", bold=True)

        def task_generator():
            compositions = itertools.combinations_with_replacement(
                self._heavy_atoms, n_atoms
            )
            num_bonds = (n_atoms * (n_atoms - 1)) // 2
            for comp in compositions:
                for bonds in itertools.product(
                    self.config.BOND_OPTIONS, repeat=num_bonds
                ):
                    yield (comp, bonds)

        total_tasks = math.comb(len(self._heavy_atoms) + n_atoms - 1, n_atoms) * (
            len(self.config.BOND_OPTIONS) ** ((n_atoms * (n_atoms - 1)) // 2)
        )

        worker_func = partial(
            worker_process_compositions,
            n_atoms=n_atoms,
            allowed_valences_map=self._allowed_valences,
            config=self.config,
        )

        q = Queue(maxsize=self.n_workers * 2)
        writer = Process(target=self._lmdb_writer, args=(q, db_path))
        writer.start()

        total_saved = 0
        worker = partial(
            worker_process_compositions,
            n_atoms=n_atoms,
            allowed_valences_map=self._allowed_valences,
            config=self.config,
        )

        with Pool(processes=self.n_workers) as pool:
            for result_list in tqdm(
                pool.imap_unordered(worker, task_generator(), chunksize=1024),
                total=total_tasks,
                desc=f"Compositions N={n_atoms}",
            ):
                for comp, bonds, vals, rads in result_list:
                    key = f"{total_saved:012d}".encode()
                    q.put((key, pickle.dumps((comp, bonds, vals, rads))))
                    total_saved += 1

        q.put(None)
        writer.join()

    def _build_molecules_for_n(self, n_atoms: int):
        """
        Reads composition DB, builds molecules, and saves unique ones.
        """
        output_db_path = self._db_paths["molecules"].format(n=n_atoms)
        env_out = lmdb.open(
            output_db_path, map_size=10**11, writemap=True, metasync=False, sync=False
        )

        input_db_path = self._db_paths["compositions"].format(n=n_atoms)
        if not os.path.exists(input_db_path):
            click.secho(
                f"Can't find composition database at {os.path.basename(input_db_path)}...",
                fg="red",
            )
            return

        click.secho(f"Building from {os.path.basename(input_db_path)}...", bold=True)
        q = Queue(maxsize=self.n_workers * 2)
        writer = Process(target=self._lmdb_writer, args=(q, output_db_path))
        writer.start()

        env_in = lmdb.open(input_db_path, readonly=True, lock=False)
        total_tasks = env_in.stat()["entries"]
        with env_in.begin() as txn, Pool(self.n_workers) as pool:
            vals = (v for _, v in txn.cursor())
            for result in tqdm(
                pool.imap_unordered(worker_build_molecule, vals, chunksize=1024),
                total=total_tasks,
                desc=f"Building N={n_atoms}",
            ):
                if result:
                    smi, data = result
                    q.put((smi.encode(), data))

        q.put(None)
        writer.join()

        click.secho(
            f"\nTotal unique molecules found: {env_out.stat()['entries']:,}", fg="green"
        )

    def export_to_csv(self, filename: str = "molecules.csv"):
        """Exports the final molecules from the DB to a CSV file."""

        molecules_data = []
        for n in range(1, self.max_atoms + 1):
            output_db_path = self._db_paths["molecules"].format(n=n)
            output_csv_path = os.path.join(self.output_dir, filename)

            click.secho(f"Exporting molecules to {output_csv_path}...", bold=True)
            env = lmdb.open(output_db_path, readonly=True)

            with env.begin() as txn:
                for smi_key, value in tqdm(
                    txn.cursor(), total=txn.stat()["entries"], desc="Reading DB"
                ):
                    smi = smi_key.decode()
                    _, atom_counts = pickle.loads(value)
                    molecules_data.append({"SMILES": smi, **atom_counts})

        if not molecules_data:
            click.secho("No molecules to export.", fg="red")
            return

        df = pd.DataFrame(molecules_data).fillna(0).astype(int, errors="ignore")

        # Reorder columns to have SMILES first
        cols = ["SMILES"] + [col for col in df.columns if col != "SMILES"]
        df = df[cols]

        df = df[cols]

        df.to_csv(output_csv_path, index=False)
        click.secho(f"Export complete. There are {len(df)} molecules.", fg="green")

    def _check_obabel_installed(self):
        """Checks if Open Babel is installed and executable."""
        try:
            # Use -V as it's a common way to get version and confirm installation
            subprocess.run(["obabel", "-V"], capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            click.secho(
                "Open Babel (obabel) is not installed or not found in PATH. "
                "Please install it to use this feature.",
                fg="red",
                err=True,
            )
            raise click.exceptions.Exit(1)

    def _get_all_smiles_from_db(self) -> List[str]:
        """Collects all unique SMILES strings from the molecule databases."""
        click.secho(
            f"Collecting SMILES from LMDB databases in {self.output_dir}...", bold=True
        )
        all_smiles = set()
        for n in range(1, self.max_atoms + 1):
            db_path = self._db_paths["molecules"].format(n=n)
            if not os.path.exists(db_path):
                logging.info(
                    f"Molecule database for N={n} not found at {db_path}, skipping."
                )
                continue

            env = lmdb.open(db_path, readonly=True, lock=False)
            with env.begin() as txn:
                num_entries = txn.stat()["entries"]
                if num_entries == 0:
                    logging.info(
                        f"No entries in molecule database for N={n} at {db_path}, skipping."
                    )
                    continue
                logging.info(f"Reading {num_entries} SMILES from N={n} database...")
                for smi_key, _ in txn.cursor():
                    all_smiles.add(smi_key.decode())
            env.close()

        if not all_smiles:
            click.secho("No molecules found in databases to export.", fg="yellow")
            return []

        return sorted(list(all_smiles))

    def export_images(self):
        """Exports molecules from DB to PNG (RDKit & OpenBabel)."""
        self._check_obabel_installed()  # OpenBabel is used for one of the image types

        img_dir = os.path.join(self.output_dir, "molecule_images")
        img_obabel_dir = os.path.join(self.output_dir, "molecule_images_obabel")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(img_obabel_dir, exist_ok=True)

        sorted_smiles = self._get_all_smiles_from_db()
        if not sorted_smiles:
            return

        click.secho(
            f"\nExporting {len(sorted_smiles)} unique molecules as images to {self.output_dir}...",
            bold=True,
        )

        total_exported_rdkit = 0
        total_exported_obabel = 0
        for i, smi in enumerate(tqdm(sorted_smiles, desc="Exporting images")):
            mol_filename_base = f"mol_{i}"
            try:
                # RDKit PNG
                rdkit_mol = Chem.MolFromSmiles(smi)
                if rdkit_mol:
                    Draw.MolsToImage([rdkit_mol]).save(
                        os.path.join(img_dir, f"{mol_filename_base}.png")
                    )
                    total_exported_rdkit += 1
                else:
                    logging.warning(
                        f"Could not generate RDKit mol for SMILES: {smi} (file: {mol_filename_base}.png)"
                    )

                # OpenBabel PNG
                result_obabel_png = subprocess.run(
                    [
                        "obabel",
                        "-:" + smi,
                        "-O" + os.path.join(img_obabel_dir, f"{mol_filename_base}.png"),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result_obabel_png.returncode == 0:
                    total_exported_obabel += 1
                else:
                    logging.error(
                        f"OpenBabel PNG failed for SMILES: {smi} (file: {mol_filename_base}.png). Error: {result_obabel_png.stderr.strip()}"
                    )
            except Exception as e:
                logging.error(
                    f"Generic error processing SMILES for images: {smi} (file base: {mol_filename_base}). Error: {e}"
                )

        if total_exported_rdkit > 0 or total_exported_obabel > 0:
            click.secho(
                f"\nFinished exporting images. {total_exported_rdkit} RDKit images, "
                f"{total_exported_obabel} OpenBabel images processed into {self.output_dir}",
                fg="green",
            )
        else:
            click.secho("\nNo molecules were processed for image export.", fg="yellow")

    def export_xyz(self):
        """Exports molecules from DB to XYZ files using OpenBabel."""
        self._check_obabel_installed()

        xyz_dir = os.path.join(self.output_dir, "molecule_xyz")
        os.makedirs(xyz_dir, exist_ok=True)

        sorted_smiles = self._get_all_smiles_from_db()
        if not sorted_smiles:
            return

        click.secho(
            f"\nExporting {len(sorted_smiles)} unique molecules as XYZ files to {xyz_dir}...",
            bold=True,
        )

        total_exported = 0
        for i, smi in enumerate(tqdm(sorted_smiles, desc="Exporting XYZ files")):
            mol_filename_base = f"mol_{i}"
            try:
                # XYZ
                result_xyz = subprocess.run(
                    [
                        "obabel",
                        "-:" + smi,
                        "--gen3D",
                        "-O" + os.path.join(xyz_dir, f"{mol_filename_base}.xyz"),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result_xyz.returncode == 0:
                    total_exported += 1
                else:
                    logging.error(
                        f"OpenBabel XYZ failed for SMILES: {smi} (file: {mol_filename_base}.xyz). Error: {result_xyz.stderr.strip()}"
                    )
            except Exception as e:
                logging.error(
                    f"Generic error processing SMILES for XYZ: {smi} (file base: {mol_filename_base}). Error: {e}"
                )

        if total_exported > 0:
            click.secho(
                f"\nFinished exporting XYZ. {total_exported} molecules processed into {xyz_dir}",
                fg="green",
            )
        else:
            click.secho("\nNo molecules were processed for XYZ export.", fg="yellow")

    def _atom_counts_from_smiles(self, smi: str) -> Optional[Dict[str, int]]:
        """
        Returns atom counts including H (explicitly added), e.g. {"C": 2, "H": 7, "N": 1}
        Returns None if SMILES is invalid.
        """
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        molH = Chem.AddHs(mol)
        counts: Dict[str, int] = {}
        for a in molH.GetAtoms():
            sym = a.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
        return counts

    def _heavy_atom_count_from_smiles(self, smi: str) -> Optional[int]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "H")

    def _save_smiles_to_molecule_dbs(self, smiles_iter: Iterator[str]) -> int:
        """
        Appends SMILES into the existing molecules_n{n}.lmdb DBs based on heavy atom count.
        Stores value compatible with export_to_csv(): pickle.dumps((None, atom_counts)).
        Returns number of new writes attempted (duplicates will be ignored by overwrite=False).
        """
        # Group items by N heavy atoms so we can use one writer per DB path.
        by_n: DefaultDict[int, List[Tuple[bytes, bytes]]] = defaultdict(list)

        total = 0
        for smi in smiles_iter:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # Canonicalise
            smi_can = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)

            n_heavy = sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "H")
            self.max_atoms = max(self.max_atoms, n_heavy)
            # if n_heavy < 1 or n_heavy > self.max_atoms:
            #     # Skip fragments outside configured space
            #     continue

            atom_counts = self._atom_counts_from_smiles(smi_can)
            if atom_counts is None:
                continue

            key = smi_can.encode("utf-8")
            val = pickle.dumps((None, atom_counts))
            by_n[n_heavy].append((key, val))
            total += 1

        if not by_n:
            return 0

        # Start a writer process per n (same pattern as _build_molecules_for_n)
        writers = {}
        queues = {}

        for n, items in by_n.items():
            db_path = self._db_paths["molecules"].format(n=n)
            os.makedirs(db_path, exist_ok=True)  # LMDB is a directory
            q = Queue(maxsize=self.n_workers * 2)
            p = Process(target=self._lmdb_writer, args=(q, db_path))
            p.start()
            queues[n] = q
            writers[n] = p

        # Feed writers
        for n, items in by_n.items():
            q = queues[n]
            for kv in items:
                q.put(kv)

        # Stop writers
        for n, q in queues.items():
            q.put(None)
        for n, p in writers.items():
            p.join()

        return total

    def build_cleavage_products_to_db(
        self,
        parent_smiles: str,
        *,
        max_cuts: int = 2,
        include_ring_bonds: bool = True,
        include_parent: bool = True,
    ) -> int:
        """
        Enumerates cleavage products from `parent_smiles` and appends all resulting
        fragment SMILES into the existing molecules_n{n}.lmdb DB(s).

        Returns number of fragment SMILES processed for saving (duplicates ignored by LMDB).
        """
        if parent_smiles is None:
            click.secho(
                "No parent SMILES provided for cleavage product generation.", fg="red"
            )
            return 0
        click.secho("\n--- Stage 3: Generating Cleavage Products ---", bold=True)

        res = enumerate_cleavage_products(
            parent_smiles,
            max_cuts=max_cuts,
            include_ring_bonds=include_ring_bonds,
        )

        def frag_generator():
            if include_parent:
                yield parent_smiles
            for r in res:
                # r.fragments is a tuple of SMILES for that cut pattern
                for frag_smi in r.fragments:
                    print(frag_smi)
                    yield frag_smi

        n_saved = self._save_smiles_to_molecule_dbs(frag_generator())
        click.secho(f"Cleavage products processed for saving: {n_saved}", fg="green")
        return n_saved

    def build_cleavage_products_list_to_db(
        self,
        parent_smiles_list: List[str],
        *,
        max_cuts: int = 1,
        include_ring_bonds: bool = True,
        include_parent: bool = True,
    ) -> int:
        """
        Same as build_cleavage_products_to_db but for a list of parent SMILES.
        """

        def frag_generator():
            for smi in parent_smiles_list:
                if include_parent:
                    yield smi
                res = enumerate_cleavage_products(
                    smi,
                    max_cuts=max_cuts,
                    include_ring_bonds=include_ring_bonds,
                )
                for r in res:
                    for frag_smi in r.fragments:
                        yield frag_smi

        n_saved = self._save_smiles_to_molecule_dbs(frag_generator())
        click.secho(
            f"Cleavage products processed for saving (all parents): {n_saved}",
            fg="green",
        )
        return n_saved
