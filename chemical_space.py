import math
import subprocess
import itertools
from utilities import Utilities
import networkx as nx
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable, Draw
from tqdm import tqdm

import os
import pandas as pd
import numpy as np
import numba
from numba import jit

import logging
from functools import lru_cache, cache
from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, FrozenSet, Iterator
from collections import Counter

import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count

BOND_ORDER_MAPPING = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
    Chem.BondType.QUADRUPLE: 4.0,
    Chem.BondType.QUINTUPLE: 5.0,
    Chem.BondType.HEXTUPLE: 6.0,
    Chem.BondType.ONEANDAHALF: 1.5,
    Chem.BondType.TWOANDAHALF: 2.5,
    Chem.BondType.THREEANDAHALF: 3.5,
    Chem.BondType.FOURANDAHALF: 4.5,
    Chem.BondType.FIVEANDAHALF: 5.5,
    Chem.BondType.IONIC: 0.5,  # Ionic bonds are not typically represented numerically
    Chem.BondType.HYDROGEN: 0.1,  # Hydrogen bonds are weak interactions
    Chem.BondType.THREECENTER: 0.5,  # Approximate, as three-center bonds vary
    Chem.BondType.DATIVEONE: 1.0,  # Dative bonds typically behave like single bonds
    Chem.BondType.DATIVE: 1.0,
    Chem.BondType.DATIVEL: 1.0,
    Chem.BondType.DATIVER: 1.0,
    Chem.BondType.OTHER: None,  # Undefined bond type
    Chem.BondType.ZERO: 0.0,  # No bond
}

def _is_valid_molecule(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


def composition_to_mol(
    composition, possible_bonds, possible_valence, possible_radicals
):
    """Constructs a molecule from atom composition and a given bond combination."""
    mol = Chem.RWMol()
    for i, comp in enumerate(composition):
        atom = Chem.Atom(comp)
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(possible_radicals[i])
        atom.SetNumExplicitHs(possible_valence[i])
        mol.AddAtom(atom)

    atom_indices = range(len(composition))
    for (i, j), bond_type in zip(itertools.combinations(atom_indices, 2), possible_bonds):
        if bond_type != Chem.BondType.ZERO:
            mol.AddBond(i, j, bond_type)

    if len(Chem.rdmolops.GetMolFrags(mol)) > 1:
        return None

    try:
        mol.UpdatePropertyCache()
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        logging.debug(f"Sanitization failed: {e}")
        return None

def _process_generate_molecules(c):
    mol = composition_to_mol(c[0], c[1], c[2], c[3])
    if not mol:
        return None

    try:
        smi = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
        if not _is_valid_molecule(smi):
            return None
    except Exception as e:
        return None
    return smi

def _process_generate_molecules_batch(c_batch):
    valid_smi = []
    for c in c_batch:
        if c is None:
            continue
        
        smi = _process_generate_molecules(c)
        if smi:
            valid_smi.append(smi)
    return valid_smi

def batch_generator(source_generator: Iterator, batch_size: int) -> Iterator[List]:
    """
    Yields batches of a given size from a source generator.
    """
    while True:
        batch = list(itertools.islice(source_generator, batch_size))
        
        if not batch:
            return
            
        yield batch


@dataclass(unsafe_hash=True)
class ChemicalSpace:
    max_atoms: int = field(default=4)
    atoms: FrozenSet[str] = field(default_factory=lambda: frozenset(["C", "H", "O"]))
    bond_types: FrozenSet[Chem.BondType] = field(
        default_factory=lambda: frozenset(
            [
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                # Chem.BondType.AROMATIC,
                # Chem.BondType.QUADRUPLE,
                # Chem.BondType.QUINTUPLE,
                # Chem.BondType.HEXTUPLE,
                # Chem.BondType.ONEANDAHALF,
                # Chem.BondType.TWOANDAHALF,
                # Chem.BondType.THREEANDAHALF,
                # Chem.BondType.FOURANDAHALF,
                # Chem.BondType.FIVEANDAHALF,
                # Chem.BondType.IONIC,
                # Chem.BondType.HYDROGEN,
                # Chem.BondType.THREECENTER,
                # Chem.BondType.DATIVEONE,
                # Chem.BondType.DATIVE,
                # Chem.BondType.DATIVEL,
                # Chem.BondType.DATIVER,
                Chem.BondType.ZERO,
                # Chem.BondType.OTHER,
            ]
        )
    )

    molecules: Set[str] = field(init=False, default_factory=set)
    n_workers: int = field(default=4)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self):
        """
        Initialize derived attributes and set up logging after instance creation.
        """
        self.heavy_atoms = [atom for atom in self.atoms if atom != "H"]
        self.periodic_table = Chem.GetPeriodicTable()
        self.allowed_valences = {
            atom: tuple(self.periodic_table.GetValenceList(atom))
            for atom in self.atoms
        }
        Utilities._setup_logging()

    @staticmethod
    def n_choose_k(n, k):
        """Calculates nCk (n choose k) efficiently."""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        if k > n // 2:
            k = n - k # Optimization: C(n, k) = C(n, n-k)

        res = 1
        for i in range(k):
            res = res * (n - i) // (i + 1)
        return res

    def count_total_compositions(self) -> int:
        """
        Pre-computes the total number of compositions that will be generated
        """
        total_count = 0
        num_heavy_atom_types = len(self.heavy_atoms)
        num_bond_types = len(self.bond_types)
        num_valence_options = 5 # 0-4 explicit Hs
        num_radical_options = 5 # 0-4 radical electrons

        for total_heavy_atoms in range(1, self.max_atoms + 1):
            # Number of ways to choose 'total_heavy_atoms' from 'num_heavy_atom_types' with replacement (multiset coefficient)
            # (n + k - 1) choose k, where n = num_heavy_atom_types, k = total_heavy_atoms
            num_atom_compositions = (
                self.n_choose_k(num_heavy_atom_types + total_heavy_atoms - 1, total_heavy_atoms)
            )

            num_potential_bonds = int(total_heavy_atoms * (total_heavy_atoms - 1) / 2)
            num_bond_combinations = num_bond_types ** num_potential_bonds

            num_valence_combinations = num_valence_options ** total_heavy_atoms
            num_radical_combinations = num_radical_options ** total_heavy_atoms

            # For each unique atom composition, we multiply by bond, valence, and radical combinations
            total_count += (
                num_atom_compositions *
                num_bond_combinations *
                num_valence_combinations *
                num_radical_combinations
            )
        return total_count

    def generate_compositions(self) -> Iterator[Tuple]:
        """Generates compositions as a generator to save memory."""
        atom_list = list(self.heavy_atoms)
        num_valence_options = range(5) # 0-4 explicit Hs
        num_radical_options = range(5) # 0-4 radical electrons

        for total_heavy_atoms in range(1, self.max_atoms + 1):
            for composition_tuple in itertools.combinations_with_replacement(atom_list, total_heavy_atoms):
                num_potential_bonds = int(len(composition_tuple) * (len(composition_tuple) - 1) / 2)
                for possible_bonds_tuple in itertools.product(self.bond_types, repeat=num_potential_bonds):
                    for possible_valences_tuple in itertools.product(num_valence_options, repeat=len(composition_tuple)):
                        for possible_radicals_tuple in itertools.product(num_radical_options, repeat=len(composition_tuple)):
                            yield (
                                composition_tuple,
                                possible_bonds_tuple,
                                possible_valences_tuple,
                                possible_radicals_tuple,
                            )

    def generate_molecules(self):
        self.molecules.clear()
        try:
            total_compositions = self.count_total_compositions()
            logging.info(f"Attempting to process a total of {total_compositions:,} mathematical combinations.")
            if total_compositions > 200_000_000_000:
                 logging.warning("Total number of combinations is extremely large. This will likely never finish.")
        except OverflowError:
            total_compositions = -1
            logging.error("Total number of combinations is too large to fit in a standard integer. Cannot show progress.")

        compositions = self.generate_compositions()

        # batch_size = 32768
        batch_size = 131072
        batches = batch_generator(compositions, batch_size)

        logging.info(f"Starting molecule generation with {self.n_workers} processes...")

        if total_compositions > 0:
            total_batches = math.ceil(total_compositions / batch_size)
        else:
            total_batches = -1

        with Pool(processes=self.n_workers) as pool:
            iterator = pool.imap_unordered(_process_generate_molecules_batch, batches)
            progress_bar = tqdm(iterator, total=total_batches, desc="Generating Molecules")
            for smi_list in progress_bar:
                if smi_list:
                    self.molecules.update(smi_list)
                    progress_bar.set_postfix({"found": f"{len(self.molecules):,}"})
        # self.molecules = {smi for smi in results if smi is not None}

        logging.info(f"Generated {len(self.molecules):,} valid molecules")

    def save_molecules_to_csv_and_images(self, dir_path):
        molecules = sorted(self.molecules)
        pd.DataFrame(molecules).to_csv(f"{dir_path}/molecules.csv")

        os.makedirs(f"{dir_path}/molecule_images", exist_ok=True)
        os.makedirs(f"{dir_path}/molecule_xyz", exist_ok=True)
        os.makedirs(f"{dir_path}/molecule_images_obabel", exist_ok=True)

        for i, smi in enumerate(molecules):
            Draw.MolsToImage([Chem.MolFromSmiles(smi)]).save(
                f"{dir_path}/molecule_images/mol_{i}.png"
            )
            subprocess.run(
                [
                    "obabel",
                    "-:" + smi,
                    "-O" + f"{dir_path}/molecule_images_obabel/mol_{i}.png",
                ]
            )
            subprocess.run(
                [
                    "obabel",
                    "-:" + smi,
                    "-h",
                    "--gen3D",
                    "-O" + f"{dir_path}/molecule_xyz/mol_{i}.xyz",
                ]
            )


if __name__ == "__main__":
    import argparse

    from rdkit import RDLogger
    
    RDLogger.DisableLog("rdApp.error")

    parser = argparse.ArgumentParser(
        prog="crn_py",
    )
    parser.add_argument("-n", "--max_atoms", type=int, default=2)
    args = parser.parse_args()

    chemical_space = ChemicalSpace(max_atoms=args.max_atoms, n_workers=12)

    print("Generating molecules...")
    m = Chem.RWMol()
    m.AddAtom(Chem.Atom("C"))
    m.AddAtom(Chem.Atom("O"))

    m.AddBond(0, 1, Chem.BondType.ZERO)
    m.GetAtomWithIdx(0).SetNumRadicalElectrons(1)
    m.GetAtomWithIdx(1).SetNumRadicalElectrons(0)
    m.GetAtomWithIdx(0).SetNoImplicit(False)
    m.GetAtomWithIdx(1).SetNoImplicit(False)
    m.GetAtomWithIdx(0).SetNumExplicitHs(1)
    m.GetAtomWithIdx(1).SetNumExplicitHs(0)
    Chem.SanitizeMol(m)
    # params = Chem.SmilesParserParams()
    # params.removeHs = False
    # m = Draw.PrepareMolForDrawing(m, addChiralHs=True)
    # m = Chem.MolFromSmiles(Chem.MolToSmiles(m), params)
    # Draw.MolsToImage([m], explicitMethyl=True).save("test.png")
    print(Chem.MolToSmiles(m))
    print(Chem.MolToMolBlock(m))
    subprocess.run(
     ["obabel", "-:" + Chem.MolToSmiles(m), "-h", "--gen3D", "-Otest.xyz"]
    )
    subprocess.run(["obabel", "-:" + Chem.MolToSmiles(m), "-Otest.png"])

    g = nx.Graph()
    g.add_nodes_from(range(0, 2))
    g.add_edge(0, 1)
    d = list(nx.connected_components(g))
    print(d)

    chemical_space.generate_molecules()
    print(chemical_space.molecules)
    try:
      os.mkdir(str(args.max_atoms))
    except:
      pass
    chemical_space.save_molecules_to_csv_and_images(str(args.max_atoms))
