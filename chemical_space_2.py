from config import MolConfig
from functools import wraps
from collections import Counter
from rdkit import Chem
import itertools
from dataclasses import dataclass, field
from os import cpu_count
import os
from multiprocessing import Process, Queue, Pool
import networkx as nx
import logging
import math
from functools import partial
from tqdm import tqdm
import lmdb
import pickle
import pandas as pd

import argparse
import click
from typing import Set, List, Tuple, Dict, Optional, FrozenSet, Iterator

########################################
# Worker Functions for multi processes #
########################################

def worker_process_compositions(task_tuple, n_atoms, allowed_valences_map, config):
	composition, bonds = task_tuple
	plausible_combinations = []

	# Remove fragmented molecules
	if n_atoms > 1:
		G = nx.Graph()
		G.add_nodes_from(range(n_atoms))

		atom_pairs = list(itertools.combinations(range(n_atoms), 2))

		for idx, (i, j) in enumerate(atom_pairs):
			if bonds[idx] != Chem.BondType.ZERO:
				G.add_edge(i, j)

		if not nx.is_connected(G):
			return []

	# Calculate sum of bond orders for each atom to filter out invalid composition 
	# and find valid radical/hydrogen combinations
	bond_sum = [0] * n_atoms
	if n_atoms > 1:
		atom_pairs = list(itertools.combinations(range(n_atoms), 2))
		for idx, (i, j) in enumerate(atom_pairs):
			v = config.BOND_VALUES.get(bonds[idx], 0.0)
			bond_sum[i] += v
			bond_sum[j] += v

	for radicals in itertools.product(config.RADICAL_OPTIONS, repeat=n_atoms):
		valences = []
		is_plausible = True
		for i, sym in enumerate(composition):
			needed_hs = None
			for total_valence in allowed_valences_map.get(sym, []):
				h = int(total_valence - bond_sum[i] - radicals[i])
				if h in config.VALENCE_OPTIONS:
					needed_hs = h
					break
			if needed_hs is None:
				is_plausible = False
				break
			valences.append(needed_hs)
		
		if is_plausible:
			plausible_combinations.append(
				(composition, bonds, tuple(valences), radicals)
			)
			
	return plausible_combinations


def worker_build_molecule(data: bytes) -> Tuple[str, bytes] | None:
	try:
		composition, bonds, valences, radicals = pickle.loads(data)
		mol = Chem.RWMol()
		for i, comp in enumerate(composition):
			atom = Chem.Atom(comp)
			atom.SetNoImplicit(True) # Important for explicit control
			atom.SetNumRadicalElectrons(radicals[i])
			atom.SetNumExplicitHs(valences[i])
			mol.AddAtom(atom)

		if len(composition) > 1:
			atom_indices = range(len(composition))
			for (i, j), bond_type in zip(itertools.combinations(atom_indices, 2), bonds):
				if bond_type != Chem.BondType.ZERO:
					mol.AddBond(i, j, bond_type)

		Chem.SanitizeMol(mol)
		
		# Final check for fragments which can occur after sanitization
		# May able to remove this, since we already check this 
		# with networkx in generating composition
		if len(Chem.rdmolops.GetMolFrags(mol)) > 1:
			 return None

		smi = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
		mol = Chem.AddHs(mol)
		atom_counts = dict(Counter(atom.GetSymbol() for atom in mol.GetAtoms()))
		
		return (smi, pickle.dumps((mol, atom_counts)))
	except Exception:
		return None

########################
# Chemical Space Class #
########################

@dataclass
class ChemicalSpace:
	max_atoms: int = field(default=4)
	atoms: FrozenSet[str] = field(default_factory=lambda: frozenset(["C", "H", "O"]))
	config: MolConfig = field(default_factory=lambda: MolConfig())

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
		logging.info(f"ChemicalSpace initialized for {self.atoms} up to {self.max_atoms} heavy atoms.")

	def generate_compositions(self):
		click.secho("--- Stage 1: Generating Plausible Compositions ---", bold=True)
		for n in range(1, self.max_atoms + 1):
			self._generate_compositions_for_n(n)

	def build_molecules(self):
		click.secho("\n--- Stage 2: Building Molecules from Compositions ---", bold=True)
		for n in range(1, self.max_atoms + 1):
			self._build_molecules_for_n(n)

	def explore(self):
		self.generate_compositions()
		self.build_molecules()
		click.secho(f"\nExploration complete. Results are in '{self.output_dir}'", fg='green')
		self.export_to_csv()

	def _lmdb_writer(self, q: Queue, db_path: str, batch_size: int = 4096):
		env = lmdb.open(
			db_path,
			map_size=10**11,
			writemap=True,
			metasync=False,
			sync=False,
		)
		batch = {}
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

	def _generate_compositions_for_n(self, n_atoms: int):
		"""
		Generates and saves plausible compositions for a specific number of heavy atoms.
		"""

		db_path = self._db_paths["compositions"].format(n=n_atoms)
		if os.path.exists(db_path):
			click.secho(f"Skipping N={n_atoms}, database already exists: {db_path}", fg='green')
			return
			
		click.secho(f"\nGenerating for N={n_atoms} heavy atoms...", bold=True)

		def task_generator():
			compositions = itertools.combinations_with_replacement(self._heavy_atoms, n_atoms)
			num_bonds = (n_atoms * (n_atoms - 1)) // 2
			for comp in compositions:
				for bonds in itertools.product(self.config.BOND_OPTIONS, repeat=num_bonds):
					yield (comp, bonds)

		total_tasks = (
			math.comb(len(self._heavy_atoms) + n_atoms - 1, n_atoms) * (len(self.config.BOND_OPTIONS) ** ((n_atoms * (n_atoms - 1)) // 2))
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
		env_out = lmdb.open(output_db_path, map_size=10**11, writemap=True, metasync=False, sync=False)

		input_db_path = self._db_paths["compositions"].format(n=n_atoms)
		if not os.path.exists(input_db_path):
			click.secho(f"Can't find composition database at {os.path.basename(input_db_path)}...", fg="red")
			return
		
		click.secho(f"Building from {os.path.basename(input_db_path)}...", bold=True)
		q = Queue(maxsize=self.n_workers * 2)
		writer = Process(target=self._lmdb_writer, args=(q, output_db_path))
		writer.start()

		env_in = lmdb.open(input_db_path, readonly=True, lock=False)
		total_tasks = env_in.stat()['entries']
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

		click.secho(f"\nTotal unique molecules found: {env_out.stat()['entries']:,}", fg='green')

	def export_to_csv(self, filename: str = "molecules.csv"):
		"""Exports the final molecules from the DB to a CSV file."""

		molecules_data = []
		for n in range(1, self.max_atoms + 1):
			output_db_path = self._db_paths["molecules"].format(n=n)
			output_csv_path = os.path.join(self.output_dir, filename)
			
			click.secho(f"Exporting molecules to {output_csv_path}...", bold=True)
			env = lmdb.open(output_db_path, readonly=True)
			
			with env.begin() as txn:
				for smi_key, value in tqdm(txn.cursor(), total=txn.stat()['entries'], desc="Reading DB"):
					smi = smi_key.decode()
					_, atom_counts = pickle.loads(value)
					molecules_data.append({"SMILES": smi, **atom_counts})
			
		if not molecules_data:
			click.secho("No molecules to export.", fg="red")
			return

		df = pd.DataFrame(molecules_data).fillna(0).astype(int, errors='ignore')

		# Reorder columns to have SMILES first
		cols = ["SMILES"] + [col for col in df.columns if col != "SMILES"]
		df = df[cols]
		
		df.to_csv(output_csv_path, index=False)
		click.secho(f"Export complete. There are {len(df)} molecules.", fg="green")

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
	# DisableLog("rdApp.*")

def common_options(f):
	@wraps(f)
	@click.option("-n", "--max-atoms", type=int, default=3, show_default=True, help="Maximum number of heavy atoms.")
	@click.option("-a", "--atoms", multiple=True, default=['C', 'O', 'H'], show_default=True, help="Heavy atoms to include.")
	@click.option("-w", "--workers", type=int, default=cpu_count(), show_default=True, help="Number of worker processes.")
	@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="chemical_space_results", show_default=True, help="Directory to save all results.")
	def wrapper(*args, **kwargs):
		return f(*args, **kwargs)

	return wrapper

@cli.command()
@common_options
def explore(max_atoms, atoms, workers, output_dir):
	"""Run the full workflow: generate, build, and export."""
	space = ChemicalSpace(
		atoms=frozenset(atoms),
		max_atoms=max_atoms,
		n_workers=workers,
		output_dir=output_dir
	)
	space.explore()

@cli.command()
@common_options
def generate_compositions(max_atoms, atoms, workers, output_dir):
	"""Stage 1: Generate and save plausible compositions."""
	space = ChemicalSpace(
		atoms=frozenset(atoms),
		max_atoms=max_atoms,
		n_workers=workers,
		output_dir=output_dir
	)
	space.generate_compositions()

@cli.command()
@common_options
def build_molecules(max_atoms, atoms, workers, output_dir):
	"""Stage 2: Build molecules from existing compositions."""
	space = ChemicalSpace(
		atoms=frozenset(atoms),
		max_atoms=max_atoms,
		n_workers=workers,
		output_dir=output_dir
	)
	space.build_molecules()

@cli.command()
@click.option("-n", "--max-atoms", type=int, default=3, show_default=True, help="Maximum number of heavy atoms to check for DBs.")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="chemical_space_results", show_default=True, help="Directory containing results.")
@click.option("-f", "--filename", type=str, default="molecules.csv", show_default=True, help="Output CSV filename.")
def export_csv(max_atoms, output_dir, filename):
	"""Stage 3: Export all found molecules to a single CSV file."""
	space = ChemicalSpace(max_atoms=max_atoms, output_dir=output_dir)
	space.export_to_csv(filename=filename)

def main():
	cli()

if __name__ == "__main__":
	from multiprocessing import set_start_method
	set_start_method("spawn")
	main()
