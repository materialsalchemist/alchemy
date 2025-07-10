import pandas as pd
import click
import itertools
from collections import Counter, defaultdict
import os
from multiprocessing import Pool, Process, Queue
from functools import partial
import lmdb
import pickle
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions
import logging
from dataclasses import dataclass, field
from os import cpu_count
from typing import List, Dict, Tuple, Optional
from functools import wraps
import networkx as nx
import json

# Suppress verbose RDKit logging
RDLogger.DisableLog('rdApp.*')

########################################
# Worker Functions for multi processes #
########################################

_molecule_list = []
_lookup_map = {}
_atom_order = []

def init_worker_find_candidates(molecules: List[Dict], lookup_map: Dict, atom_order: List[str]):
	"""Initializes globals for the find_candidates worker pool."""
	global _molecule_list, _lookup_map, _atom_order
	_molecule_list = molecules
	_lookup_map = lookup_map
	_atom_order = atom_order

def worker_find_candidates(reactant_indices: Tuple[int, int]) -> List[Tuple[str, str, str]]:
	global _molecule_list, _lookup_map, _atom_order
	idx1, idx2 = reactant_indices
	mol1 = _molecule_list[idx1]
	mol2 = _molecule_list[idx2]

	target_counts = tuple(mol1['counts'][atom] + mol2['counts'][atom] for atom in _atom_order)

	product_candidates_smiles = _lookup_map.get(target_counts, [])
	
	reactions = []
	for p_smiles in product_candidates_smiles:
		if p_smiles != mol1['SMILES'] and p_smiles != mol2['SMILES']:
			reactions.append((mol1['SMILES'], mol2['SMILES'], p_smiles))
			
	return reactions

def worker_verify_reaction(reaction_tuple_bytes: bytes) -> Optional[str]:
	try:
		r1_smi, r2_smi, p_smi = pickle.loads(reaction_tuple_bytes)
		
		if any(Chem.MolFromSmiles(s) is None for s in [r1_smi, r2_smi, p_smi]):
			return None

		m1 = Chem.MolFromSmiles(r1_smi)
		m2 = Chem.MolFromSmiles(r2_smi)
		p = Chem.MolFromSmiles(p_smi)
		
		r1_can = Chem.MolToSmiles(m1, canonical=True, allHsExplicit=True)
		r2_can = Chem.MolToSmiles(m2, canonical=True, allHsExplicit=True)
		p_can  = Chem.MolToSmiles(p,  canonical=True, allHsExplicit=True)
		reaction_smarts = f"{r1_can}.{r2_can}>>{p_can}"

		rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts, useSmiles=True)

		# rxn.Validate -> tuple(n_warnings, n_errors)
		if rxn.Validate(silent=False)[1] == 0:
			return reaction_smarts
		else:
			return None
			
	except Exception as e:
		print(e)
		return None

######################
# ReactionSpace Class #
######################

@dataclass
class ReactionSpace:
	input_csv: str
	output_dir: str = "reaction_space_results"
	n_workers: int = field(default_factory=cpu_count)
	
	_db_paths: Dict[str, str] = field(init=False, default_factory=dict)

	def __post_init__(self):
		os.makedirs(self.output_dir, exist_ok=True)
		db_dir = os.path.join(self.output_dir, "db")
		os.makedirs(db_dir, exist_ok=True)
		
		self._db_paths = {
			"candidates": os.path.join(db_dir, "reaction_candidates.lmdb"),
			"verified": os.path.join(db_dir, "verified_reactions.lmdb"),
		}
		logging.info(f"ReactionSpace initialized. Results will be saved in '{self.output_dir}'")

	def _lmdb_writer(self, q: Queue, db_path: str):
		"""A process that listens on a queue and writes data to an LMDB database."""
		env = lmdb.open(db_path, map_size=10**11, writemap=True, metasync=False, sync=False)
		count = 0
		with env.begin(write=True) as txn:
			while True:
				item = q.get()
				if item is None:
					break
				# Use a unique key for each entry
				key = f"{count:012d}".encode()
				txn.put(key, item, overwrite=False)
				count += 1

	def find_reaction_candidates(self):
		"""
		Find all combinations C_i + C_j = C_k
		based on atomic composition.
		"""
		click.secho("\n--- Step 1: Finding Reaction Candidates ---", bold=True)
		
		if not os.path.exists(self.input_csv):
			click.secho(f"Error: Input file not found at {self.input_csv}", fg="red")
			raise FileNotFoundError
		
		df = pd.read_csv(self.input_csv)
		atom_cols = [col for col in df.columns if col != 'SMILES']
		df[atom_cols] = df[atom_cols].fillna(0).astype(int)
		
		click.secho(f"Loaded {len(df)} molecules from {self.input_csv}", fg="green")

		molecules = []
		lookup_map = defaultdict(list)
		for _, row in df.iterrows():
			counts = row[atom_cols].to_dict()
			counts_tuple = tuple(counts[atom] for atom in atom_cols)
			molecules.append({'SMILES': row['SMILES'], 'counts': counts})
			lookup_map[counts_tuple].append(row['SMILES'])

		click.secho("Injecting [H] and [H][H] into the reactant pool.", fg="yellow")
		
		if 'H' not in atom_cols:
			click.secho("Warning: No 'H' column in input CSV. Cannot add H/[H]2 reactants.", fg="red")
		else:
			h_species_to_add = {'[H]': 1, '[H][H]': 2}
			
			for smi, h_count in h_species_to_add.items():
				if Chem.MolFromSmiles(smi) is None:
					logging.warning(f"Could not parse hydrogen species SMILES: {smi}. Skipping.")
					continue

				counts = {atom: 0 for atom in atom_cols}
				counts['H'] = h_count
				counts_tuple = tuple(counts[atom] for atom in atom_cols)

				molecules.append({'SMILES': smi, 'counts': counts})
				lookup_map[counts_tuple].append(smi)
		
		reactant_pairs = list(itertools.combinations_with_replacement(range(len(molecules)), 2))

		q = Queue(maxsize=self.n_workers * 2)
		writer = Process(target=self._lmdb_writer, args=(q, self._db_paths["candidates"]))
		writer.start()

		total_saved = 0
		with Pool(processes=self.n_workers, initializer=init_worker_find_candidates, initargs=(molecules, lookup_map, atom_cols)) as pool:
			for result_list in tqdm(
				pool.imap_unordered(worker_find_candidates, reactant_pairs, chunksize=1024),
				total=len(reactant_pairs),
				desc="Finding Candidates"
			):
				if result_list:
					for reaction_tuple in result_list:
						q.put(pickle.dumps(reaction_tuple))
						total_saved += 1
		
		q.put(None)
		writer.join()
		click.secho(f"Found and saved {total_saved:,} potential reaction candidates.", fg="green")

	def verify_reactions(self):
		"""
		Step 2: Read candidate reactions from the database and verify them using RDKit.
		"""
		click.secho("\n--- Step 2: Verifying Reactions with RDKit ---", bold=True)
		
		input_db_path = self._db_paths["candidates"]
		if not os.path.exists(input_db_path):
			click.secho(f"Candidate database not found. Please run 'find-candidates' first.", fg="red")
			return

		env_in = lmdb.open(input_db_path, readonly=True, lock=False)
		total_tasks = env_in.stat()['entries']

		if total_tasks == 0:
			click.secho("No candidates to verify.", fg="yellow")
			return

		q = Queue(maxsize=self.n_workers * 2)
		verified_db_path = self._db_paths["verified"]
		writer_env = lmdb.open(verified_db_path, map_size=10**11, writemap=True)

		with env_in.begin() as txn, Pool(self.n_workers) as pool, writer_env.begin(write=True) as writer_txn:
			cursor = txn.cursor()
			vals = (value for _, value in cursor)
			
			total_verified = 0
			for result in tqdm(pool.imap_unordered(worker_verify_reaction, vals, chunksize=1024), total=total_tasks, desc="Verifying Reactions"):
				if result:
					key = result.encode()
					writer_txn.put(key, b'1', overwrite=False)
					total_verified += 1
		
		click.secho(f"Verified {writer_env.stat()['entries']:,} unique, chemically plausible reactions.", fg="green")


	def export_to_csv(self, filename: str = "reactions.csv"):
		"""Exports the final verified reactions from the DB to a CSV file."""
		click.secho("\n--- Step 3a: Exporting Verified Reactions to CSV ---", bold=True)
		
		db_path = self._db_paths["verified"]
		output_csv_path = os.path.join(self.output_dir, filename)

		if not os.path.exists(db_path):
			click.secho("Verified reactions database not found. Nothing to export.", fg="red")
			return
			
		env = lmdb.open(db_path, readonly=True)
		num_reactions = env.stat()['entries']

		if num_reactions == 0:
			click.secho("No verified reactions to export.", fg="yellow")
			return

		reaction_data = []
		with env.begin() as txn:
			for key, _ in tqdm(txn.cursor(), total=num_reactions, desc="Exporting to CSV"):
				reaction_smarts = key.decode()
				reactants, _, product = reaction_smarts.partition('>>')
				r1, r2 = reactants.split(".", 1)
				reaction_data.append({"Reactant1": r1, "Reactant2": r2, "Product": product})

		df = pd.DataFrame(reaction_data)
		df.to_csv(output_csv_path, index=False)
		click.secho(f"Successfully exported {len(df)} reactions to {output_csv_path}", fg="green")

	def generate_reaction_network_graph(self, filename: str = "reaction_network.json"):
		"""Generates a NetworkX graph from verified reactions and saves it as JSON."""
		click.secho("\n--- Step 3b: Generating Reaction Network Graph (JSON) ---", bold=True)

		db_path = self._db_paths["verified"]
		output_json_path = os.path.join(self.output_dir, filename)

		if not os.path.exists(db_path):
			click.secho("Verified reactions database not found. Cannot generate graph.", fg="red")
			return

		env = lmdb.open(db_path, readonly=True, lock=False)
		num_reactions = env.stat()['entries']

		if num_reactions == 0:
			click.secho("No verified reactions to build graph from.", fg="yellow")
			return

		G = nx.DiGraph()
		with env.begin() as txn:
			for key, _ in tqdm(txn.cursor(), total=num_reactions, desc="Building graph"):
				reaction_smarts = key.decode()
				reactants_smarts, _, product_smarts = reaction_smarts.partition('>>')
				
				G.add_node(reaction_smarts, type='reaction')

				reactant_list = reactants_smarts.split('.')
				for r_smi in reactant_list:
					if r_smi:
						G.add_node(r_smi, type='molecule')
						G.add_edge(r_smi, reaction_smarts)
				
				if product_smarts:
					G.add_node(product_smarts, type='molecule')
					G.add_edge(reaction_smarts, product_smarts)
		
		graph_data = nx.node_link_data(G, edges="links")
		with open(output_json_path, 'w') as f:
			json.dump(graph_data, f, indent=2)
			
		click.secho(f"Successfully generated and saved reaction network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to {output_json_path}", fg="green")


	def explore(self):
		"""Run the full workflow: find candidates, verify, export CSV, and export graph."""
		self.find_reaction_candidates()
		self.verify_reactions()
		self.export_to_csv()
		self.generate_reaction_network_graph()

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def common_options(func):
	@wraps(func)
	@click.option(
		"-i", "--input-csv", type=click.Path(exists=True, dir_okay=False),
	   default="chemical_space_results/molecules.csv", show_default=True,
	   help="Input CSV file with molecules and atom counts.",
	 )
	@click.option(
		"-o", "--output-dir", type=click.Path(file_okay=False),
		default="reaction_space_results", show_default=True,
		help="Directory to save all results."
	)
	@click.option(
		"-w", "--workers", type=int, default=cpu_count(), show_default=True,
		help="Number of worker processes."
	)

	def wrapper(*args, **kwargs):
		return func(*args, **kwargs)

	return wrapper

@cli.command()
@common_options
def explore(input_csv, output_dir, workers):
	"""Run the full workflow: find, verify, and export reactions."""
	space = ReactionSpace(input_csv=input_csv, output_dir=output_dir, n_workers=workers)
	space.explore()

@cli.command()
@common_options
def find_candidates(input_csv, output_dir, workers):
	"""Step 1: Find reaction candidates based on atom conservation."""
	space = ReactionSpace(input_csv=input_csv, output_dir=output_dir, n_workers=workers)
	space.find_reaction_candidates()

@cli.command()
@common_options
def verify_reactions(input_csv, output_dir, workers):
	"""Step 2: Verify candidates from DB using RDKit."""
	space = ReactionSpace(input_csv=input_csv, output_dir=output_dir, n_workers=workers)
	space.verify_reactions()

@cli.command()
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="reaction_space_results", show_default=True, help="Directory containing the reaction databases.")
@click.option("-f", "--filename", type=str, default="reactions.csv", show_default=True, help="Output CSV filename.")
def export_csv(output_dir, filename):
	"""Step 3a: Export verified reactions to a single CSV file."""
	space = ReactionSpace(input_csv="", output_dir=output_dir) # input_csv not needed for export
	space.export_to_csv(filename=filename)

@cli.command()
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="reaction_space_results", show_default=True, help="Directory containing the reaction databases.")
@click.option("-f", "--filename", type=str, default="reaction_network.json", show_default=True, help="Output JSON filename for the graph.")
def export_graph(output_dir, filename):
	"""Step 3b: Generate and save the reaction network as a JSON file."""
	space = ReactionSpace(input_csv="", output_dir=output_dir) # input_csv not needed for export
	space.generate_reaction_network_graph(filename=filename)

def main():
	cli()

if __name__ == "__main__":
	from multiprocessing import set_start_method
	set_start_method("spawn")
	main()
