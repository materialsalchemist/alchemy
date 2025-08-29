import pandas as pd
import click
import itertools
from collections import Counter, defaultdict
import os
from multiprocessing import Pool, Process, Queue, Manager
from functools import partial, wraps
import lmdb
import pickle
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions
import logging
from dataclasses import dataclass, field
from os import cpu_count
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='rxnmapper')
from rxnmapper import RXNMapper, BatchedMapper

# Suppress verbose RDKit logging
RDLogger.DisableLog('rdApp.*')

########################################
# Worker Functions for multi processes #
########################################

def get_dissociation_fragments(mol_smiles: str) -> Set[Tuple[str, str]]:
	"""
	Dissociates a molecule by breaking each bond once to generate G0 reaction fragments.
	"""

	try:
		mol = Chem.MolFromSmiles(mol_smiles)
		if not mol:
			return set()

		mol = Chem.AddHs(mol)
	except Exception:
		return set()

	fragments = set()
	bonds = mol.GetBonds()
	
	for i in range(len(bonds)):
		emol = Chem.EditableMol(mol)
		
		begin_atom_idx = bonds[i].GetBeginAtomIdx()
		end_atom_idx = bonds[i].GetEndAtomIdx()
		
		emol.RemoveBond(begin_atom_idx, end_atom_idx)
		
		fragmented_mol = emol.GetMol()
		
		num_frags = len(Chem.GetMolFrags(fragmented_mol))
		
		if num_frags == 2:
			try:
				frag_mols = Chem.GetMolFrags(fragmented_mol, asMols=True)

				f1_smi = Chem.MolToSmiles(frag_mols[0], canonical=True)
				f2_smi = Chem.MolToSmiles(frag_mols[1], canonical=True)
				
				sorted_frags = tuple(sorted((f1_smi, f2_smi)))
				fragments.add(sorted_frags)
			except Exception:
				continue
					
	return fragments

def worker_generate_new_reactions_g1(reaction_pair: Tuple[str, str]) -> List[str]:
	"""
	Worker function to generate G1 reactions from pairs of G0 reactions.
	Implements both transfer and rearrangement reactions as described in the paper.
	"""
	r1_str, r2_str = reaction_pair

	try:
		# Parse reactions: "Parent>>Frag1.Frag2"
		p1, frags1_str = r1_str.split(">>")
		p2, frags2_str = r2_str.split(">>")
		
		# Split fragments
		frags1 = sorted(frags1_str.split('.'))
		frags2 = sorted(frags2_str.split('.'))
		
		new_reactions = []
		
		# Find common fragments between the two reactions
		common_frags = set(frags1) & set(frags2)
		
		if len(common_frags) == 0:
			return []

		# Case 1: Rearrangement reaction A -> D
		# Where A -> B + C and D -> B + C (same products)
		if p1 != p2 and frags1 == frags2:
			parents = sorted([p1, p2])
			reaction_smi = f"{parents[0]}>>{parents[1]}"
			
			return [reaction_smi]

		# Case 2: Transfer reaction A + E -> D + C
		# Where A -> B + C and D -> B + E, B is common
		if len(common_frags) == 1:
			common_frag = common_frags.pop()
			
			unique_frag1 = next(f for f in frags1 if f != common_frag)
			unique_frag2 = next(f for f in frags2 if f != common_frag)

			reactants = sorted([p1, unique_frag2])
			products = sorted([p2, unique_frag1])

			if reactants != products:
				reactants_str = '.'.join(reactants)
				products_str = '.'.join(products)
				reaction_smi = f"{reactants_str}>>{products_str}"

				return [reaction_smi]
		
		return new_reactions
		
	except Exception:
		return []

def worker_generate_higher_gen_reactions(reaction_pair: Tuple[str, str], max_reaction_complexity: int = 3) -> List[str]:
	"""
	Worker function to generate higher generation reactions (G2+) from any two reactions.
	"""
	r1_str, r2_str = reaction_pair
	
	try:
		r1_reactants_str, r1_products_str = r1_str.split(">>")
		r2_reactants_str, r2_products_str = r2_str.split(">>")
		
		r1_reactants = set(r1_reactants_str.split('.'))
		r1_products = set(r1_products_str.split('.'))
		r2_reactants = set(r2_reactants_str.split('.'))
		r2_products = set(r2_products_str.split('.'))
		
		new_reactions = []
		
		# Case 1: Product of r1 is reactant of r2
		shared_r1p_r2r = r1_products & r2_reactants
		if shared_r1p_r2r:
			for shared in shared_r1p_r2r:
				new_reactants = (r1_reactants | (r2_reactants - {shared}))
				new_products = ((r1_products - {shared}) | r2_products)
				
				if new_reactants != new_products and len(new_reactants) <= max_reaction_complexity and len(new_products) <= max_reaction_complexity:
					reactants_str = '.'.join(sorted(new_reactants))
					products_str = '.'.join(sorted(new_products))
					reaction_smi = f"{reactants_str}>>{products_str}"

					new_reactions.append(reaction_smi)
		
		# Case 2: Product of r2 is reactant of r1
		shared_r2p_r1r = r2_products & r1_reactants
		if shared_r2p_r1r:
			for shared in shared_r2p_r1r:
				new_reactants = (r2_reactants | (r1_reactants - {shared}))
				new_products = ((r2_products - {shared}) | r1_products)
				
				if new_reactants != new_products and len(new_reactants) <= max_reaction_complexity and len(new_products) <= max_reaction_complexity:
					reactants_str = '.'.join(sorted(new_reactants))
					products_str = '.'.join(sorted(new_products))
					reaction_smi = f"{reactants_str}>>{products_str}"

					new_reactions.append(reaction_smi)
		
		# Case 3: Reactants share species (parallel reactions with common reactant)
		shared_reactants = r1_reactants & r2_reactants
		if shared_reactants:
			for shared in shared_reactants:
				new_reactants = (r1_reactants - {shared}) | (r2_reactants - {shared}) | {shared}
				new_products = r1_products | r2_products
				
				if new_reactants != new_products and len(new_reactants) <= max_reaction_complexity and len(new_products) <= max_reaction_complexity:
					reactants_str = '.'.join(sorted(new_reactants))
					products_str = '.'.join(sorted(new_products))
					reaction_smi = f"{reactants_str}>>{products_str}"

					new_reactions.append(reaction_smi)
		
		return new_reactions
		
	except Exception:
		return []

def worker_verify_reaction(reaction_smi_bytes: bytes, confidence_threshold: float = 1.0) -> List[str]:
	rxn_mapper = RXNMapper()

	try:
		reaction_smi = reaction_smi_bytes.decode('utf-8')
		reactants_str, products_str = reaction_smi.split('>>')

		reactant_smis = reactants_str.split('.')
		product_smis = products_str.split('.')

		all_smis = reactant_smis + product_smis
		if any(not s or Chem.MolFromSmiles(s) is None for s in all_smis):
			return None

		reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smis]
		product_mols = [Chem.MolFromSmiles(s) for s in product_smis]

		r_cans = sorted([Chem.MolToSmiles(m, canonical=True) for m in reactant_mols])
		p_cans = sorted([Chem.MolToSmiles(m, canonical=True) for m in product_mols])

		if r_cans == p_cans:
			return None

		final_canonical_reaction = f"{'.'.join(r_cans)}>>{'.'.join(p_cans)}"

	except Exception:
		return None

	try:
		result = rxn_mapper.get_attention_guided_atom_maps([reaction_smi])[0]

		if result and result.get('confidence', 0.0) >= confidence_threshold:
			return final_canonical_reaction
		
		return None

	except Exception as e:
		logging.warning(f"rxnmapper failed for a batch: {e}")


######################
# ReactionSpace Class #
######################

@dataclass
class ReactionSpace:
	input_csv: str
	output_dir: str = "reaction_space_results"
	n_workers: int = field(default_factory=cpu_count)
	num_generations: int = 2
	max_reaction_complexity: int = 3
	
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
		Generates reaction candidates using hierarchical generation (G0, G1, G2+) 
		and writes them directly to an LMDB database.
		"""
		click.secho("\n--- Starting Hierarchical Reaction Network Generation ---", bold=True)

		if not os.path.exists(self.input_csv):
			click.secho(f"Error: Input file not found at {self.input_csv}", fg="red")
			raise FileNotFoundError

		df = pd.read_csv(self.input_csv)
		initial_molecules = df['SMILES'].tolist()
		click.secho(f"Loaded {len(initial_molecules)} molecules from {self.input_csv}", fg="green")

		env = lmdb.open(self._db_paths["candidates"], map_size=10**11, writemap=True)
		
		all_reactions_written = set()
		current_generation_reactions = []

		try:
			with env.begin(write=True) as txn:
				# --- Generation 0: Initial Dissociations ---
				click.secho("\n--- G0: Performing Initial Bond Dissociations ---", bold=True)
				g0_reactions = []
				with Pool(self.n_workers) as pool:
					results = list(tqdm(
						pool.imap(get_dissociation_fragments, initial_molecules),
						total=len(initial_molecules),
						desc="G0: Dissociating"
					))

				for parent_smi, frag_set in zip(initial_molecules, results):
					for f1, f2 in frag_set:
						reaction_smi = f"{parent_smi}>>{f1}.{f2}"
						if reaction_smi not in all_reactions_written:
							all_reactions_written.add(reaction_smi)
							g0_reactions.append(reaction_smi)
							txn.put(reaction_smi.encode('utf-8'), b'G0', overwrite=False)
				
				click.secho(f"Generated and stored {len(g0_reactions):,} unique G0 reactions.", fg="green")
				current_generation_reactions = g0_reactions

				# --- Generation 1: Transfer and Rearrangement Reactions ---
				if self.num_generations >= 1 and len(current_generation_reactions) >= 2:
					click.secho(f"\n--- G1: Generating Transfer and Rearrangement Reactions ---", bold=True)
					
					reaction_pairs = list(itertools.combinations(current_generation_reactions, 2))
					g1_reactions = []

					with Pool(self.n_workers) as pool:
						results_iterator = pool.imap_unordered(worker_generate_new_reactions_g1, reaction_pairs, chunksize=1024)
						
						for res_list in tqdm(results_iterator, total=len(reaction_pairs), desc="G1: Processing"):
							for candidate in res_list:
								if candidate not in all_reactions_written:
									all_reactions_written.add(candidate)
									g1_reactions.append(candidate)
									txn.put(candidate.encode('utf-8'), b'G1', overwrite=False)
					
					click.secho(f"Generated and stored {len(g1_reactions):,} unique G1 reactions.", fg="green")
					current_generation_reactions = g1_reactions

				# --- Generation 2+: Higher-order reactions ---
				for gen in range(2, self.num_generations + 1):
					if len(current_generation_reactions) < 2:
						click.secho(f"Not enough reactions to generate G{gen}. Stopping.", fg="yellow")
						break
					
					click.secho(f"\n--- G{gen}: Generating Higher-order Reactions ---", bold=True)
					
					previous_reactions = all_reactions_written - set(current_generation_reactions)

					reaction_pairs = itertools.product(current_generation_reactions, previous_reactions)
					total_pairs = len(current_generation_reactions) * len(previous_reactions)

					next_gen_reactions = []
					
					with Pool(self.n_workers) as pool:
						partial_worker = partial(
							worker_generate_higher_gen_reactions,
							max_reaction_complexity=self.max_reaction_complexity
						)

						results_iterator = pool.imap_unordered(partial_worker, reaction_pairs, chunksize=1024)
						
						for res_list in tqdm(results_iterator, total=total_pairs, desc=f"G{gen}: Processing"):
							for candidate in res_list:
								if candidate not in all_reactions_written:

									reactants = candidate.split('>>')[0].split('.')
									products = candidate.split('>>')[1].split('.')

									all_reactions_written.add(candidate)
									next_gen_reactions.append(candidate)
									txn.put(candidate.encode('utf-8'), f'G{gen}'.encode(), overwrite=False)
					
					click.secho(f"Generated and stored {len(next_gen_reactions):,} unique G{gen} reactions.", fg="green")
					current_generation_reactions = next_gen_reactions
					
					if len(next_gen_reactions) == 0:
						click.secho(f"No new reactions generated at G{gen}. Stopping.", fg="yellow")
						break

		finally:
			env.close()

		click.secho(f"\n--- Finalizing ---", bold=True)
		click.secho(f"Total unique reactions written to database: {len(all_reactions_written):,}")
		click.secho("Hierarchical reaction network generation complete.", fg="green")

	def verify_reactions(self):
		click.secho("\n--- Step 2: Verifying Reactions with RDKit ---", bold=True)
		
		input_db_path = self._db_paths["candidates"]
		if not os.path.exists(input_db_path):
			click.secho(f"Candidate database not found. Please run 'find-candidates' first.", fg="red")
			return

		env_in = lmdb.open(input_db_path, readonly=True, lock=False)
		total_tasks = env_in.stat()['entries']

		if total_tasks == 0:
			click.secho("No candidates to verify.", fg="yellow")
			env_in.close()
			return

		writer_env = lmdb.open(self._db_paths["verified"], map_size=10**11, writemap=True)

		with env_in.begin() as txn_in, writer_env.begin(write=True) as txn_out, Pool(self.n_workers) as pool:
			cursor = txn_in.cursor()
			reaction_keys = (key for key, _ in cursor)
			
			for result in tqdm(pool.imap_unordered(worker_verify_reaction, reaction_keys, chunksize=1024), total=total_tasks, desc="Verifying Reactions"):
				if result:
					key = result.encode('utf-8')
					txn_out.put(key, b'', overwrite=False)
		
		click.secho(f"Verified {writer_env.stat()['entries']:,} unique, chemically plausible reactions.", fg="green")
		env_in.close()
		writer_env.close()

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
			env.close()
			return

		reaction_data = []
		with env.begin() as txn:
			for key, _ in tqdm(txn.cursor(), total=num_reactions, desc="Exporting to CSV"):
				reaction_smarts = key.decode()
				reactants, products = reaction_smarts.split('>>')
				reaction_data.append({"Reactants": reactants, "Products": products})

		df = pd.DataFrame(reaction_data)
		df.to_csv(output_csv_path, index=False)
		click.secho(f"Successfully exported {len(df)} reactions to {output_csv_path}", fg="green")
		env.close()


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
	@click.option(
		"-g", "--generations", type=int, default=2, show_default=True,
		help="Number of bimolecular reaction generations to run."
	)
	@click.option(
		"-c", "--max-complexity", type=int, default=3, show_default=True,
		help="Maximum number of reactants/products for higher generation reactions."
	)

	def wrapper(*args, **kwargs):
		return func(*args, **kwargs)

	return wrapper

@cli.command()
@common_options
def explore(input_csv, output_dir, workers, generations, max_complexity):
	"""Run the full workflow: find candidates, verify, and export results."""
	space = ReactionSpace(
		input_csv=input_csv, 
		output_dir=output_dir, 
		n_workers=workers,
		num_generations=generations,
		max_reaction_complexity=max_complexity,
	)
	space.explore()

@cli.command()
@common_options
def find_candidates(input_csv, output_dir, workers, generations, max_complexity):
	"""Step 1: Generate reaction candidates from initial molecules."""
	space = ReactionSpace(
		input_csv=input_csv, 
		output_dir=output_dir, 
		n_workers=workers,
		num_generations=generations,
		max_reaction_complexity=max_complexity,
	)
	space.find_reaction_candidates()

@cli.command()
@common_options
def verify_reactions(input_csv, output_dir, workers, generations, max_complexity):
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
