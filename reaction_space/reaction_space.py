import pandas as pd
import click
import itertools
import os
from multiprocessing import Pool, Process, Queue
from functools import partial
import lmdb
from tqdm import tqdm
import logging
from dataclasses import dataclass, field
from os import cpu_count
from typing import List, Dict, Tuple
import networkx as nx
import json
import math

from .workers import (
	get_dissociation_fragments,
	worker_generate_new_reactions_g1,
	worker_generate_higher_gen_reactions,
	worker_verify_reaction_batch,
)

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
			"candidates_g0": os.path.join(db_dir, "reaction_candidates_g0.lmdb"),
			"candidates_g1": os.path.join(db_dir, "reaction_candidates_g1.lmdb"),
			"candidates_g2plus": os.path.join(db_dir, "reaction_candidates_g{gen}_c{max_c}.lmdb"),
			"verified": os.path.join(db_dir, "verified_reactions.lmdb"),
		}
		logging.info(f"ReactionSpace initialized. Results will be saved in '{self.output_dir}'")

	def _lmdb_writer(self, q: Queue, db_path: str, batch_size: int = 4096):
		"""A process that listens on a queue and writes data to an LMDB database in batches."""
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

	def _lmdb_batch_iterator(self, db_paths: List[str], batch_size: int):
		"""
		A generator that streams keys from multiple LMDB databases and yields them in batches.
		"""
		batch = []

		seen_keys = set()
		for db_path in db_paths:
			if not os.path.exists(db_path):
				continue
			
			env = lmdb.open(db_path, readonly=True, lock=False)
			with env.begin() as txn:
				cursor = txn.cursor()
				for key, _ in cursor:
					if key not in seen_keys:
						seen_keys.add(key)
						batch.append(key)

						if len(batch) >= batch_size:
							yield batch
							batch = []

			env.close()
		
		if batch:
			yield batch

	def find_reaction_candidates(self):
		"""
		Generates reaction candidates using hierarchical generation (G0, G1, G2+)
		and writes them to generation-specific LMDB databases.
		"""
		click.secho("\n--- Starting Hierarchical Reaction Network Generation ---", bold=True)

		if not os.path.exists(self.input_csv):
			click.secho(f"Error: Input file not found at {self.input_csv}", fg="red")
			raise FileNotFoundError

		df = pd.read_csv(self.input_csv)
		initial_molecules = df['SMILES'].tolist()
		click.secho(f"Loaded {len(initial_molecules)} molecules from {self.input_csv}", fg="green")

		all_reactions_written = set()
		
		def read_keys_from_db(db_path):
			if not os.path.exists(db_path): return []
			env = lmdb.open(db_path, readonly=True, lock=False)
			with env.begin() as txn:

				keys = [key.decode('utf-8') for key, _ in txn.cursor()]

			env.close()
			return keys

		# --- Generation 0: Initial Dissociations ---
		click.secho("\n--- G0: Performing Initial Bond Dissociations ---", bold=True)
		g0_db_path = self._db_paths["candidates_g0"]
		if os.path.exists(g0_db_path):
			click.secho(f"Skipping G0, database already exists.", fg='green')
			g0_reactions = read_keys_from_db(g0_db_path)
		else:
			q = Queue(maxsize=self.n_workers * 2)
			writer = Process(target=self._lmdb_writer, args=(q, g0_db_path))
			writer.start()

			g0_reactions = []
			with Pool(self.n_workers) as pool:
				results_iterator = pool.imap(get_dissociation_fragments, initial_molecules)

				for parent_smi, frag_set in tqdm(zip(initial_molecules, results_iterator), total=len(initial_molecules), desc="G0: Dissociating"):
					for f1, f2 in frag_set:
						reaction_smi = f"{parent_smi}>>{f1}.{f2}"

						if reaction_smi not in all_reactions_written:
							all_reactions_written.add(reaction_smi)
							g0_reactions.append(reaction_smi)
							q.put((reaction_smi.encode('utf-8'), b'G0'))

			q.put(None)
			writer.join()

		click.secho(f"Found {len(g0_reactions):,} unique G0 reactions.", fg="green")
		all_reactions_written.update(g0_reactions)
		current_generation_reactions = g0_reactions

		# --- Generation 1: Transfer and Rearrangement Reactions ---
		if self.num_generations >= 1:
			click.secho(f"\n--- G1: Generating Transfer and Rearrangement Reactions ---", bold=True)
			g1_db_path = self._db_paths["candidates_g1"]
			if os.path.exists(g1_db_path):
				click.secho(f"Skipping G1, database already exists.", fg='green')
				g1_reactions = read_keys_from_db(g1_db_path)
			else:
				if len(current_generation_reactions) < 2:
					 click.secho("Not enough G0 reactions to generate G1. Skipping.", fg="yellow")
					 g1_reactions = []
				else:
					reaction_pairs = itertools.combinations(current_generation_reactions, 2)
					total_pairs = math.comb(len(current_generation_reactions), 2)
					
					q = Queue(maxsize=self.n_workers * 2)
					writer = Process(target=self._lmdb_writer, args=(q, g1_db_path))
					writer.start()

					g1_reactions = []
					with Pool(self.n_workers) as pool:
						results_iterator = pool.imap_unordered(worker_generate_new_reactions_g1, reaction_pairs, chunksize=1024)

						for res_list in tqdm(results_iterator, total=total_pairs, desc="G1: Processing"):
							for candidate in res_list:
								if candidate not in all_reactions_written:
									all_reactions_written.add(candidate)
									g1_reactions.append(candidate)
									q.put((candidate.encode('utf-8'), b'G1'))

					q.put(None)
					writer.join()

			click.secho(f"Found {len(g1_reactions):,} unique G1 reactions.", fg="green")
			all_reactions_written.update(g1_reactions)
			current_generation_reactions = g1_reactions

		# --- Generation 2+: Higher-order reactions ---
		for gen in range(2, self.num_generations + 1):
			max_c = gen + 1

			click.secho(f"\n--- G{gen}: Generating Higher-order Reactions (max_complexity={self.max_reaction_complexity}) ---", bold=True)
			g2plus_db_path = self._db_paths["candidates_g2plus"].format(gen=gen, max_c=min(max_c, self.max_reaction_complexity))

			if os.path.exists(g2plus_db_path):
				click.secho(f"Skipping G{gen}, database already exists.", fg='green')
				next_gen_reactions = read_keys_from_db(g2plus_db_path)
			else:
				if len(current_generation_reactions) < 1:
					click.secho(f"Not enough reactions from previous generation to generate G{gen}. Stopping.", fg="yellow")
					break
				
				previous_reactions = list(all_reactions_written - set(current_generation_reactions))
				if not previous_reactions:
					click.secho(f"No previous reactions to combine with for G{gen}. Stopping.", fg="yellow")
					break
				
				reaction_pairs = itertools.product(current_generation_reactions, previous_reactions)
				total_pairs = len(current_generation_reactions) * len(previous_reactions)
				
				q = Queue(maxsize=self.n_workers * 2)
				writer = Process(target=self._lmdb_writer, args=(q, g2plus_db_path))
				writer.start()

				next_gen_reactions = []
				with Pool(self.n_workers) as pool:
					partial_worker = partial(worker_generate_higher_gen_reactions, max_reaction_complexity=self.max_reaction_complexity)
					results_iterator = pool.imap_unordered(partial_worker, reaction_pairs, chunksize=1024)

					for res_list in tqdm(results_iterator, total=total_pairs, desc=f"G{gen}: Processing"):
						for candidate in res_list:
							if candidate not in all_reactions_written:
								all_reactions_written.add(candidate)
								next_gen_reactions.append(candidate)
								q.put((candidate.encode('utf-8'), f'G{gen}'.encode()))

				q.put(None)
				writer.join()

			click.secho(f"Found {len(next_gen_reactions):,} unique G{gen} reactions.", fg="green")
			if not next_gen_reactions:
				click.secho(f"No new reactions generated at G{gen}. Stopping.", fg="yellow")
				break

			all_reactions_written.update(next_gen_reactions)
			current_generation_reactions = next_gen_reactions

		click.secho(f"\n--- Finalizing ---", bold=True)
		click.secho(f"Total unique reaction candidates considered: {len(all_reactions_written):,}")
		click.secho("Hierarchical reaction network generation complete.", fg="green")

	def verify_reactions(self):
		click.secho("\n--- Step 2: Verifying Reactions with RXNMapper ---", bold=True)
		
		candidate_db_paths = []

		g0_path = self._db_paths["candidates_g0"]
		if os.path.exists(g0_path):
			candidate_db_paths.append(g0_path)

		if self.num_generations >= 1:
			g1_path = self._db_paths["candidates_g1"]
			if os.path.exists(g1_path):
				candidate_db_paths.append(g1_path)

		for gen in range(2, self.num_generations + 1):
			max_c = gen + 1
			db_path = self._db_paths["candidates_g2plus"].format(gen=gen, max_c=min(max_c, self.max_reaction_complexity))
			if os.path.exists(db_path):
				candidate_db_paths.append(db_path)

		total_candidates = 0
		for db_path in candidate_db_paths:
			if os.path.exists(db_path):
				env = lmdb.open(db_path, readonly=True, lock=False)
				total_candidates += env.stat()['entries']
				env.close()

		if total_candidates == 0:
			click.secho("No candidates to verify. Please run 'find-candidates' first with the appropriate settings.", fg="yellow")
			return
		
		click.secho(f"Found {total_candidates:,} total candidates to verify from specified databases.", fg="green")
		
		batch_size = 256
		batch_generator = self._lmdb_batch_iterator(candidate_db_paths, batch_size)
		total_batches = math.ceil(total_candidates / batch_size)

		writer_env = lmdb.open(self._db_paths["verified"], map_size=10**11, writemap=True)

		with writer_env.begin(write=True) as txn_out, Pool(self.n_workers) as pool:
			for verified_batch in tqdm(pool.imap_unordered(worker_verify_reaction_batch, batch_generator), total=total_batches, desc="Verifying Batches"):
				for result_smi in verified_batch:
					if result_smi:
						key = result_smi.encode('utf-8')
						txn_out.put(key, b'', overwrite=False)
		
		click.secho(f"Verified and saved {writer_env.stat()['entries']:,} unique, chemically plausible reactions.", fg="green")
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
