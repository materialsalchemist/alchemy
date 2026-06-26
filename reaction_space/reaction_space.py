import hashlib
import pandas as pd
import click
import itertools
import os
import collections
from multiprocessing import Pool, Process, Queue
from functools import partial
import lmdb
from tqdm import tqdm

import logging
from dataclasses import dataclass, field
from os import cpu_count
from typing import List, Dict, Tuple, Iterator
import networkx as nx
import json
import math
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from reaction_space.utils import element_counts

from .workers import (
	get_dissociation_fragments,
	worker_systematic_recombination,
	worker_radical_addition,
	worker_generate_new_reactions_g1,
	worker_generate_higher_gen_reactions,
	worker_verify_reaction_batch,
)

from .utils import canonicalize_smiles, canonicalize_smiles_list, chunked_iterable, RADICAL_THRESHOLD


@dataclass
class ReactionSpace:
	input_csv: str
	custom_reactants_csv: str = None
	output_dir: str = "reaction_space_results"
	n_workers: int = field(default_factory=cpu_count)
	num_generations: int = 2
	max_reaction_complexity: int = 3
	radical_threshold: int = RADICAL_THRESHOLD
	require_custom_reactant: bool = False

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

	def _get_reaction_hash(self, smi: str) -> bytes:
		"""Returns a compact hash of the reaction SMILES for duplicate tracking."""
		return hashlib.blake2b(smi.encode("utf-8"), digest_size=12).digest()

	def _lmdb_writer(self, q: Queue, db_path: str, batch_size: int = 4096):
		"""A process that listens on a queue and writes data to an LMDB database in batches."""
		env = lmdb.open(
			db_path,
			map_size=10**12,
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
		env.close()

	def _iter_smiles_from_db(self, db_path: str) -> Iterator[str]:
		"""Yields reaction SMILES from an LMDB database one by one."""
		if not os.path.exists(db_path):
			return
		env = lmdb.open(db_path, readonly=True, lock=False)
		with env.begin() as txn:
			cursor = txn.cursor()
			for _, value in cursor:
				try:
					value_data = json.loads(value.decode("utf-8"))
					yield value_data["smi"]
				except (json.JSONDecodeError, KeyError, Exception):
					continue
		env.close()

	def _lmdb_batch_iterator(self, db_paths: List[str], batch_size: int, custom_reactants_filter: set = None):
		"""
		A generator that streams keys from multiple LMDB databases and yields them in batches.
		"""
		batch = []

		def __should_keep_reaction(smi: str, filter_set: set) -> bool:
			if not filter_set:
				return True

			try:
				reactants_str, products_str = smi.split(">>")
				reactants = {r for r in reactants_str.split(".") if r}
				products = {p for p in products_str.split(".") if p}

				has_custom_in_reactants = not filter_set.isdisjoint(reactants)
				has_custom_in_products = not filter_set.isdisjoint(products)

				# XOR condition: Keep if custom reactant is in reactants OR products, but NOT both.
				return has_custom_in_reactants ^ has_custom_in_products
			except (Exception, ValueError, AttributeError) as e:
				logging.error(f"Batch Iterator error: {e}")
				return False

		for db_path in db_paths:
			if not os.path.exists(db_path):
				continue

			try:
				env = lmdb.open(db_path, readonly=True, lock=False)
				with env.begin() as txn:
					cursor = txn.cursor()
					for _, value in cursor:
						try:
							value_data = json.loads(value.decode("utf-8"))
							smi = value_data["smi"]
							gen = value_data["gen"]

							if custom_reactants_filter and not __should_keep_reaction(smi, custom_reactants_filter):
								continue

							batch.append((smi, gen))

							if len(batch) >= batch_size:
								yield batch
								batch = []
						except (json.JSONDecodeError, KeyError, Exception) as e:
							logging.warning(f"Skipping malformed entry: {e}")
							continue

				env.close()

				if batch:
					yield batch
					batch = []
			except Exception as e:
				logging.error(f"Error reading {db_path}: {e}")

	def _get_db_count(self, db_path: str) -> int:
		"""Returns the number of entries in an LMDB database."""
		if not os.path.exists(db_path):
			return 0
		try:
			env = lmdb.open(db_path, readonly=True, lock=False)
			count = env.stat()["entries"]
			env.close()
			return count
		except Exception:
			return 0

	def find_reaction_candidates(self):
		"""
		Generates reaction candidates using hierarchical generation (G0, G1, G2+)
		and writes them to generation-specific LMDB databases.
		Uses memory-efficient hashing and streaming to scale to millions of reactions.
		"""
		click.secho("\n--- Starting Hierarchical Reaction Network Generation ---", bold=True)

		if not os.path.exists(self.input_csv):
			click.secho(f"Error: Input file not found at {self.input_csv}", fg="red")
			raise FileNotFoundError

		df = pd.read_csv(self.input_csv)
		initial_molecules = df["SMILES"].tolist()
		click.secho(f"Loaded {len(initial_molecules)} molecules from {self.input_csv}", fg="green")

		initial_molecules_set = {canonicalize_smiles(s) for s in initial_molecules if s and isinstance(s, str)}
		all_reactions_seen_hashes = set()

		# --- Generation 0: Initial Dissociations ---
		click.secho("\n--- G0: Performing Initial Bond Dissociations ---", bold=True)
		g0_db_path = self._db_paths["candidates_g0"]
		if os.path.exists(g0_db_path):
			click.secho(f"Skipping G0, database already exists.", fg="green")
			g0_count = self._get_db_count(g0_db_path)
			for smi in tqdm(self._iter_smiles_from_db(g0_db_path), total=g0_count, desc="G0: Loading hashes"):
				all_reactions_seen_hashes.add(self._get_reaction_hash(smi))
		else:
			q = Queue(maxsize=self.n_workers * 2)
			writer = Process(target=self._lmdb_writer, args=(q, g0_db_path))
			writer.start()

			with Pool(self.n_workers) as pool:
				results_iterator = pool.imap(get_dissociation_fragments, initial_molecules)

				for parent_smi, frag_set in tqdm(
					zip(initial_molecules, results_iterator),
					total=len(initial_molecules),
					desc="G0: Dissociating",
				):
					for f1, f2 in frag_set:
						if not (f1 in initial_molecules_set and f2 in initial_molecules_set):
							continue

						# Dissociation: C -> A + B
						rxn_smi = f"{parent_smi}>>{f1}.{f2}"
						r_counts = element_counts(parent_smi)
						p_counts = element_counts(f"{f1}.{f2}")
						if r_counts == p_counts:
							h = self._get_reaction_hash(rxn_smi)
							if h not in all_reactions_seen_hashes:
								all_reactions_seen_hashes.add(h)
								key = hashlib.sha256(rxn_smi.encode("utf-8")).hexdigest().encode("utf-8")
								value = json.dumps({"smi": rxn_smi, "gen": "G0"}).encode("utf-8")
								q.put((key, value))

						# Addition: A + B -> C
						rxn_smi = f"{f1}.{f2}>>{parent_smi}"
						if r_counts == p_counts:
							h = self._get_reaction_hash(rxn_smi)
							if h not in all_reactions_seen_hashes:
								all_reactions_seen_hashes.add(h)
								key = hashlib.sha256(rxn_smi.encode("utf-8")).hexdigest().encode("utf-8")
								value = json.dumps({"smi": rxn_smi, "gen": "G0"}).encode("utf-8")
								q.put((key, value))

			q.put(None)
			writer.join()

		click.secho(f"Found {len(all_reactions_seen_hashes):,} unique G0 reactions.", fg="green")

		# --- Generation 1: Transfer and Rearrangement Reactions ---
		if self.num_generations >= 1:
			click.secho(f"\n--- G1: Generating Transfer and Rearrangement Reactions ---", bold=True)
			g1_db_path = self._db_paths["candidates_g1"]
			if os.path.exists(g1_db_path):
				click.secho(f"Skipping G1, database already exists.", fg="green")
				g1_count = self._get_db_count(g1_db_path)
				for smi in tqdm(self._iter_smiles_from_db(g1_db_path), total=g1_count, desc="G1: Loading hashes"):
					all_reactions_seen_hashes.add(self._get_reaction_hash(smi))
			else:
				click.secho("G1: Indexing G0 fragments to find valid reaction pairs...", fg="cyan")
				fragment_to_rxn_idx = collections.defaultdict(list)
				g0_smiles = []
				g0_total = self._get_db_count(g0_db_path)
				for idx, rxn in enumerate(
					tqdm(self._iter_smiles_from_db(g0_db_path), total=g0_total, desc="G1: Indexing")
				):
					g0_smiles.append(rxn)
					try:
						_, right = rxn.split(">>")
						for frag in right.split("."):
							if frag:
								fragment_to_rxn_idx[frag].append(idx)
					except ValueError:
						continue

				if not g0_smiles:
					click.secho("Not enough G0 reactions to generate G1. Skipping.", fg="yellow")
				else:
					seen_pairs = set()
					for indices in tqdm(fragment_to_rxn_idx.values(), desc="G1: Finding pairs"):
						if len(indices) > 1:
							# For fragments with very high frequency, consider sampling or limiting to avoid N^2 explosion
							if len(indices) > 2000:
								logging.warning(f"Fragment indexing: skipping extremely high-frequency fragment.")
								continue
							for i in range(len(indices)):
								for j in range(i + 1, len(indices)):
									idx1, idx2 = indices[i], indices[j]
									pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
									seen_pairs.add(pair)

					reaction_pairs = ((g0_smiles[i], g0_smiles[j]) for i, j in seen_pairs)
					total_pairs = len(seen_pairs)
					click.secho(f"G1: Found {total_pairs:,} candidate pairs.", fg="cyan")

					q = Queue(maxsize=self.n_workers * 2)
					writer = Process(target=self._lmdb_writer, args=(q, g1_db_path))
					writer.start()

					with Pool(self.n_workers) as pool:
						with tqdm(total=total_pairs, desc="G1: Processing") as pbar:
							for batch in chunked_iterable(reaction_pairs, 1_000_000):
								results_iterator = pool.imap_unordered(
									worker_generate_new_reactions_g1, batch, chunksize=1024
								)
								for res_list in results_iterator:
									for candidate in res_list:
										h = self._get_reaction_hash(candidate)
										if h not in all_reactions_seen_hashes:
											all_reactions_seen_hashes.add(h)
											key = hashlib.sha256(candidate.encode("utf-8")).hexdigest().encode("utf-8")
											value = json.dumps({"smi": candidate, "gen": "G1"}).encode("utf-8")
											q.put((key, value))
									pbar.update(1)

					q.put(None)
					writer.join()
					del seen_pairs
					del fragment_to_rxn_idx
					del g0_smiles

		# --- Generation 2+: Higher-order reactions ---
		for gen in range(2, self.num_generations + 1):
			max_c = self.max_reaction_complexity
			click.secho(f"\n--- G{gen}: Generating Higher-order Reactions (max_complexity={max_c}) ---", bold=True)
			g2plus_db_path = self._db_paths["candidates_g2plus"].format(gen=gen, max_c=max_c)

			if os.path.exists(g2plus_db_path):
				click.secho(f"Skipping G{gen}, database already exists.", fg="green")
				g2_total = self._get_db_count(g2plus_db_path)
				for smi in tqdm(
					self._iter_smiles_from_db(g2plus_db_path), total=g2_total, desc=f"G{gen}: Loading hashes"
				):
					all_reactions_seen_hashes.add(self._get_reaction_hash(smi))
				continue

			# current_generation is G(gen-1)
			if gen == 2:
				current_gen_db = self._db_paths["candidates_g1"]
				previous_gens_dbs = [self._db_paths["candidates_g0"]]
			else:
				current_gen_db = self._db_paths["candidates_g2plus"].format(
					gen=gen - 1, max_c=self.max_reaction_complexity
				)
				previous_gens_dbs = [self._db_paths["candidates_g0"], self._db_paths["candidates_g1"]]
				for prev_gen in range(2, gen - 1):
					previous_gens_dbs.append(
						self._db_paths["candidates_g2plus"].format(gen=prev_gen, max_c=self.max_reaction_complexity)
					)

			click.secho(f"G{gen}: Indexing previous reactions...", fg="cyan")
			prev_product_to_rxn = collections.defaultdict(list)
			prev_reactant_to_rxn = collections.defaultdict(list)
			all_prev_smiles = []

			for db_p in previous_gens_dbs:
				p_total = self._get_db_count(db_p)
				for rxn in tqdm(
					self._iter_smiles_from_db(db_p), total=p_total, desc=f"G{gen}: Indexing {os.path.basename(db_p)}"
				):
					idx = len(all_prev_smiles)
					all_prev_smiles.append(rxn)
					try:
						reactants, products = rxn.split(">>")
						for r in reactants.split("."):
							if r:
								prev_reactant_to_rxn[r].append(idx)
						for p in products.split("."):
							if p:
								prev_product_to_rxn[p].append(idx)
					except ValueError:
						continue

			click.secho(f"G{gen}: Pairing with current generation and processing...", fg="cyan")

			q = Queue(maxsize=self.n_workers * 2)
			writer = Process(target=self._lmdb_writer, args=(q, g2plus_db_path))
			writer.start()

			def pair_generator():
				curr_total = self._get_db_count(current_gen_db)
				for rxn_curr in tqdm(
					self._iter_smiles_from_db(current_gen_db), total=curr_total, desc=f"G{gen}: Pairing", leave=False
				):
					try:
						reactants, products = rxn_curr.split(">>")
						curr_reactants = {r for r in reactants.split(".") if r}
						curr_products = {p for p in products.split(".") if p}

						matched_prev_indices = set()
						# Case 1: Product of r_curr is reactant of r_prev
						for p in curr_products:
							if p in prev_reactant_to_rxn:
								matched_prev_indices.update(prev_reactant_to_rxn[p])
						# Case 2: Product of r_prev is reactant of r_curr
						for r in curr_reactants:
							if r in prev_product_to_rxn:
								matched_prev_indices.update(prev_product_to_rxn[r])
						# Case 3: Reactants share species
						for r in curr_reactants:
							if r in prev_reactant_to_rxn:
								matched_prev_indices.update(prev_reactant_to_rxn[r])

						for prev_idx in matched_prev_indices:
							yield (rxn_curr, all_prev_smiles[prev_idx])
					except ValueError:
						continue

			with Pool(self.n_workers) as pool:
				partial_worker = partial(
					worker_generate_higher_gen_reactions, max_reaction_complexity=self.max_reaction_complexity
				)
				# Use a smaller chunksize for imap_unordered to keep the generator responsive
				results_iterator = pool.imap_unordered(partial_worker, pair_generator(), chunksize=256)

				for res_list in tqdm(results_iterator, desc=f"G{gen}: Processing"):
					for candidate in res_list:
						h = self._get_reaction_hash(candidate)
						if h not in all_reactions_seen_hashes:
							all_reactions_seen_hashes.add(h)
							key = hashlib.sha256(candidate.encode("utf-8")).hexdigest().encode("utf-8")
							value = json.dumps({"smi": candidate, "gen": f"G{gen}"}).encode("utf-8")
							q.put((key, value))

			q.put(None)
			writer.join()
			del all_prev_smiles, prev_product_to_rxn, prev_reactant_to_rxn

		click.secho(f"\nTotal unique reaction candidates: {len(all_reactions_seen_hashes):,}", bold=True)
		click.secho("Hierarchical reaction network generation complete.", fg="green")

	def verify_reactions(self):
		click.secho("\n--- Step 2: Verifying Reactions with RXNMapper ---", bold=True)

		custom_reactants_filter = set()
		if self.require_custom_reactant:
			click.secho(
				"--require-custom-reactant is active for verification.",
				fg="cyan",
				bold=True,
			)
			if self.custom_reactants_csv and os.path.exists(self.custom_reactants_csv):
				try:
					df_custom = pd.read_csv(self.custom_reactants_csv)
					if "SMILES" in df_custom.columns:
						custom_reactants_filter = set(df_custom["SMILES"].dropna().tolist())
						if custom_reactants_filter:
							click.secho(
								f"Loaded {len(custom_reactants_filter)} custom molecules for filtering.",
								fg="green",
							)
						else:
							click.secho(
								"Custom reactants file is empty. No filtering will be applied.",
								fg="yellow",
							)
					else:
						click.secho(
							"Custom reactants file missing 'SMILES' column. No filtering will be applied.",
							fg="yellow",
						)
				except Exception as e:
					click.secho(
						f"Error reading custom reactants file: {e}. No filtering will be applied.",
						fg="red",
					)
			else:
				click.secho(
					"Custom reactants file not provided or found. No filtering will be applied.",
					fg="yellow",
				)

		candidate_db_paths = []
		if not self.require_custom_reactant:
			g0_path = self._db_paths["candidates_g0"]
			if os.path.exists(g0_path):
				candidate_db_paths.append(g0_path)

		if self.num_generations >= 1:
			g1_path = self._db_paths["candidates_g1"]
			if os.path.exists(g1_path):
				candidate_db_paths.append(g1_path)

		for gen in range(2, self.num_generations + 1):
			max_c = gen + 2
			db_path = self._db_paths["candidates_g2plus"].format(
				gen=gen, max_c=min(max_c, self.max_reaction_complexity)
			)
			if os.path.exists(db_path):
				candidate_db_paths.append(db_path)

		total_candidates = 0
		for db_path in candidate_db_paths:
			if os.path.exists(db_path):
				env = lmdb.open(db_path, readonly=True, lock=False)
				total_candidates += env.stat()["entries"]
				env.close()

		if total_candidates == 0:
			click.secho(
				"No candidates to verify. Please run 'find-candidates' first.",
				fg="yellow",
			)
			return

		click.secho(
			f"Found {total_candidates:,} total candidates to verify from specified databases.",
			fg="green",
		)
		if custom_reactants_filter:
			click.secho(
				"Filtering candidates to only include those involving custom reactants...",
				fg="cyan",
			)

		batch_size = 4
		batch_generator = self._lmdb_batch_iterator(
			candidate_db_paths,
			batch_size,
			custom_reactants_filter=custom_reactants_filter,
		)
		total_batches = math.ceil(total_candidates / batch_size)

		writer_env = lmdb.open(self._db_paths["verified"], map_size=10**11, writemap=True)

		with writer_env.begin(write=True) as txn_out, Pool(self.n_workers) as pool:
			partial_worker = partial(
				worker_verify_reaction_batch,
				threshold=self.radical_threshold,
			)
			for verified_batch in tqdm(
				pool.imap_unordered(partial_worker, batch_generator),
				total=total_batches,
				desc="Verifying Batches",
			):
				for result_smi, gen in verified_batch:
					if result_smi:
						key = hashlib.sha256(result_smi.encode("utf-8")).hexdigest().encode("utf-8")
						value = json.dumps({"smi": result_smi, "gen": gen}).encode("utf-8")
						txn_out.put(key, value, overwrite=False)

		click.secho(
			f"Verified and saved {writer_env.stat()['entries']:,} unique, chemically plausible reactions.",
			fg="green",
		)
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
		num_reactions = env.stat()["entries"]

		if num_reactions == 0:
			click.secho("No verified reactions to export.", fg="yellow")
			env.close()
			return

		reaction_data = []
		reactions_by_gen = {}
		with env.begin() as txn:
			for key, value in tqdm(txn.cursor(), total=num_reactions, desc="Exporting to CSV"):
				# reaction_smarts = key.decode()
				try:
					value_data = json.loads(value.decode("utf-8"))
					reaction_smarts = value_data["smi"]
					gen = value_data["gen"]

					reactants, products = reaction_smarts.split(">>")
					reaction_dict = {"reactants": reactants, "products": products}

					if gen not in reactions_by_gen:
						reactions_by_gen[gen] = []

					reactions_by_gen[gen].append(reaction_dict)
					reaction_data.append(reaction_dict)

				except (json.JSONDecodeError, KeyError) as e:
					logging.warning(f"Skipping malformed entry with key {key.hex()}: {e}")
					continue

		df = pd.DataFrame(reaction_data)
		df.to_csv(output_csv_path, index=False)
		click.secho(
			f"Successfully exported {len(df)} reactions to {output_csv_path}",
			fg="green",
		)

		for gen_name, gen_data in reactions_by_gen.items():
			if gen_data:
				gen_output_path = os.path.join(self.output_dir, f"reactions_{gen_name}.csv")
				df_gen = pd.DataFrame(gen_data)
				df_gen.to_csv(gen_output_path, index=False)
				click.secho(
					f"Exported {len(df_gen)} verified {gen_name.upper()} reactions to {gen_output_path}",
					fg="green",
				)

		env.close()

	def generate_reaction_network_graph(self, filename: str = "reaction_network.json"):
		"""Generates a NetworkX graph from verified reactions and saves it as JSON."""
		click.secho("\n--- Step 3b: Generating Reaction Network Graph (JSON) ---", bold=True)

		db_path = self._db_paths["verified"]
		output_json_path = os.path.join(self.output_dir, filename)

		if not os.path.exists(db_path):
			click.secho(
				"Verified reactions database not found. Cannot generate graph.",
				fg="red",
			)
			return

		env = lmdb.open(db_path, readonly=True, lock=False)
		num_reactions = env.stat()["entries"]

		if num_reactions == 0:
			click.secho("No verified reactions to build graph from.", fg="yellow")
			return

		def get_canonical(smi):
			try:
				from rdkit import Chem

				mol = Chem.MolFromSmiles(smi)
				if mol:
					return Chem.MolToSmiles(mol, isomericSmiles=True)
			except:
				pass
			return smi

		G = nx.DiGraph()
		with env.begin() as txn:
			for key, _ in tqdm(txn.cursor(), total=num_reactions, desc="Building graph"):
				reaction_smi = key.decode()
				left, _, right = reaction_smi.partition(">>")

				reactant_list = [get_canonical(s) for s in left.split(".") if s]
				product_list = [get_canonical(s) for s in right.split(".") if s]

				# Generate a canonical reaction string for the node ID
				can_rxn_smi = ".".join(sorted(reactant_list)) + ">>" + ".".join(sorted(product_list))

				# We use the canonical SMILES as the ID, but keep the original for display
				G.add_node(can_rxn_smi, type="reaction", smiles=reaction_smi)

				for r_smi in reactant_list:
					G.add_node(r_smi, type="molecule", smiles=r_smi)
					G.add_edge(r_smi, can_rxn_smi)

				for p_smi in product_list:
					G.add_node(p_smi, type="molecule", smiles=p_smi)
					G.add_edge(can_rxn_smi, p_smi)

		graph_data = nx.node_link_data(G, edges="links")
		with open(output_json_path, "w") as f:
			json.dump(graph_data, f, indent=2)

		click.secho(
			f"Successfully generated and saved reaction network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to {output_json_path}",
			fg="green",
		)

	def export_images(self, image_dir: str = "reaction_images"):
		"""Exports verified reactions from the database to PNG images."""
		click.secho("\n--- Step 3c: Exporting Reactions to Images ---", bold=True)

		db_path = self._db_paths["verified"]
		if not os.path.exists(db_path):
			click.secho(
				"Verified reactions database not found. Please run 'verify-reactions' first.",
				fg="red",
			)
			return

		env = lmdb.open(db_path, readonly=True)
		num_reactions = env.stat()["entries"]

		if num_reactions == 0:
			click.secho("No verified reactions to export as images.", fg="yellow")
			env.close()
			return

		img_output_dir = os.path.join(self.output_dir, image_dir)
		os.makedirs(img_output_dir, exist_ok=True)

		click.secho(
			f"Exporting {num_reactions} reactions as images to {img_output_dir}...",
			bold=True,
		)

		total_exported = 0
		with env.begin() as txn:
			for i, (key, _) in enumerate(tqdm(txn.cursor(), total=num_reactions, desc="Exporting reaction images")):
				reaction_smi = key.decode()
				try:
					rxn = AllChem.ReactionFromSmarts(reaction_smi, useSmiles=True)
					img = Draw.ReactionToImage(rxn)
					img.save(os.path.join(img_output_dir, f"reaction_{i}.png"))
					total_exported += 1
				except Exception as e:
					logging.warning(f"Could not generate image for reaction {i} ({reaction_smi}). Error: {e}")

		env.close()
		if total_exported > 0:
			click.secho(
				f"\nFinished exporting {total_exported} reaction images to {img_output_dir}",
				fg="green",
			)
		else:
			click.secho("\nNo reactions were processed for image export.", fg="yellow")

	def _get_candidate_db_paths(self) -> List[str]:
		"""Helper to collect all existing candidate LMDB paths."""
		candidate_db_paths = []
		g0_path = self._db_paths["candidates_g0"]
		if os.path.exists(g0_path):
			candidate_db_paths.append(g0_path)

		if self.num_generations >= 1:
			g1_path = self._db_paths["candidates_g1"]
			if os.path.exists(g1_path):
				candidate_db_paths.append(g1_path)

		for gen in range(2, self.num_generations + 1):
			max_c = gen + 2
			db_path = self._db_paths["candidates_g2plus"].format(
				gen=gen, max_c=min(max_c, self.max_reaction_complexity)
			)
			if os.path.exists(db_path):
				candidate_db_paths.append(db_path)
		return candidate_db_paths

	def export_to_kuzu(self, kuzu_dir: str = None, use_verified: bool = True, model_path: str = "model.script"):
		"""Exports the reaction network to KuzuDB."""
		from .kuzu_exporter import export_to_kuzu as kuzu_export

		if kuzu_dir is None:
			kuzu_dir = os.path.join(self.output_dir, "kuzu_db")

		kuzu_export(self, kuzu_dir=kuzu_dir, use_verified=use_verified, model_path=model_path)

	def update_kuzu_energies(self, kuzu_dir: str = None, model_path: str = "model.script"):
		"""Updates free energy predictions in an existing KuzuDB."""
		from .kuzu_exporter import update_kuzu_predictions

		if kuzu_dir is None:
			kuzu_dir = os.path.join(self.output_dir, "kuzu_db")

		update_kuzu_predictions(kuzu_dir=kuzu_dir, model_path=model_path)

	def explore(self):
		"""Run the full workflow: find candidates, verify, export CSV, and export graph."""
		self.find_reaction_candidates()
		self.verify_reactions()
		self.export_to_csv()
		self.generate_reaction_network_graph()
