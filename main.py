import itertools
	
import networkx as nx
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable
from tqdm import tqdm
import numpy as np
import numba
from numba import jit

import logging
from functools import lru_cache, cache
from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional, FrozenSet
from collections import Counter

import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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

@dataclass(unsafe_hash=True)
class ReactionNetwork:
	"""
	Generates and manages chemical reaction networks using vectorized operations.
	"""

	max_atoms: int = field(default=4)
	atoms: FrozenSet[str] = field(default_factory=lambda: frozenset(["C", "H", "O"]))
	bond_types: FrozenSet[Chem.BondType] = field(default_factory=lambda: frozenset([
		Chem.BondType.SINGLE,
		Chem.BondType.DOUBLE,
		Chem.BondType.TRIPLE,
		Chem.BondType.AROMATIC,
		Chem.BondType.QUADRUPLE,
		Chem.BondType.QUINTUPLE,
		Chem.BondType.HEXTUPLE,
		Chem.BondType.ONEANDAHALF,
		Chem.BondType.TWOANDAHALF,
		Chem.BondType.THREEANDAHALF,
		Chem.BondType.FOURANDAHALF,
		Chem.BondType.FIVEANDAHALF ,
		Chem.BondType.AROMATIC,
		Chem.BondType.IONIC,
		Chem.BondType.HYDROGEN,
		Chem.BondType.THREECENTER,
		Chem.BondType.DATIVEONE,
		Chem.BondType.DATIVE,
		Chem.BondType.DATIVEL,
		Chem.BondType.DATIVER,
		# Chem.BondType.ZERO,
		# Chem.BondType.OTHER,
	]))
	
	molecules: Set[str] = field(init=False, default_factory=set)
	reactions: Set[Tuple[str, str]] = field(init=False, default_factory=set)

	graph: nx.DiGraph = field(init=False, default_factory=nx.DiGraph)
	heavy_atoms: List[str] = field(init=False)
	periodic_table: Chem.rdchem.GetPeriodicTable = field(init=False)

	n_workers: int = field(default=4)
	_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

	def __post_init__(self):
		"""
		Initialize derived attributes and set up logging after instance creation.
		"""
		self.heavy_atoms = [atom for atom in self.atoms if atom != "H"]
		self.periodic_table = Chem.GetPeriodicTable()
		self._setup_logging()

	@staticmethod
	def _setup_logging() -> None:
		"""
		Configure logging settings for the reaction network generator.
		"""
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - ReactionNetwork::%(levelname)s - %(message)s'
		)

	def _chunk_iterator(self, iterator, chunk_size):
		"""Split an iterator into chunks for parallel processing."""
		iterator = iter(iterator)
		return iter(lambda: list(itertools.islice(iterator, chunk_size)), [])

	@staticmethod
	def _is_valid_molecule(smiles: str) -> bool:
		try:
			mol = Chem.MolFromSmiles(smiles)
			Chem.SanitizeMol(mol)
			return True
		except:
			return False

	def get_max_valence(self, atom_symbol: str) -> int:
		"""Get the maximum valence for an atom using RDKit's periodic table."""
		atomic_num = self.periodic_table.GetAtomicNumber(atom_symbol)
		return max(self.periodic_table.GetValenceList(atomic_num))
	
	@staticmethod
	def get_bond_order(bond_type: Chem.BondType) -> float:
		"""Get the numerical order of a bond type."""
		return BOND_ORDER_MAPPING.get(bond_type, 1.0)

	def can_form_bond(self, atom1: Chem.Atom, atom2: Chem.Atom, bond_type: Chem.BondType) -> bool:
		"""Check if two atoms can form a bond of given type based on their valences."""
		bond_order = self.get_bond_order(bond_type)
		if bond_order is None:
			return False
		
		val1 = self.get_atom_current_valence(atom1)
		val2 = self.get_atom_current_valence(atom2)
		max_val1 = self.get_max_valence(atom1.GetSymbol())
		max_val2 = self.get_max_valence(atom2.GetSymbol())
		
		return (
			val1 + bond_order <= max_val1 and 
			val2 + bond_order <= max_val2
		)

	@staticmethod
	def get_atom_current_valence(atom: Chem.Atom) -> int:
		"""Calculate current valence of an atom including implicit hydrogens."""
		explicit_valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
		implicit_h = atom.GetNumImplicitHs()
		return explicit_valence + implicit_h

	def _deduplicate_reactions(self):
		"""
		Remove duplicate reactions by keeping only one if the reactants are identical (even if reordered),
		but keep all if the reactants are actually different.
		"""
		unique_reactions = {}

		for reactants, product in self.reactions:
			sorted_reactants = " + ".join(sorted(reactants.split(" + ")))

			if product not in unique_reactions:
				unique_reactions[product] = {sorted_reactants: (reactants, product)}
			else:
				if sorted_reactants not in unique_reactions[product]:
					unique_reactions[product][sorted_reactants] = (reactants, product)

		self.reactions = set(
			reaction for reactant_dict in unique_reactions.values() for reaction in reactant_dict.values()
		)

	def generate_compositions(self):
		"""Generates all valid compositions of atoms up to max_atoms, ensuring order independence."""
		atom_list = list(self.heavy_atoms) 
		compositions_with_bonds = []

		for total_atoms in range(1, self.max_atoms + 1): 
			for atom_counts in itertools.combinations_with_replacement(atom_list, total_atoms):
				composition = {atom: atom_counts.count(atom) for atom in atom_list}

				num_bonds = total_atoms - 1  

				for bond_combo in itertools.combinations_with_replacement(self.bond_types, num_bonds):
					compositions_with_bonds.append((composition, bond_combo))

		return compositions_with_bonds

	def composition_to_mol(self, composition, bond_combo):
		"""Constructs a molecule from atom composition and a given bond combination."""
		mol = Chem.RWMol()

		atom_indices = {
			atom: [mol.AddAtom(Chem.Atom(atom)) for _ in range(count)] 
			for atom, count in composition.items()
		}
		
		all_indices = [idx for indices in atom_indices.values() for idx in indices]

		for i, bond_type in enumerate(bond_combo):
			mol.AddBond(all_indices[i], all_indices[i + 1], bond_type)

		try:
			mol.UpdatePropertyCache()
			Chem.SanitizeMol(mol)
			return mol
		except Exception as e:
			logging.debug(f"Sanitization failed: {e}")
			return None

	def _process_molecule_chunk(self, chunk):
		"""Process a chunk of molecule compositions."""
		local_molecules = set()
		for comp, bond_combo in chunk:
			mol = self.composition_to_mol(comp, bond_combo)
			if mol:
				smi = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
				if self._is_valid_molecule(smi):
					local_molecules.add(smi)
		return local_molecules

	def generate_molecules(self):
		"""Generate molecules using parallel processing."""
		self.molecules.clear()
		compositions = list(self.generate_compositions())
		chunk_size = max(1, len(compositions) // (self.n_workers * 4))
		chunks = list(self._chunk_iterator(compositions, chunk_size))

		with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
			futures = [executor.submit(self._process_molecule_chunk, chunk) 
					  for chunk in chunks]

			with tqdm(total=len(chunks), desc="Generating molecules") as pbar:
				for future in concurrent.futures.as_completed(futures):
					with self._lock:
						self.molecules.update(future.result())
					pbar.update(1)

		logging.info(f"Generated {len(self.molecules)} valid molecules")


	def _process_dissociation_chunk(self, smiles_chunk):
		"""Process a chunk of molecules for dissociation reactions."""
		local_reactions = set()
		for smiles in smiles_chunk:
			mol = Chem.MolFromSmiles(smiles)
			if not mol:
				continue

			for bond in mol.GetBonds():
				new_mol = Chem.RWMol(mol)
				a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
				new_mol.RemoveBond(a1, a2)

				new_mol.GetAtomWithIdx(a1).SetNoImplicit(True)
				new_mol.GetAtomWithIdx(a2).SetNoImplicit(True)

				try:
					frags = Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=True)
					frag_smis = [Chem.MolToSmiles(f, canonical=True, allHsExplicit=True) for f in frags]
					if len(frag_smis) == 2:
						frag1, frag2 = frag_smis
					elif len(frag_smis) == 1:
						frag1, frag2 = frag_smis[0], ""
					else:
						raise ValueError(f"Too many fragments for {smiles}")

					local_reactions.add((smiles, " + ".join(frag_smis)))
				except Exception as e:
					logging.debug(f"Failed dissociation for {smiles}: {e}")

		return local_reactions

	def generate_bond_dissociation(self):
		"""Generate bond dissociation reactions in parallel."""
		chunks = list(self._chunk_iterator(list(self.molecules), max(1, len(self.molecules) // (self.n_workers * 4))))

		with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
			futures = [executor.submit(self._process_dissociation_chunk, chunk) for chunk in chunks]

			with tqdm(total=len(chunks), desc="Generating dissociations") as pbar:
				for future in concurrent.futures.as_completed(futures):
					with self._lock:
						self.reactions.update(future.result())
					pbar.update(1)

		logging.info(f"Generated {len(self.reactions)} dissociation reactions")

	def _process_formation_chunk(self, mol_pairs_chunk):
		"""Process a chunk of molecule pairs for bond formation."""
		local_reactions = set()

		for mol1_smi, mol2_smi in mol_pairs_chunk:
			mol1 = Chem.MolFromSmiles(mol1_smi)
			mol2 = Chem.MolFromSmiles(mol2_smi)

			if not (mol1 and mol2):
				continue

			combined = Chem.CombineMols(mol1, mol2)
			num_atoms_mol1 = mol1.GetNumAtoms()

			atoms1 = [(i, combined.GetAtomWithIdx(i)) for i in range(num_atoms_mol1)]
			atoms2 = [(j, combined.GetAtomWithIdx(j)) for j in range(num_atoms_mol1, combined.GetNumAtoms())]

			for (i, atom1), (j, atom2) in itertools.product(atoms1, atoms2):
				atom1_sym = atom1.GetSymbol()
				atom2_sym = atom2.GetSymbol()

				for bond_type in self.bond_types:
					if not self.can_form_bond(atom1, atom2, bond_type):
						logging.debug(f"Valence check failed for {atom1_sym}-{atom2_sym} with bond type {bond_type}")
						continue

					rwmol = Chem.RWMol(combined)
					try:
						rwmol.AddBond(i, j, bond_type)
						rwmol.UpdatePropertyCache()
						Chem.SanitizeMol(rwmol)

						product_smi = Chem.MolToSmiles(rwmol, canonical=True, allHsExplicit=True)
						if self._is_valid_molecule(product_smi):
							reactants = " + ".join([mol1_smi, mol2_smi])
							# reactants = " + ".join(sorted([mol1_smi, mol2_smi]))
							local_reactions.add((reactants, product_smi))
							logging.debug(f"Successfully created: {reactants} -> {product_smi}")
						else:
							logging.debug(f"Validation failed for product: {product_smi}")
					except Exception:
						continue

		return local_reactions

	def generate_bond_formation(self):
		"""Generate bond formation reactions in parallel."""
		mol_pairs = list(itertools.product(self.molecules, repeat=2))
		chunk_size = max(1, len(mol_pairs) // (self.n_workers * 4))
		chunks = list(self._chunk_iterator(mol_pairs, chunk_size))

		with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
			futures = [
				executor.submit(self._process_formation_chunk, chunk) 
				for chunk in chunks
			]

			with tqdm(total=len(chunks), desc="Generating formations") as pbar:
				for future in concurrent.futures.as_completed(futures):
					with self._lock:
						self.reactions.update(future.result())
					pbar.update(1)

		self._deduplicate_reactions()
		logging.info(f"Generated {len(self.reactions)} formation reactions")

	def generate_rearrangements(self):
		"""Generates rearrangement reactions where atoms shift within a molecule."""

		new_reactions = set()

		for smiles in self.molecules:
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				continue

			# Swap atom connectivity in a way that keeps valency valid
			for bond in mol.GetBonds():
				if bond.IsInRing():
					continue

				new_mol = Chem.RWMol(mol)
				a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
				new_mol.RemoveBond(a1, a2)
				new_mol.AddBond(a1, a2, bond.GetBondType())
				new_mol.UpdatePropertyCache()

				new_mol = new_mol.GetMol()
				if new_mol and Chem.SanitizeMol(new_mol, catchErrors=True) == 0:
					rearranged_smiles = Chem.MolToSmiles(new_mol)
					if rearranged_smiles and rearranged_smiles != smiles:
						new_reactions.add((smiles, rearranged_smiles))

		self.reactions.update(new_reactions)
		logging.info(f"Generated {len(new_reactions)} rearrangement reactions.")

	def generate_transfer_reactions(self):
		"""
		Generate transfer reactions by combining two bond dissociation reactions
		that share a common fragment.

		For example, if we have:
				Reaction 1: A -> B + C
				Reaction 2: D -> B + E
		then a transfer reaction can be defined as:
				A + E -> C + D

		This function iterates over pairs of dissociation reactions, checks for exactly one
		common fragment in their products, and constructs a new reaction accordingly.
		"""

		new_reactions = set()

		for (react1, prod1), (react2, prod2) in itertools.combinations(self.reactions, 2):
			frags1 = set(prod1.split(" + "))
			frags2 = set(prod2.split(" + "))

			common = frags1.intersection(frags2)
			if len(common) == 1:
				common_frag = common.pop()

				p1_rest = frags1 - {common_frag}
				p2_rest = frags2 - {common_frag}

				if len(p1_rest) == 1 and len(p2_rest) == 1:
					frag1 = next(iter(p1_rest))
					frag2 = next(iter(p2_rest))

					new_left = f"{react1} + {frag2}"
					new_right = f"{frag1} + {react2}"
					new_reactions.add((new_left, new_right))

		self.reactions.update(new_reactions)

	def construct_graph(self):
		"""Builds the reaction network as a bipartite graph with metadata."""
		self.graph.clear()
		
		for reactant, product in self.reactions:
			reaction_node = f"Reaction: {reactant} → {product}"
			
			# Add nodes with metadata
			self.graph.add_node(reactant, type="molecule")
			self.graph.add_node(reaction_node, type="reaction")
			self.graph.add_node(product, type="molecule")
			
			# Add edges with metadata
			self.graph.add_edge(reactant, reaction_node, type="reactant")
			self.graph.add_edge(reaction_node, product, type="product")


	def save_network(self, filename="reaction_network.graphml"):
		"""Saves the reaction network as a GraphML file."""
		nx.write_graphml(self.graph, filename)
		print(f"Reaction network saved as {filename}")

	def save_graph_as_png(self, filename = "reaction_network.png"):
		"""Generates and saves the reaction network visualization with improved styling."""

		plt.figure(figsize=(12, 8))
		
		if len(self.graph) > 100:
			pos = nx.spring_layout(self.graph, k=1, iterations=50)
		else:
			pos = nx.kamada_kawai_layout(self.graph)
		
		# Draw nodes with different colors for molecules and reactions
		molecule_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'molecule']
		reaction_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'reaction']
		
		nx.draw_networkx_nodes(self.graph, pos, nodelist=molecule_nodes, node_color='lightblue', node_size=700)
		nx.draw_networkx_nodes(self.graph, pos, nodelist=reaction_nodes, node_color='lightgreen', node_size=500)
		
		nx.draw_networkx_edges(self.graph, pos, edge_color="gray", arrows=True)
		nx.draw_networkx_labels(self.graph, pos, font_size=8)
		
		plt.title("Chemical Reaction Network")
		plt.axis('off')
		
		try:
			plt.savefig(filename, dpi=300, bbox_inches="tight")
			plt.close()
			logging.info(f"Reaction network visualization saved as {filename}")
		except Exception as e:
			logging.debug(f"Error saving visualization to {filename}: {str(e)}")


if __name__ == "__main__":
	import argparse
	from rdkit import RDLogger

	RDLogger.DisableLog("rdApp.error")

	parser = argparse.ArgumentParser(
		prog="crn_py",
	)
	parser.add_argument("-n", "--max_atoms", type=int, default=2)
	args = parser.parse_args()

	network = ReactionNetwork(max_atoms=args.max_atoms, n_workers=12)

	print("Generating molecules...")
	network.generate_molecules()
	print(network.molecules)

	print("Generating bond dissociation reactions...")
	network.generate_bond_dissociation()
	print(network.reactions)

	print("Generating bond formation reactions...")
	network.generate_bond_formation()
	print(f"There are {len(network.reactions)} reactions")
	print(network.reactions)

	# print("Generating rearrangements reactions...")
	# network.generate_rearrangements()
	# print(f"There are {len(network.reactions)} reactions")
	#
	# print("Generating transfer reactions...")
	# print(network.reactions)
	# network.generate_transfer_reactions()
	# print(f"There are {len(network.reactions)} reactions")
	#
	# print("Constructing reaction network graph...")
	# network.construct_graph()
	#
	# print(network.reactions)
	#
	# network.save_network()
	# network.save_graph_as_png()
