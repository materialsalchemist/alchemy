from collections import Counter
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def count_atom_types(smiles):
	mol = Chem.MolFromSmiles(smiles, sanitize=True)
	mol = Chem.rdmolops.AddHs(mol)
	return Counter([atom.GetSymbol() for atom in mol.GetAtoms()])

class ReactionNetwork:
	def __init__(self, max_atoms=4, atoms=None):
		self.max_atoms = max_atoms
		self.atoms = atoms if atoms is not None else ["C", "H", "O"]
		self.heavy_atoms = list(filter(lambda x: x != "H", self.atoms))
		self.molecules = set()
		self.reactions = set()
		self.graph = nx.DiGraph()

	def generate_compositions(self):
		ranges = [range(self.max_atoms + 1)] * len(self.heavy_atoms)
		return (
			{ atom: count for atom, count in zip(self.heavy_atoms, composition) if count > 0 }
			for composition in itertools.product(*ranges) if any(composition)
		)

	def composition_to_mol(self, composition):
		mol = Chem.RWMol()
		
		atom_indices = {
			atom: [mol.AddAtom(Chem.Atom(atom)) for _ in range(count)] 
			for atom, count in composition.items()
		}
		
		# Connect all atoms linearly (naive structure)
		all_indices = [idx for indices in atom_indices.values() for idx in indices]
		for i in range(len(all_indices) - 1):
			mol.AddBond(all_indices[i], all_indices[i + 1], Chem.BondType.SINGLE)

		try:
			mol.UpdatePropertyCache()
			Chem.SanitizeMol(mol)

			return mol

		except Exception as e:
			print("Sanitization failed:", e)
			return None

	def generate_molecules(self):
		"""Systematically generates molecular fragments within the C-H-O subspace."""

		self.molecules.clear()

		for comp in self.generate_compositions():
			mol = self.composition_to_mol(comp)
			if mol:
				# self.molecules.add(Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True))
				smi = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
				logging.info(f"{comp} -> {smi}")
				self.molecules.add(smi)

		logging.info(f"Generated {len(self.molecules)} molecules.")

	def generate_bond_dissociation(self):
		"""Generates G0 bond dissociation reactions."""

		for smiles in tqdm(self.molecules, desc="Generating bond dissociations"):
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				continue

			for bond in mol.GetBonds():
				new_mol = Chem.RWMol(mol)

				a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
				new_mol.RemoveBond(a1, a2)

				new_mol.GetAtomWithIdx(a1).SetNoImplicit(True)
				new_mol.GetAtomWithIdx(a2).SetNoImplicit(True)

				frags = Chem.GetMolFrags(new_mol.GetMol(), asMols=True)
				frag_smis = sorted([Chem.MolToSmiles(f) for f in frags])
				if len(frag_smis) == 2:
					frag1, frag2 = frag_smis
				elif len(frag_smis) == 1:
					frag1, frag2 = frag_smis[0], ""
				else:
					raise ValueError(f"Too many fragments for {smiles}")

				if count_atom_types(frag1) + count_atom_types(frag2) != count_atom_types(smiles):
					logging.warning(f"Mismatch atoms' counts for {smiles}; {prod1} {prod2}")
					return

				self.reactions.add((smiles, " + ".join(frag_smis)))

		logging.info(f"Generated {len(self.reactions)} dissociation reactions.")

	def generate_bond_formation(self):
		"""Generates bond formation (association) reactions."""

		new_reactions = set()

		for mol1, mol2 in itertools.combinations(self.molecules, 2):
			mol1_obj = Chem.MolFromSmiles(mol1)
			mol2_obj = Chem.MolFromSmiles(mol2)

			if mol1_obj and mol2_obj:
				merged = Chem.CombineMols(mol1_obj, mol2_obj)
				if merged and Chem.SanitizeMol(merged, catchErrors=True) == 0:
					smiles_merged = Chem.MolToSmiles(merged)
					new_reactions.add((f"{mol1} + {mol2}", smiles_merged))

		self.reactions.update(new_reactions)

	def generate_rearrangements(self):
		"""Generates rearrangement reactions where atoms shift within a molecule."""

		new_reactions = set()

		for smiles in self.molecules:
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				continue

			# Swap atom connectivity in a way that keeps valency valid
			for bond in mol.GetBonds():
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
			logging.error(f"Error saving visualization to {filename}: {str(e)}")


if __name__ == "__main__":
	import argparse
	import logging

	parser = argparse.ArgumentParser(
		prog="crn_py",
	)
	parser.add_argument("-n", "--max_atoms", type=int, default=2)
	args = parser.parse_args()
	
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

	network = ReactionNetwork(max_atoms=args.max_atoms)

	print("Generating molecules...")
	network.generate_molecules()
	# print(network.molecules)

	print("Generating bond dissociation reactions...")
	network.generate_bond_dissociation()
	print(sorted(network.reactions))
	#
	# print("Generating rearrangement reactions...")
	# network.generate_bond_formation()
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
