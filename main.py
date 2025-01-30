import itertools
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

import argparse
import logging

class ReactionNetwork:
	def __init__(self, max_atoms=4, atoms=["C", "H", "O"]):
		self.max_atoms = max_atoms
		self.atoms = atoms
		self.molecules = set()
		self.reactions = set()
		self.graph = nx.DiGraph()

	def generate_compositions(self):
		ranges = [range(self.max_atoms + 1)] * len(self.atoms)
		return (
			{ atom: count for atom, count in zip(self.atoms, composition) if count > 0 }
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

		return mol.GetMol()

	def generate_molecules(self):
		"""Systematically generates molecular fragments within the C-H-O subspace."""

		self.molecules.clear()

		for comp in self.generate_compositions():
				mol = self.composition_to_mol(comp)
				if mol and Chem.SanitizeMol(mol, catchErrors=True) == 0:
					self.molecules.add(Chem.MolToSmiles(mol))
		
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
				
				fragments = Chem.GetMolFrags(new_mol.GetMol(), asMols=True)
				if len(fragments) == 2:
					smi1, smi2 = sorted([Chem.MolToSmiles(fragments[0]), Chem.MolToSmiles(fragments[1])])
					self.reactions.add((smiles, f"{smi1} + {smi2}"))

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

				new_mol = new_mol.GetMol()
				if new_mol and Chem.SanitizeMol(new_mol, catchErrors=True) == 0:
					rearranged_smiles = Chem.MolToSmiles(new_mol)
					if rearranged_smiles and rearranged_smiles != smiles:
						new_reactions.add((smiles, rearranged_smiles))

		self.reactions.update(new_reactions)

	def generate_transfer_reactions(self):
		"""Generates transfer reactions where atoms exchange between molecules."""
		new_reactions = set()

		for reactant1, product1 in self.reactions:
			for reactant2, product2 in self.reactions:
				if product1 == product2 and reactant1 != reactant2:
					new_reactions.add((reactant1, reactant2))

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
	parser = argparse.ArgumentParser(
		prog="crn_py",
	)
	parser.add_argument("-n", "--max_atoms", type=int, default=2)
	args = parser.parse_args()

	network = ReactionNetwork(max_atoms=args.max_atoms)

	print("Generating molecules...")
	network.generate_molecules()

	print("Generating bond dissociation reactions...")
	network.generate_bond_dissociation()
	print(f"There are {len(network.reactions)} dissociation reactions")

	print("Generating rearrangement and transfer reactions...")
	network.generate_bond_formation()
	network.generate_rearrangements()
	network.generate_transfer_reactions()
	print(f"There are {len(network.reactions)} reactions")

	print("Constructing reaction network graph...")
	network.construct_graph()

	print(network.graph)

	network.save_network()
	network.save_graph_as_png()
