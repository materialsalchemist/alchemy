from typing import Set, Tuple, List
from rdkit import Chem, RDLogger
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='rxnmapper')
from rxnmapper import RXNMapper

# Suppress verbose RDKit logging
RDLogger.DisableLog('rdApp.*')

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

rxn_mapper_instance = None

def worker_verify_reaction_batch(
	reaction_smi_bytes_batch: List[bytes], 
	confidence_threshold: float = 0.99
) -> List[str]:
	"""
	Worker function to verify a BATCH of reaction SMILES.
	Initializes the RXNMapper model only once per process.
	"""
	global rxn_mapper_instance

	if rxn_mapper_instance is None:
		rxn_mapper_instance = RXNMapper()

	valid_reactions = []
	reactions_to_map = []
	original_reactions_map = {}

	for reaction_smi_bytes in reaction_smi_bytes_batch:
		try:
			reaction_smi = reaction_smi_bytes.decode('utf-8')
			reactants_str, products_str = reaction_smi.split('>>')
			reactant_mols = [Chem.MolFromSmiles(s) for s in reactants_str.split('.')]
			product_mols = [Chem.MolFromSmiles(s) for s in products_str.split('.')]

			if any(m is None for m in reactant_mols + product_mols):
				continue
			
			r_cans = sorted([Chem.MolToSmiles(m, canonical=True) for m in reactant_mols])
			p_cans = sorted([Chem.MolToSmiles(m, canonical=True) for m in product_mols])

			if r_cans == p_cans:
				continue

			final_canonical_reaction = f"{'.'.join(r_cans)}>>{'.'.join(p_cans)}"
			original_reactions_map[reaction_smi] = final_canonical_reaction
			reactions_to_map.append(reaction_smi)
		except Exception as e:
			print(e)
			continue
	
	if not reactions_to_map:
		return []

	try:
		results = rxn_mapper_instance.get_attention_guided_atom_maps(reactions_to_map)

		for original_smi, result in zip(reactions_to_map, results):
			confidence = result.get('confidence', 0.0)
			mapped_rxn = result.get('mapped_rxn', "")

			if confidence >= confidence_threshold:
				canonical_form = original_reactions_map.get(original_smi)
				if canonical_form:
					valid_reactions.append(canonical_form)

	except Exception as e:
		logging.warning(f"rxnmapper failed for a batch: {e}")

	return valid_reactions
