from typing import Set, Tuple, List
from rdkit import Chem, RDLogger
import logging
import warnings
from reaction_space.utils import element_counts

warnings.filterwarnings("ignore", category=UserWarning, module="rxnmapper")
from rxnmapper import RXNMapper

from .utils import (
	canonicalize_smiles,
	canonicalize_smiles_list,
	add_atomic_compositions,
	verify_reaction,
)

# Suppress verbose RDKit logging
RDLogger.DisableLog("rdApp.*")


def get_dissociation_fragments(mol_smiles: str) -> Set[Tuple[str, str]]:
	"""
	Dissociates a molecule by breaking each bond once to generate G0 reaction fragments.
	"""
	try:
		mol = Chem.MolFromSmiles(mol_smiles)
		if not mol:
			return set()

		mol_smiles = Chem.MolToSmiles(mol, canonical=True)

	except Exception:
		return set()

	mol = Chem.AddHs(mol)

	fragments = set()
	bonds = mol.GetBonds()

	for i in range(len(bonds)):
		emol = Chem.EditableMol(mol)

		begin_atom_idx = bonds[i].GetBeginAtomIdx()
		end_atom_idx = bonds[i].GetEndAtomIdx()

		emol.RemoveBond(begin_atom_idx, end_atom_idx)

		fragmented_mol = emol.GetMol()

		num_frags = len(Chem.GetMolFrags(fragmented_mol))

		if num_frags != 2:
			continue

		try:
			frag_mols = Chem.GetMolFrags(fragmented_mol, asMols=True)

			f1_mol, f2_mol = frag_mols[0], frag_mols[1]
			Chem.SanitizeMol(f1_mol)
			Chem.SanitizeMol(f2_mol)

			f1_smi = Chem.MolToSmiles(f1_mol, canonical=True)
			f2_smi = Chem.MolToSmiles(f2_mol, canonical=True)
			if f1_smi == "[HH]":
				f1_smi = "[H]"
			if f2_smi == "[HH]":
				f2_smi = "[H]"

			if not f1_smi or not f2_smi:
				continue

			f1_smi = canonicalize_smiles(f1_smi)
			f2_smi = canonicalize_smiles(f2_smi)
			sorted_frags = tuple(sorted((f1_smi, f2_smi)))
			fragments.add(sorted_frags)
		except Exception as e:
			continue
	return fragments


def combine_fragments(mol1: Chem.Mol, mol2: Chem.Mol) -> str:
	"""Helper to form a new bond between two radical fragments."""
	try:
		combo = Chem.CombineMols(mol1, mol2)
		emol = Chem.EditableMol(combo)

		radical_atoms = [a.GetIdx() for a in combo.GetAtoms() if a.GetNumRadicalElectrons() == 1]

		if len(radical_atoms) != 2:
			return None

		# Add a new single bond between the radical atoms
		emol.AddBond(radical_atoms[0], radical_atoms[1], order=Chem.rdchem.BondType.SINGLE)

		new_mol = emol.GetMol()

		# Reset radical counts on the atoms that just formed a bond
		for idx in radical_atoms:
			new_mol.GetAtomWithIdx(idx).SetNumRadicalElectrons(0)

		Chem.SanitizeMol(new_mol)
		final_mol = Chem.RemoveHs(new_mol)

		return Chem.MolToSmiles(final_mol, canonical=True)
	except Exception:
		return None


def worker_systematic_recombination(reactants: Tuple[str, str]) -> List[str]:
	"""
	Generates reactions by breaking a custom molecule and recombining its fragments
	with a G0 fragment.
	"""
	g0_frag_smi, custom_mol_smi = reactants
	new_reactions = []

	try:
		g0_frag_mol = Chem.MolFromSmiles(g0_frag_smi)
		if not g0_frag_mol:
			return []

		custom_frag_pairs = get_dissociation_fragments(custom_mol_smi)
		if not custom_frag_pairs:
			return []

		for f2_smi, f3_smi in custom_frag_pairs:
			f2_mol = Chem.MolFromSmiles(f2_smi)
			f3_mol = Chem.MolFromSmiles(f3_smi)
			if not f2_mol or not f3_mol:
				continue

			# Reaction A: Combine G0 fragment with f2, leaving f3
			new_mol_A = combine_fragments(g0_frag_mol, f2_mol)
			if new_mol_A:
				reacts = sorted(canonicalize_smiles_list([g0_frag_smi, custom_mol_smi]))
				prods = sorted(canonicalize_smiles_list([new_mol_A, f3_smi]))
				if reacts != prods:
					r = element_counts(reacts)
					p = element_counts(prods)
					if r == p:
						new_reactions.append(f"{'.'.join(reacts)}>>{'.'.join(prods)}")

			# Reaction B: Combine G0 fragment with f3, leaving f2
			new_mol_B = combine_fragments(g0_frag_mol, f3_mol)
			if new_mol_B:
				reacts = sorted(canonicalize_smiles_list([g0_frag_smi, custom_mol_smi]))
				prods = sorted(canonicalize_smiles_list([new_mol_B, f2_smi]))
				if reacts != prods:
					r = element_counts(reacts)
					p = element_counts(prods)
					if r == p:
						new_reactions.append(f"{'.'.join(reacts)}>>{'.'.join(prods)}")

	except Exception as e:
		logging.warning(f"Error in recombination worker for {reactants}: {e}")
		return []

	return list(set(new_reactions))


def worker_radical_addition(reactants: Tuple[str, str]) -> List[str]:
	"""
	Performs a radical addition reaction where a G0 fragment adds across a
	double or triple bond of a custom molecule.

	Reactant 1: The radical SMILES (from G0).
	Reactant 2: The add-on molecule SMILES (from the custom list).
	"""
	g0_frag_smi, custom_mol_smi = reactants
	new_reactions = []

	try:
		g0_frag_mol = Chem.MolFromSmiles(g0_frag_smi)
		custom_mol = Chem.MolFromSmiles(custom_mol_smi)
		if not g0_frag_mol or not custom_mol:
			return []

		# Find the radical atom in the G0 fragment
		rad_idx_in_frag = -1
		for atom in g0_frag_mol.GetAtoms():
			if atom.GetNumRadicalElectrons() == 1:
				rad_idx_in_frag = atom.GetIdx()
				break

		if rad_idx_in_frag == -1:
			return []

		# Find all unsaturated bonds (double or triple) in the custom molecule
		unsaturated_bonds = [
			b for b in custom_mol.GetBonds() if b.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
		]

		for bond in unsaturated_bonds:
			atom1_idx = bond.GetBeginAtomIdx()
			atom2_idx = bond.GetEndAtomIdx()

			for target_atom_idx, other_atom_idx in [
				(atom1_idx, atom2_idx),
				(atom2_idx, atom1_idx),
			]:
				combo = Chem.CombineMols(custom_mol, g0_frag_mol)
				emol = Chem.EditableMol(combo)

				offset = custom_mol.GetNumAtoms()
				emol_rad_idx = rad_idx_in_frag + offset

				# Decrease bond order in the custom molecule
				original_bond = emol.GetBondBetweenAtoms(target_atom_idx, other_atom_idx)
				original_order = original_bond.GetBondType()
				new_order = Chem.BondType.SINGLE if original_order == Chem.BondType.DOUBLE else Chem.BondType.DOUBLE
				emol.SetBondType(target_atom_idx, other_atom_idx, new_order)

				emol.AddBond(target_atom_idx, emol_rad_idx, order=Chem.BondType.SINGLE)

				product_mol = emol.GetMol()

				# Update radical electrons
				# The G0 radical is quenched. The "other" atom of the bond becomes the new radical.
				product_mol.GetAtomWithIdx(emol_rad_idx).SetNumRadicalElectrons(0)
				product_mol.GetAtomWithIdx(other_atom_idx).SetNumRadicalElectrons(1)

				try:
					Chem.SanitizeMol(product_mol)
					product_smi = Chem.MolToSmiles(product_mol, canonical=True)

					# Create canonical reaction SMILES
					reacts_smi = ".".join(sorted([g0_frag_smi, custom_mol_smi]))
					reaction = f"{reacts_smi}>>{product_smi}"
					r = element_counts(reacts_smi)
					p = element_counts(product_smi)
					if r == p:
						new_reactions.append(reaction)
				except Exception:
					continue

	except Exception:
		return []

	return list(set(new_reactions))


def worker_generate_new_reactions_g1(reaction_pair: Tuple[str, str]) -> List[str]:
	"""
	Worker function to generate G1 reactions from pairs of G0 reactions.
	Implements both transfer and rearrangement reactions as described in the paper.

	1. Transfer reactions: Given A -> B + C and D -> B + E, create A + E -> D + C
	         (One bond breaks, one bond forms, bimolecular)

	2. Rearrangement reactions: Given A -> B + C and D -> B + C, create A -> D
	         (One bond breaks, one bond forms, unimolecular)
	"""
	r1_str, r2_str = reaction_pair

	try:
		# Parse reactions: "Parent>>Frag1.Frag2"
		p1, frags1_str = r1_str.split(">>")
		p2, frags2_str = r2_str.split(">>")

		# Split fragments
		frags1 = frags1_str.split(".")
		frags2 = frags2_str.split(".")

		frags1_set = set(frags1)
		frags2_set = set(frags2)

		new_reactions = []

		# Find common fragments between the two reactions
		common_frags = frags1_set & frags2_set

		if len(common_frags) == 0:
			return []

		# Case 1: Rearrangement reaction A -> D
		# Where A -> B + C and D -> B + C (same products)
		if p1 != p2 and sorted(frags1) == sorted(frags2):
			parents = sorted([p1, p2])

			r_cans = canonicalize_smiles(parents[0])
			p_cans = canonicalize_smiles(parents[1])

			r = element_counts(r_cans)
			p = element_counts(p_cans)

			if r_cans != p_cans and r == p:
				reaction_smi = f"{r_cans}>>{p_cans}"
				new_reactions.append(reaction_smi)

		# Case 2: Transfer reaction A + E -> D + C
		# Where A -> B + C and D -> B + E, B is common
		elif len(common_frags) == 1:
			common_frag = next(iter(common_frags))

			temp_frags1 = list(frags1)
			temp_frags2 = list(frags2)

			try:
				# Remove one instance of the common fragment
				temp_frags1.remove(common_frag)
				temp_frags2.remove(common_frag)
			except ValueError:
				# This should not happen if common_frags was derived from these lists
				logging.warning(f"Fragment removal error for {reaction_pair}")
				return []

			if len(temp_frags1) == 1 and len(temp_frags2) == 1:
				unique_frag1 = temp_frags1[0]
				unique_frag2 = temp_frags2[0]

				reactants = canonicalize_smiles_list(sorted([p1, unique_frag2]))
				products = canonicalize_smiles_list(sorted([p2, unique_frag1]))

				# print(unique_frag1, unique_frag2)
				# print(reactants, products)
				# print(r1_str, r2_str)

				if verify_reaction(reactants, products):
					reactants_str = ".".join(reactants)
					products_str = ".".join(products)
					reaction_smi = f"{reactants_str}>>{products_str}"

					r = element_counts(reactants_str)
					p = element_counts(products_str)
					if r == p:
						new_reactions.append(reaction_smi)

		return new_reactions

	except Exception as e:
		logging.warning(f"Error processing reaction pair {reaction_pair}: {e}")
		return []


def worker_generate_higher_gen_reactions(reaction_pair: Tuple[str, str], max_reaction_complexity: int = 3) -> List[str]:
	"""
	Worker function to generate higher generation reactions (G2+) from any two reactions.
	"""
	r1_str, r2_str = reaction_pair

	try:
		r1_reactants_str, r1_products_str = r1_str.split(">>")
		r2_reactants_str, r2_products_str = r2_str.split(">>")

		r1_reactants = set(r1_reactants_str.split("."))
		r1_products = set(r1_products_str.split("."))
		r2_reactants = set(r2_reactants_str.split("."))
		r2_products = set(r2_products_str.split("."))

		new_reactions = []

		# Case 1: Product of r1 is reactant of r2
		# A -> B + C and B + D -> E + F creates A + D -> C + E + F
		shared_r1p_r2r = r1_products & r2_reactants
		for shared in shared_r1p_r2r:
			remaining_r1_reactants = r1_reactants
			remaining_r1_products = r1_products - {shared}
			remaining_r2_reactants = r2_reactants - {shared}
			remaining_r2_products = r2_products

			new_reactants = remaining_r1_reactants | remaining_r2_reactants
			new_products = remaining_r1_products | remaining_r2_products

			new_reactants = canonicalize_smiles_list(list(new_reactants))
			new_products = canonicalize_smiles_list(list(new_products))

			if not verify_reaction(new_reactants, new_products):
				continue

			if (
				1 <= len(new_reactants) <= max_reaction_complexity
				and 1 <= len(new_products) <= max_reaction_complexity
				and new_reactants != new_products
			):
				reactants_str = ".".join(sorted(new_reactants))
				products_str = ".".join(sorted(new_products))
				reaction_smi = f"{reactants_str}>>{products_str}"
				r = element_counts(reactants_str)
				p = element_counts(products_str)
				if r == p:
					new_reactions.append(reaction_smi)

		# Case 2: Product of r2 is reactant of r1
		# A + B -> C + D and C -> E + F creates A + B + E -> D + F
		shared_r2p_r1r = r2_products & r1_reactants
		for shared in shared_r2p_r1r:
			remaining_r1_reactants = r1_reactants - {shared}
			remaining_r1_products = r1_products
			remaining_r2_reactants = r2_reactants
			remaining_r2_products = r2_products - {shared}

			new_reactants = remaining_r1_reactants | remaining_r2_reactants
			new_products = remaining_r1_products | remaining_r2_products

			new_reactants = canonicalize_smiles_list(list(new_reactants))
			new_products = canonicalize_smiles_list(list(new_products))

			if not verify_reaction(new_reactants, new_products):
				continue
			if (
				1 <= len(new_reactants) <= max_reaction_complexity
				and 1 <= len(new_products) <= max_reaction_complexity
				and new_reactants != new_products
			):
				reactants_str = ".".join(sorted(new_reactants))
				products_str = ".".join(sorted(new_products))
				reaction_smi = f"{reactants_str}>>{products_str}"
				r = element_counts(reactants_str)
				p = element_counts(products_str)
				if r == p:
					new_reactions.append(reaction_smi)

		# Case 3: Reactants share species (parallel reactions with common reactant)
		# A + B -> C + D and A + E -> F + G creates B + E -> C + D + F + G (removing A)
		shared_reactants = r1_reactants & r2_reactants
		for shared in shared_reactants:
			remaining_r1_reactants = r1_reactants - {shared}
			remaining_r2_reactants = r2_reactants - {shared}

			new_reactants = remaining_r1_reactants | remaining_r2_reactants
			new_products = r1_products | r2_products

			new_reactants = canonicalize_smiles_list(list(new_reactants))
			new_products = canonicalize_smiles_list(list(new_products))
			if not verify_reaction(new_reactants, new_products):
				continue
			if (
				1 <= len(new_reactants) <= max_reaction_complexity
				and 1 <= len(new_products) <= max_reaction_complexity
				and new_reactants != new_products
			):
				reactants_str = ".".join(sorted(new_reactants))
				products_str = ".".join(sorted(new_products))
				reaction_smi = f"{reactants_str}>>{products_str}"
				r = element_counts(reactants_str)
				p = element_counts(products_str)
				if r == p:
					new_reactions.append(reaction_smi)

		return new_reactions

	except Exception:
		return []


rxn_mapper_instance = None


def worker_verify_reaction_batch(
	reaction_smi_bytes_batch: List[bytes],
	confidence_threshold: float = 0.9,
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
	reactions_gen = []
	original_reactions_map = {}

	for i, (reaction_smi, gen) in enumerate(reaction_smi_bytes_batch):
		try:
			reactants_str, products_str = reaction_smi.split(">>")
			reactant_mols = [Chem.MolFromSmiles(s) for s in reactants_str.split(".")]
			product_mols = [Chem.MolFromSmiles(s) for s in products_str.split(".")]

			if any(m is None for m in reactant_mols + product_mols):
				continue

			r_cans = sorted([Chem.MolToSmiles(m, canonical=True) for m in reactant_mols])
			p_cans = sorted([Chem.MolToSmiles(m, canonical=True) for m in product_mols])

			if r_cans == p_cans:
				continue

			final_canonical_reaction = f"{'.'.join(r_cans)}>>{'.'.join(p_cans)}"
			original_reactions_map[reaction_smi] = final_canonical_reaction
			reactions_to_map.append(reaction_smi)
			reactions_gen.append(gen)
		except Exception as e:
			print(e)
			continue

	if not reactions_to_map:
		return []

	# try:
	#     results = rxn_mapper_instance.get_attention_guided_atom_maps(reactions_to_map)

	#     for original_smi, gen, result in zip(reactions_to_map, reactions_gen, results):
	#         confidence = result.get('confidence', 0.0)
	#         mapped_rxn = result.get('mapped_rxn', "")

	#         if confidence >= confidence_threshold:
	#             canonical_form = original_reactions_map.get(original_smi)
	#             if canonical_form:
	#                 valid_reactions.append((canonical_form, gen))

	# except Exception as e:
	#     logging.warning(f"rxnmapper failed for a batch: {e}")

	for original_smi, gen in zip(reactions_to_map, reactions_gen):
		canonical_form = original_reactions_map.get(original_smi)
		if canonical_form:
			valid_reactions.append((canonical_form, gen))

	return valid_reactions
