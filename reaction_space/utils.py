from typing import Set, Tuple, List, Dict
import itertools
from rdkit import Chem, RDLogger
from collections import Counter

import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="rxnmapper")

# Suppress verbose RDKit logging
RDLogger.DisableLog("rdApp.*")

RADICAL_THRESHOLD = 2


def mol_to_explicit_smiles(mol, canonical=True) -> str:
	if not mol:
		return ""

	mol_copy = Chem.Mol(mol)

	for atom in mol_copy.GetAtoms():
		num_implicit_hs = atom.GetNumImplicitHs()

		if num_implicit_hs > 0:
			atom.SetNumExplicitHs(num_implicit_hs)
			atom.SetNoImplicit(True)

	return Chem.MolToSmiles(mol_copy, canonical=canonical)


def canonicalize_smiles(s):
	mol = Chem.MolFromSmiles(s)
	if mol is None:
		return s
	Chem.SanitizeMol(mol)  # Ensure the molecule is sanitized
	return mol_to_explicit_smiles(mol) if mol else s


def canonicalize_smiles_list(smiles_list):
	return sorted([canonicalize_smiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None])


def add_atomic_compositions(smiles_list: List[str]) -> Dict[str, int]:
	"""Return a single aggregated atomic composition dictionary for the input SMILES list.

	The returned dictionary maps atomic symbol (e.g. 'C', 'O') to the total count
	summed over all SMILES in the input list. Note: implicit hydrogens are not
	counted because they are not present as explicit atoms in the RDKit Mol
	unless hydrogens are added beforehand (use Chem.AddHs to include them).
	"""

	total_counts: Dict[str, int] = {}

	for smi in smiles_list:
		mol = Chem.MolFromSmiles(smi)

		if mol is None:
			raise Exception(f"Invalid SMILES string: {smi}")

		for atom in mol.GetAtoms():
			sym = atom.GetSymbol()
			total_counts[sym] = total_counts.get(sym, 0) + 1

	return total_counts


def is_bare_atom_smiles(smiles: str) -> bool:
	"""Return True for standalone unhydrogenated atoms such as [C], [O], [N], or [H]."""
	mol = Chem.MolFromSmiles(smiles)
	if mol is None or mol.GetNumAtoms() != 1 or mol.GetNumBonds() != 0:
		return False
	atom = mol.GetAtomWithIdx(0)
	return atom.GetTotalNumHs() == 0


def contains_bare_atom_species(smiles_list: List[str]) -> bool:
	"""Return True if any molecule in a reaction side is a standalone atom species."""
	return any(is_bare_atom_smiles(smi) for smi in smiles_list)


def total_radical_electrons(smiles_list: List[str]) -> int:
	"""Return total radical electrons across a list of molecule SMILES."""
	total = 0
	for smi in smiles_list:
		mol = Chem.MolFromSmiles(smi)
		if mol is None:
			continue
		total += sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
	return total


def exceeds_radical_threshold(reactants: List[str], products: List[str], threshold: int = RADICAL_THRESHOLD) -> bool:
	"""Return True if either reaction side has more radical electrons than threshold."""
	return max(total_radical_electrons(reactants), total_radical_electrons(products)) > threshold


def verify_reaction(reactants: List[str], products: List[str]) -> bool:
	"""Verify if a reaction is valid based on atomic composition and overlap.

	Returns True if the reaction is valid (same atomic composition and no overlap
	between reactants and products), otherwise returns False.
	"""

	if add_atomic_compositions(reactants) != add_atomic_compositions(products):
		return False

	if len(set(reactants).intersection(set(products))) != 0:
		return False

	return True


def mol_from_smiles_with_H(smiles: str) -> Chem.Mol:
	"""Parse SMILES, sanitize, and make hydrogens explicit."""
	smiles = (smiles or "").strip()
	if not smiles:
		raise ValueError("Empty SMILES")

	mol = Chem.MolFromSmiles(smiles, sanitize=True)
	if mol is None:
		raise ValueError(f"Invalid SMILES: {smiles}")

	# Remove atom mapping if present
	for a in mol.GetAtoms():
		if a.HasProp("molAtomMapNumber"):
			a.ClearProp("molAtomMapNumber")

	# IMPORTANT: make implicit H atoms explicit
	mol = Chem.AddHs(mol)
	return mol


def element_counts(side_smiles: str) -> Counter:
	"""Count atoms by element (including H) on one side of a reaction."""
	total = Counter()
	side_smiles = (side_smiles or "").strip()
	if not side_smiles:
		return total

	for part in side_smiles.split("."):
		part = part.strip()
		if not part:
			continue
		mol = mol_from_smiles_with_H(part)
		for atom in mol.GetAtoms():
			total[atom.GetSymbol()] += 1

	return total


def chunked_iterable(iterable, size):
	"""Yield successive chunks from an iterable."""
	it = iter(iterable)
	while True:
		chunk = list(itertools.islice(it, size))
		if not chunk:
			break
		yield chunk
