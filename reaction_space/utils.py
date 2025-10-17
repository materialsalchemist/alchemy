from typing import Set, Tuple, List, Dict
from rdkit import Chem, RDLogger

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='rxnmapper')

# Suppress verbose RDKit logging
RDLogger.DisableLog('rdApp.*')

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

	return mol_to_explicit_smiles(mol) if mol else smi

def canonicalize_smiles_list(smiles_list):
	return sorted([
		canonicalize_smiles(s)
		for s in smiles_list
		if Chem.MolFromSmiles(s) is not None
	])

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
