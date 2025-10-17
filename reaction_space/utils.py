from typing import Set, Tuple, List
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
