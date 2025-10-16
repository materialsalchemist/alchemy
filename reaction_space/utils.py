from typing import Set, Tuple, List
from rdkit import Chem, RDLogger

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='rxnmapper')

# Suppress verbose RDKit logging
RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(s):
	mol = Chem.MolFromSmiles(s)

	if mol is not None:
		mol = Chem.RemoveHs(mol)
		return Chem.MolToSmiles(mol, canonical=True)

	return s

def canonicalize_smiles_list(smiles_list):
	return sorted([
		canonicalize_smiles(s)
		for s in smiles_list
		if Chem.MolFromSmiles(s) is not None
	])
