from rdkit import Chem
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MolConfig:
	"""Configuration for molecular properties."""
	BOND_OPTIONS: tuple = (
		Chem.BondType.ZERO,
		Chem.BondType.SINGLE,
		Chem.BondType.DOUBLE,
		Chem.BondType.TRIPLE,
	)
	BOND_VALUES: dict = field(default_factory=lambda: {
		Chem.BondType.ZERO: 0.0,
		Chem.BondType.SINGLE: 1.0,
		Chem.BondType.DOUBLE: 2.0,
		Chem.BondType.TRIPLE: 3.0,
	})
	VALENCE_OPTIONS: tuple = (0, 1, 2, 3, 4)
	RADICAL_OPTIONS: tuple = (0, 1, 2, 3, 4)
