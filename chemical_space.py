import subprocess
import networkx as nx
import itertools
from utilities import Utilities
import networkx as nx
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable, Draw
from tqdm import tqdm

import os
import pandas as pd
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
class ChemicalSpace:
    max_atoms: int = field(default=4)
    atoms: FrozenSet[str] = field(default_factory=lambda: frozenset(["C", "H", "O"]))
    bond_types: FrozenSet[Chem.BondType] = field(
        default_factory=lambda: frozenset(
            [
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                # Chem.BondType.AROMATIC,
                # Chem.BondType.QUADRUPLE,
                # Chem.BondType.QUINTUPLE,
                # Chem.BondType.HEXTUPLE,
                # Chem.BondType.ONEANDAHALF,
                # Chem.BondType.TWOANDAHALF,
                # Chem.BondType.THREEANDAHALF,
                # Chem.BondType.FOURANDAHALF,
                # Chem.BondType.FIVEANDAHALF,
                # Chem.BondType.IONIC,
                # Chem.BondType.HYDROGEN,
                # Chem.BondType.THREECENTER,
                # Chem.BondType.DATIVEONE,
                # Chem.BondType.DATIVE,
                # Chem.BondType.DATIVEL,
                # Chem.BondType.DATIVER,
                Chem.BondType.ZERO,
                # Chem.BondType.OTHER,
            ]
        )
    )

    molecules: Set[str] = field(init=False, default_factory=set)
    n_workers: int = field(default=4)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self):
        """
        Initialize derived attributes and set up logging after instance creation.
        """
        self.heavy_atoms = [atom for atom in self.atoms if atom != "H"]
        self.periodic_table = Chem.GetPeriodicTable()
        Utilities._setup_logging()

    @staticmethod
    def _is_valid_molecule(smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol)
            return True
        except:
            return False

    def generate_compositions(self):
        """Generates all valid compositions of atoms up to max_atoms, ensuring order independence."""
        atom_list = list(self.heavy_atoms)
        compositions = []

        for total_atoms in range(1, self.max_atoms + 1):
            for composition in itertools.combinations_with_replacement(
                atom_list, total_atoms
            ):
                num_bonds = int(len(composition) * (len(composition) - 1) / 2)
                for possible_bonds in itertools.product(
                    self.bond_types, repeat=num_bonds
                ):
                    for possible_valences in itertools.product(
                        [0, 1, 2, 3, 4], repeat=len(composition)
                    ):
                        for possible_radicals in itertools.product(
                            [0, 1, 2, 3, 4], repeat=len(composition)
                        ):
                            compositions += [
                                [
                                    composition,
                                    possible_bonds,
                                    possible_valences,
                                    possible_radicals,
                                ]
                            ]

        # compositions += [(("H",), (), ()), (("H", "H"), (Chem.BondType.SINGLE), ())]
        return compositions

    def composition_to_mol(
        self, composition, possible_bonds, possible_valence, possible_radicals
    ):
        """Constructs a molecule from atom composition and a given bond combination."""
        g = nx.Graph()
        g.add_nodes_from(range(len(composition)))

        mol = Chem.RWMol()
        for i in range(len(composition)):
            mol.AddAtom(Chem.Atom(composition[i]))
            mol.GetAtomWithIdx(i).SetNoImplicit(False)
            mol.GetAtomWithIdx(i).SetNumRadicalElectrons(possible_radicals[i])
            mol.GetAtomWithIdx(i).SetNumExplicitHs(possible_valence[i])

        bonds_index = 0
        for i in range(len(composition)):
            for j in range(i + 1, len(composition)):
                if possible_bonds[bonds_index] != Chem.BondType.ZERO:
                    mol.AddBond(i, j, possible_bonds[bonds_index])
                    g.add_edge(i, j)
                bonds_index += 1

        if len(list(nx.connected_components(g))) > 1:
            return None
        try:
            mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            return mol
        except Exception as e:
            logging.debug(f"Sanitization failed: {e}")
            # print(mol, "is wrong")
            return None

    def generate_molecules(self):
        self.molecules.clear()
        compositions = list(self.generate_compositions())
        for c in compositions:
            mol = self.composition_to_mol(c[0], c[1], c[2], c[3])
            # print(c)
            if mol:
                smi = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
                if self._is_valid_molecule(smi):
                    self.molecules.add(smi)
        logging.info(f"Generated {len(self.molecules)} valid molecules")

    def save_molecules_to_csv_and_images(self, dir_path):
        molecules = sorted(self.molecules)
        pd.DataFrame(molecules).to_csv(f"{dir_path}/molecules.csv")

        os.makedirs(f"{dir_path}/molecule_images", exist_ok=True)
        os.makedirs(f"{dir_path}/molecule_xyz", exist_ok=True)
        os.makedirs(f"{dir_path}/molecule_images_obabel", exist_ok=True)

        for i, smi in enumerate(molecules):
            Draw.MolsToImage([Chem.MolFromSmiles(smi)]).save(
                f"{dir_path}/molecule_images/mol_{i}.png"
            )
            subprocess.run(
                [
                    "obabel",
                    "-:" + smi,
                    "-O" + f"{dir_path}/molecule_images_obabel/mol_{i}.png",
                ]
            )
            subprocess.run(
                [
                    "obabel",
                    "-:" + smi,
                    "-h",
                    "--gen3D",
                    "-O" + f"{dir_path}/molecule_xyz/mol_{i}.xyz",
                ]
            )


if __name__ == "__main__":
    import argparse

    # from rdkit import RDLogger
    #
    # RDLogger.DisableLog("rdApp.error")

    parser = argparse.ArgumentParser(
        prog="crn_py",
    )
    parser.add_argument("-n", "--max_atoms", type=int, default=2)
    args = parser.parse_args()

    chemical_space = ChemicalSpace(max_atoms=args.max_atoms, n_workers=12)

    print("Generating molecules...")
    m = Chem.RWMol()
    m.AddAtom(Chem.Atom("C"))
    m.AddAtom(Chem.Atom("O"))

    m.AddBond(0, 1, Chem.BondType.ZERO)
    m.GetAtomWithIdx(0).SetNumRadicalElectrons(1)
    m.GetAtomWithIdx(1).SetNumRadicalElectrons(0)
    m.GetAtomWithIdx(0).SetNoImplicit(False)
    m.GetAtomWithIdx(1).SetNoImplicit(False)
    m.GetAtomWithIdx(0).SetNumExplicitHs(1)
    m.GetAtomWithIdx(1).SetNumExplicitHs(0)
    Chem.SanitizeMol(m)
    # params = Chem.SmilesParserParams()
    # params.removeHs = False
    # m = Draw.PrepareMolForDrawing(m, addChiralHs=True)
    # m = Chem.MolFromSmiles(Chem.MolToSmiles(m), params)
    # Draw.MolsToImage([m], explicitMethyl=True).save("test.png")
    print(Chem.MolToSmiles(m))
    print(Chem.MolToMolBlock(m))
    subprocess.run(
        ["obabel", "-:" + Chem.MolToSmiles(m), "-h", "--gen3D", "-Otest.xyz"]
    )
    subprocess.run(["obabel", "-:" + Chem.MolToSmiles(m), "-Otest.png"])

    g = nx.Graph()
    g.add_nodes_from(range(0, 2))
    g.add_edge(0, 1)
    d = list(nx.connected_components(g))
    print(d)

    chemical_space.generate_molecules()
    print(chemical_space.molecules)
    try:
        os.mkdir(str(args.max_atoms))
    except:
        pass
    chemical_space.save_molecules_to_csv_and_images(str(args.max_atoms))
