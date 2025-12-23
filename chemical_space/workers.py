import itertools
import networkx as nx
from rdkit import Chem
import pickle
from collections import Counter
from typing import Tuple


def worker_process_compositions(task_tuple, n_atoms, allowed_valences_map, config):
    composition, bonds = task_tuple
    plausible_combinations = []

    # Remove fragmented molecules
    if n_atoms > 1:
        G = nx.Graph()
        G.add_nodes_from(range(n_atoms))

        atom_pairs = list(itertools.combinations(range(n_atoms), 2))

        for idx, (i, j) in enumerate(atom_pairs):
            if bonds[idx] != Chem.BondType.ZERO:
                G.add_edge(i, j)

        if not nx.is_connected(G):
            return []

    # Calculate sum of bond orders for each atom to filter out invalid composition
    # and find valid radical/hydrogen combinations
    bond_sum = [0] * n_atoms
    if n_atoms > 1:
        atom_pairs = list(itertools.combinations(range(n_atoms), 2))
        for idx, (i, j) in enumerate(atom_pairs):
            v = config.BOND_VALUES.get(bonds[idx], 0.0)
            bond_sum[i] += v
            bond_sum[j] += v

    for radicals in itertools.product(config.RADICAL_OPTIONS, repeat=n_atoms):
        valences = []
        is_plausible = True
        for i, sym in enumerate(composition):
            needed_hs = None
            for total_valence in allowed_valences_map.get(sym, []):
                h = int(total_valence - bond_sum[i] - radicals[i])

                if h in config.VALENCE_OPTIONS:
                    needed_hs = h
                    break

            if needed_hs is None:
                is_plausible = False
                break

            valences.append(needed_hs)

        if is_plausible:
            plausible_combinations.append(
                (composition, bonds, tuple(valences), radicals)
            )

    return plausible_combinations


def worker_build_molecule(data: bytes) -> Tuple[str, bytes] | None:
    try:
        composition, bonds, valences, radicals = pickle.loads(data)
        mol = Chem.RWMol()
        for i, comp in enumerate(composition):
            atom = Chem.Atom(comp)
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(radicals[i])
            atom.SetNumExplicitHs(valences[i])
            mol.AddAtom(atom)

        if len(composition) > 1:
            atom_indices = range(len(composition))
            for (i, j), bond_type in zip(
                itertools.combinations(atom_indices, 2), bonds
            ):
                if bond_type != Chem.BondType.ZERO:
                    mol.AddBond(i, j, bond_type)

        Chem.SanitizeMol(mol)

        # Final check for fragments which can occur after sanitization
        # May able to remove this, since we already check this
        # with networkx in generating composition
        if len(Chem.rdmolops.GetMolFrags(mol)) > 1:
            return None

        smi = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
        mol = Chem.AddHs(mol)
        atom_counts = dict(Counter(atom.GetSymbol() for atom in mol.GetAtoms()))

        return (smi, pickle.dumps((mol, atom_counts)))
    except Exception:
        return None
