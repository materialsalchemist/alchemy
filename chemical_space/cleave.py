# cleave.py
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Literal

from rdkit import Chem


Mode = Literal["radical", "cap_h", "ionic"]


@dataclass(frozen=True)
class CleavageResult:
    cut_bond_indices: Tuple[int, ...]
    fragments: Tuple[str, ...]  # canonical SMILES (explicit H), sorted


def _sanitize_lenient(m: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        try:
            Chem.SanitizeMol(
                m,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
            return m
        except Exception:
            return None


def _canonical_smiles_explicit_h(m: Chem.Mol) -> str:
    """
    Canonical SMILES with explicit H atoms.
    """
    return Chem.MolToSmiles(m, canonical=True, allHsExplicit=True)


def _remove_dummies_keep_h(m: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Remove FragmentOnBonds dummy atoms ([#0]) but keep *real* H atoms.
    """
    dummy_q = Chem.MolFromSmarts("[#0]")
    m2 = Chem.DeleteSubstructs(m, dummy_q)
    return _sanitize_lenient(m2)


def _dummies_to_radicals_then_remove(m: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Homolytic cleavage:
      - each cut end gets +1 radical electron
      - dummy atoms removed
    Works for both heavy-heavy and heavy-H bonds (when include_h_bonds=True).
    """
    rw = Chem.RWMol(m)
    dummy_idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]

    neighbor_idxs: List[int] = []
    for d_idx in dummy_idxs:
        d_atom = rw.GetAtomWithIdx(d_idx)
        nbrs = list(d_atom.GetNeighbors())
        if len(nbrs) != 1:
            return None
        neighbor_idxs.append(nbrs[0].GetIdx())

    for n_idx in neighbor_idxs:
        a = rw.GetAtomWithIdx(n_idx)
        a.SetNumRadicalElectrons(a.GetNumRadicalElectrons() + 1)

    # remove dummies
    for d_idx in sorted(dummy_idxs, reverse=True):
        rw.RemoveAtom(d_idx)

    out = rw.GetMol()
    return _sanitize_lenient(out)


def _dummies_cap_h_then_remove(m: Chem.Mol) -> Optional[Chem.Mol]:
    """
    "cap_h" mode:
      - remove dummies
      - sanitize; RDKit satisfies valence with implicit H
      - then AddHs() to make those H explicit atoms
    NOTE: Cutting X–H bonds in cap_h often returns the parent (chemically expected).
    """
    out = _remove_dummies_keep_h(m)
    if out is None:
        return None
    # ensure any implicit H introduced becomes explicit atoms
    out = Chem.AddHs(out, addCoords=False)
    return _sanitize_lenient(out)


def _dummies_to_ions_then_remove(m: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Heterolytic ionic cleavage (heuristic, EN-based):
      - more electronegative side becomes anion (-1)
      - other side becomes cation (+1)
    Then remove dummies and AddHs to make H explicit.
    """
    EN = {
        "F": 3.98, "O": 3.44, "Cl": 3.16, "N": 3.04, "Br": 2.96, "I": 2.66,
        "S": 2.58, "C": 2.55, "P": 2.19, "H": 2.20, "Si": 1.90,
        "B": 2.04, "Li": 0.98, "Na": 0.93, "K": 0.82, "Mg": 1.31, "Al": 1.61,
        "Zn": 1.65,
    }

    rw = Chem.RWMol(m)
    dummy_idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]

    neighbor_info: List[Tuple[int, float]] = []
    for d_idx in dummy_idxs:
        d_atom = rw.GetAtomWithIdx(d_idx)
        nbrs = list(d_atom.GetNeighbors())
        if len(nbrs) != 1:
            return None
        nb = nbrs[0]
        en = EN.get(nb.GetSymbol(), 0.0)
        neighbor_info.append((d_idx, en))

    if not neighbor_info:
        return None

    max_en = max(en for _, en in neighbor_info)

    for d_idx, en in neighbor_info:
        nb = list(rw.GetAtomWithIdx(d_idx).GetNeighbors())[0]
        if en == max_en and max_en > 0.0:
            nb.SetFormalCharge(nb.GetFormalCharge() - 1)
        else:
            nb.SetFormalCharge(nb.GetFormalCharge() + 1)

    for d_idx in sorted(dummy_idxs, reverse=True):
        rw.RemoveAtom(d_idx)

    out = rw.GetMol()
    out = _sanitize_lenient(out)
    if out is None:
        return None

    out = Chem.AddHs(out, addCoords=False)
    return _sanitize_lenient(out)


def _process_fragment_by_mode(frag: Chem.Mol, mode: Mode) -> Optional[Chem.Mol]:
    if mode == "radical":
        # keep explicit H already present from include_h_bonds
        return _dummies_to_radicals_then_remove(frag)
    if mode == "cap_h":
        return _dummies_cap_h_then_remove(frag)
    if mode == "ionic":
        return _dummies_to_ions_then_remove(frag)
    raise ValueError(f"Unknown mode: {mode}")


def _fragment_smiles_from_cut(
    mol: Chem.Mol, bond_indices: Sequence[int], mode: Mode
) -> Optional[Tuple[str, ...]]:
    if not bond_indices:
        m0 = _sanitize_lenient(Chem.Mol(mol))
        if m0 is None:
            return None
        return tuple(sorted({_canonical_smiles_explicit_h(m0)}))

    cut_mol = Chem.FragmentOnBonds(mol, list(bond_indices), addDummies=True)
    frags = Chem.GetMolFrags(cut_mol, asMols=True, sanitizeFrags=False)

    out_smiles: List[str] = []
    for f in frags:
        f2 = _process_fragment_by_mode(f, mode)
        if f2 is None:
            return None
        out_smiles.append(_canonical_smiles_explicit_h(f2))

    out_smiles.sort()
    return tuple(out_smiles)


def enumerate_cleavage_products(
    smiles: str,
    *,
    max_cuts: int = 2,
    include_ring_bonds: bool = True,
    include_h_bonds: bool = True,
    mode: Mode = "radical",
) -> List[CleavageResult]:
    """
    Enumerate cleavage products for combinations of up to `max_cuts` broken bonds.

    include_h_bonds:
      - False: cleave only heavy-atom bonds (typical SMILES graph)
      - True : use Chem.AddHs() to create explicit H atoms and X–H bonds, so
               H-bond cleavage is included.
    Output:
      - fragment SMILES are canonical with explicit H atoms (allHsExplicit=True)
    """
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol0, addCoords=False) if include_h_bonds else Chem.Mol(mol0)
    mol = _sanitize_lenient(mol)
    if mol is None:
        raise ValueError(f"Could not sanitize molecule from SMILES: {smiles}")

    candidate_bonds: List[int] = []
    for b in mol.GetBonds():
        if not include_ring_bonds and b.IsInRing():
            continue
        candidate_bonds.append(b.GetIdx())

    results: List[CleavageResult] = []
    seen_fragment_sets = set()

    for k in range(1, max_cuts + 1):
        for bond_combo in itertools.combinations(candidate_bonds, k):
            fragset = _fragment_smiles_from_cut(mol, bond_combo, mode=mode)
            if fragset is None:
                continue
            if fragset in seen_fragment_sets:
                continue
            seen_fragment_sets.add(fragset)
            results.append(CleavageResult(tuple(bond_combo), fragset))

    return results
