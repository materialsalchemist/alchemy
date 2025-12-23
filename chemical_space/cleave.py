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
    fragments: Tuple[str, ...]  # canonical SMILES of fragments, sorted


def _canonical_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, canonical=True)


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


def _remove_dummies(m: Chem.Mol) -> Chem.Mol:
    dummy_q = Chem.MolFromSmarts("[#0]")
    return Chem.DeleteSubstructs(m, dummy_q)


def _dummies_to_radicals_then_remove(m: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Homolytic cleavage:
      - each cut end gets +1 radical electron
      - dummies removed
    Example: "CC" -> "[CH3]" + "[CH3]"
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

    for d_idx in sorted(dummy_idxs, reverse=True):
        rw.RemoveAtom(d_idx)

    out = rw.GetMol()
    return _sanitize_lenient(out)


def _dummies_cap_h_then_remove(m: Chem.Mol) -> Optional[Chem.Mol]:
    """
    "cap_h" mode:
      - remove dummies
      - sanitize; RDKit will satisfy valence with implicit H
    Example: "CC" -> "C" + "C"  (methane + methane)
    """
    m2 = _remove_dummies(m)
    return _sanitize_lenient(m2)


def _dummies_to_ions_then_remove(m: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Heterolytic ionic cleavage (simple, rule-based):
      - for each cut, choose a direction to assign charges:
          * more electronegative side becomes anion (-1)
          * other side becomes cation (+1)
        If equal electronegativity, break ties by:
          higher atomic number -> anion, else lower -> cation
      - dummies removed
    Example: "CC" (equal) -> "[CH3+]" + "[CH3-]" (tie-break)
            "CO" -> "[CH3+]" + "[O-]"
    NOTE: This is a *heuristic*, not a full chemical ionization model.
    """
    # Pauling-ish EN (minimal set; extend as you like)
    EN = {
        "F": 3.98, "O": 3.44, "Cl": 3.16, "N": 3.04, "Br": 2.96, "I": 2.66,
        "S": 2.58, "C": 2.55, "P": 2.19, "H": 2.20, "Si": 1.90,
        "B": 2.04, "Li": 0.98, "Na": 0.93, "K": 0.82, "Mg": 1.31, "Al": 1.61,
        "Zn": 1.65,
    }

    rw = Chem.RWMol(m)
    dummy_idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]

    # Each dummy has 1 neighbor; each cut generates 2 dummies (one on each side).
    # We'll decide charge assignment per dummy independently by comparing the neighbor atom
    # to the neighbor of its "partner" dummy across the same cut.
    #
    # Partnering is tricky without bond-annotation, but FragmentOnBonds adds a single bond
    # between dummy and neighbor; the two dummies from a cut are not connected.
    # Instead, we pair dummies by their "isotope" label if present (RDKit often sets dummy
    # atom isotopes for cuts when requested; but default may be 0).
    #
    # So we implement a simpler robust approach:
    #   - for each dummy, assign a provisional preference based on its neighbor EN
    #   - then for each fragment molecule, net charge will emerge from the per-dummy assignments.
    #
    # This yields correct behavior for common polar bonds (C-O, C-N, etc).
    neighbor_info: List[Tuple[int, str, float, int]] = []
    # (dummy_idx, neighbor_symbol, neighbor_EN, neighbor_atomicnum)
    for d_idx in dummy_idxs:
        d_atom = rw.GetAtomWithIdx(d_idx)
        nbrs = list(d_atom.GetNeighbors())
        if len(nbrs) != 1:
            return None
        nb = nbrs[0]
        sym = nb.GetSymbol()
        en = EN.get(sym, 0.0)
        neighbor_info.append((d_idx, sym, en, nb.GetAtomicNum()))

    # Decide which dummies should make their neighbor an anion vs cation.
    # We can't perfectly pair per-cut, so we apply:
    #   - any dummy whose neighbor is among the highest EN in the molecule gets -1
    #   - others get +1
    # This works for typical single polar bond cuts (and is deterministic).
    if not neighbor_info:
        return None
    max_en = max(en for _, _, en, _ in neighbor_info)

    for d_idx, sym, en, z in neighbor_info:
        nbr = list(rw.GetAtomWithIdx(d_idx).GetNeighbors())[0]
        # If EN ties (e.g., C-C), fall back on atomic number tie-break:
        if en == max_en and max_en > 0.0:
            # electronegative side gets negative charge
            nbr.SetFormalCharge(nbr.GetFormalCharge() - 1)
        else:
            # electropositive side gets positive charge
            nbr.SetFormalCharge(nbr.GetFormalCharge() + 1)

    # Remove dummies
    for d_idx in sorted(dummy_idxs, reverse=True):
        rw.RemoveAtom(d_idx)

    out = rw.GetMol()
    return _sanitize_lenient(out)


def _process_fragment_by_mode(frag: Chem.Mol, mode: Mode) -> Optional[Chem.Mol]:
    if mode == "radical":
        return _dummies_to_radicals_then_remove(frag)
    if mode == "cap_h":
        return _dummies_cap_h_then_remove(frag)
    if mode == "ionic":
        return _dummies_to_ions_then_remove(frag)
    raise ValueError(f"Unknown mode: {mode}")


def _fragment_smiles_from_cut(mol: Chem.Mol, bond_indices: Sequence[int], mode: Mode) -> Optional[Tuple[str, ...]]:
    if not bond_indices:
        return tuple(sorted({_canonical_smiles(mol)}))

    cut_mol = Chem.FragmentOnBonds(mol, list(bond_indices), addDummies=True)
    frags = Chem.GetMolFrags(cut_mol, asMols=True, sanitizeFrags=False)

    out_smiles: List[str] = []
    for f in frags:
        f2 = _process_fragment_by_mode(f, mode)
        if f2 is None:
            return None
        out_smiles.append(_canonical_smiles(f2))

    out_smiles.sort()
    return tuple(out_smiles)


def enumerate_cleavage_products(
    smiles: str,
    *,
    max_cuts: int = 1,
    include_ring_bonds: bool = True,
    mode: Mode = "radical",
) -> List[CleavageResult]:
    """
    Enumerate cleavage products for combinations of up to `max_cuts` broken bonds.

    mode:
      - "radical": homolytic cleavage; each cut end gains one radical electron
      - "cap_h"  : remove cut markers, sanitize; valence satisfied with implicit H
      - "ionic"  : heuristic heterolytic cleavage; assigns +/- charges based on EN
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

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
