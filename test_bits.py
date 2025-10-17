from reaction_space.workers import *
from rdkit import Chem, RDLogger

print(get_dissociation_fragments("[H][H]"))
print(Chem.MolFromSmiles("[H]").GetNumAtoms())
print(combine_fragments(Chem.MolFromSmiles("[H]"), Chem.MolFromSmiles("[H]")))
print(canonicalize_smiles("[H]"))
mol = Chem.MolFromSmiles("[CH3]Cl")
bonds = mol.GetBonds()
print('bonds:',bonds,mol.GetNumBonds())
emol = Chem.EditableMol(mol)
		
begin_atom_idx = bonds[0].GetBeginAtomIdx()
end_atom_idx = bonds[0].GetEndAtomIdx()

emol.RemoveBond(begin_atom_idx, end_atom_idx)

fragmented_mol = emol.GetMol()

num_frags = len(Chem.GetMolFrags(fragmented_mol))
frag_mols = Chem.GetMolFrags(fragmented_mol, asMols=True)
print(frag_mols)


def find_ch_bonds(smiles):
    """
    Finds and prints information about C-H bonds in a molecule.

    Args:
        smiles (str): The SMILES string of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add explicit hydrogens
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return

    print(f"Analyzing molecule: {smiles}")
    ch_bond_count = 0
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        atom1_symbol = atom1.GetSymbol()
        atom2_symbol = atom2.GetSymbol()

        # Check if one atom is Carbon and the other is Hydrogen
        if (atom1_symbol == 'C' and atom2_symbol == 'H') or \
           (atom1_symbol == 'H' and atom2_symbol == 'C'):
            ch_bond_count += 1
            print(f"Found C-H bond between Atom {atom1.GetIdx()} ({atom1_symbol}) "
                  f"and Atom {atom2.GetIdx()} ({atom2_symbol})")
    
    print(f"Total C-H bonds found: {ch_bond_count}")
    
find_ch_bonds("[CH3]")