import io
from ase import Atoms
from ase.io import jsonio

def generate_optimized_structure(smiles: str) -> "Atoms":
    """
    Generates a 3D structure from a SMILES string, optimizes it with GFN2-xTB
    via the tblite library, and returns an ASE Atoms object.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ase import Atoms
    from ase.optimize import BFGS
    from tblite.ase import TBLite

    # 1. Generate an initial 3D conformer with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not parse SMILES string: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 1
    AllChem.EmbedMolecule(mol, params)

    # Quick UFF pre-optimization
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass

    # 2. Convert to ASE Atoms
    positions = mol.GetConformer().GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atoms = Atoms(numbers=atomic_numbers, positions=positions)

    # 3. Optimize with tblite (GFN2-xTB)
    calculator = TBLite(method="GFN2-xTB", verbosity=0)
    atoms.calc = calculator

    optimizer = BFGS(atoms, logfile=None)
    optimizer.run(fmax=0.01)
    
    return atoms

def atoms_to_json(atoms: Atoms) -> str:
    """Serializes an ASE Atoms object to a JSON string."""
    string_io = io.StringIO()
    jsonio.write_json(string_io, [atoms])
    return string_io.getvalue()
