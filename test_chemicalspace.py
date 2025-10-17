from chemical_space.chemical_space import ChemicalSpace
from reaction_space.reaction_space import ReactionSpace
from os import cpu_count

space = ChemicalSpace(
        atoms=["C", "H", "N"],
        max_atoms=4,
        n_workers=cpu_count(),
        output_dir="test_chemical_space",
    )
space.explore()


