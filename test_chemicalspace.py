from chemical_space.chemical_space import ChemicalSpace
from reaction_space.reaction_space import ReactionSpace
from os import cpu_count

space = ChemicalSpace(
        atoms=["C", "O", "H"],
        max_atoms=2,
        n_workers=cpu_count(),
        output_dir="chemical_space_results",
    )
space.explore()


