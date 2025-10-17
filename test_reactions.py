from chemical_space.chemical_space import ChemicalSpace
from reaction_space.reaction_space import ReactionSpace
from os import cpu_count


space = ReactionSpace(
		input_csv='chemical_space_results/molecules.csv', 
		custom_reactants_csv="test_reactants.csv",
		output_dir="test_CCCC", 
		n_workers=cpu_count(),
		num_generations=2,
		max_reaction_complexity=4,
		g05_method='recombination',
		require_custom_reactant=False
	)
space.find_reaction_candidates()
space.verify_reactions()
space.export_to_csv(filename='reactions.csv')
