
from reaction_space.workers import (
	get_dissociation_fragments,
	worker_systematic_recombination,
	worker_radical_addition,
	worker_generate_new_reactions_g1,
	worker_generate_higher_gen_reactions,
	worker_verify_reaction_batch,
)
print(get_dissociation_fragments("CCCC"))