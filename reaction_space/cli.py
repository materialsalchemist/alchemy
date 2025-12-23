import click
from functools import wraps
from os import cpu_count
import logging
from .reaction_space import ReactionSpace

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def reaction_cli():
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def common_options(func):
	@wraps(func)
	@click.option(
		"-i", "--input-csv", type=click.Path(exists=True, dir_okay=False),
		 default="chemical_space_results/molecules.csv", show_default=True,
		 help="Input CSV file with molecules and atom counts.",
	 )
	@click.option(
		"-o", "--output-dir", type=click.Path(file_okay=False),
		default="reaction_space_results", show_default=True,
		help="Directory to save all results."
	)
	@click.option(
		"-w", "--workers", type=int, default=cpu_count(), show_default=True,
		help="Number of worker processes."
	)
	@click.option(
		"-g", "--generations", type=int, default=2, show_default=True,
		help="Number of bimolecular reaction generations to run."
	)
	@click.option(
		"-c", "--max-complexity", type=int, default=3, show_default=True,
		help="Maximum number of reactants/products for higher generation reactions."
	)

	def wrapper(*args, **kwargs):
		return func(*args, **kwargs)

	return wrapper

@reaction_cli.command()
@common_options
def explore(input_csv, output_dir, workers, generations, max_complexity):
	"""Run the full workflow: find candidates, verify, and export results."""
	space = ReactionSpace(
		input_csv=input_csv, 
		output_dir=output_dir, 
		n_workers=workers,
		num_generations=generations,
		max_reaction_complexity=max_complexity,
	)
	space.explore()

@reaction_cli.command()
@common_options
def find_candidates(input_csv, output_dir, workers, generations, max_complexity):
	"""Step 1: Generate reaction candidates from initial molecules."""
	space = ReactionSpace(
		input_csv=input_csv, 
		output_dir=output_dir, 
		n_workers=workers,
		num_generations=generations,
		max_reaction_complexity=max_complexity,
	)
	space.find_reaction_candidates()

@reaction_cli.command()
@common_options
def verify_reactions(input_csv, output_dir, workers, generations, max_complexity):
	"""Step 2: Verify candidates from DB using RDKit."""
	space = ReactionSpace(
		input_csv=input_csv, 
		output_dir=output_dir, 
		n_workers=workers,
		num_generations=generations,
		max_reaction_complexity=max_complexity,
		
	)
	space.verify_reactions()

@reaction_cli.command()
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="reaction_space_results", show_default=True, help="Directory containing the reaction databases.")
@click.option("-f", "--filename", type=str, default="reactions.csv", show_default=True, help="Output CSV filename.")
def export_csv(output_dir, filename):
	"""Step 3a: Export verified reactions to a single CSV file."""
	space = ReactionSpace(input_csv="", output_dir=output_dir) # input_csv not needed for export
	space.export_to_csv(filename=filename)

@reaction_cli.command()
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="reaction_space_results", show_default=True, help="Directory containing the reaction databases.")
@click.option("-f", "--filename", type=str, default="reaction_network.json", show_default=True, help="Output JSON filename for the graph.")
def export_graph(output_dir, filename):
	"""Step 3b: Generate and save the reaction network as a JSON file."""
	space = ReactionSpace(input_csv="", output_dir=output_dir) # input_csv not needed for export
	space.generate_reaction_network_graph(filename=filename)

@reaction_cli.command()
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="reaction_space_results", show_default=True, help="Directory containing the reaction databases and where to save images.")
def export_images(output_dir):
	"""Step 3c: Export verified reactions from database to PNG images."""
	space = ReactionSpace(input_csv="", output_dir=output_dir) # input_csv not needed for export
	space.export_images()
