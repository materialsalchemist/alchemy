import click
from os import cpu_count
import logging
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from .reaction_space import ReactionSpace
from .energy_predictor import EnergyPredictor
from .utils import RADICAL_THRESHOLD


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def reaction_cli():
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def common_options(func):
	# Apply options directly to func so they stack with options already on it.
	# (Wrapping with @wraps would clobber func's __click_params__ via __dict__ update.)
	func = click.option(
		"-c",
		"--max-complexity",
		type=int,
		default=3,
		show_default=True,
		help="Maximum number of reactants/products for higher generation reactions.",
	)(func)
	func = click.option(
		"-g",
		"--generations",
		type=int,
		default=2,
		show_default=True,
		help="Number of bimolecular reaction generations to run.",
	)(func)
	func = click.option(
		"-w", "--workers", type=int, default=cpu_count(), show_default=True, help="Number of worker processes."
	)(func)
	func = click.option(
		"-o",
		"--output-dir",
		type=click.Path(file_okay=False),
		default="reaction_space_results",
		show_default=True,
		help="Directory to save all results.",
	)(func)
	func = click.option(
		"-i",
		"--input-csv",
		type=click.Path(exists=True, dir_okay=False),
		default="chemical_space_results/molecules.csv",
		show_default=True,
		help="Input CSV file with molecules and atom counts.",
	)(func)
	return func


@reaction_cli.command()
@common_options
@click.option(
	"-r",
	"--radical-threshold",
	type=int,
	default=RADICAL_THRESHOLD,
	show_default=True,
	help="Maximum allowed radical electrons per reaction side before filtering.",
)
def explore(input_csv, output_dir, workers, generations, max_complexity, radical_threshold):
	"""Run the full workflow: find candidates, verify, and export results."""
	space = ReactionSpace(
		input_csv=input_csv,
		output_dir=output_dir,
		n_workers=workers,
		num_generations=generations,
		max_reaction_complexity=max_complexity,
		radical_threshold=radical_threshold,
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
@click.option(
	"-r",
	"--radical-threshold",
	type=int,
	default=RADICAL_THRESHOLD,
	show_default=True,
	help="Maximum allowed radical electrons per reaction side before filtering.",
)
def verify_reactions(input_csv, output_dir, workers, generations, max_complexity, radical_threshold):
	"""Step 2: Verify candidates from DB using RDKit."""
	# Without this option the staged workflow silently verified at the default
	# threshold, so a run of `find-candidates` then `verify-reactions` could not
	# reproduce a network produced by `explore -r 1`.
	space = ReactionSpace(
		input_csv=input_csv,
		output_dir=output_dir,
		n_workers=workers,
		num_generations=generations,
		max_reaction_complexity=max_complexity,
		radical_threshold=radical_threshold,
	)
	space.verify_reactions()


@reaction_cli.command()
@click.option(
	"-o",
	"--output-dir",
	type=click.Path(file_okay=False),
	default="reaction_space_results",
	show_default=True,
	help="Directory containing the reaction databases.",
)
@click.option("-f", "--filename", type=str, default="reactions.csv", show_default=True, help="Output CSV filename.")
def export_csv(output_dir, filename):
	"""Step 3a: Export verified reactions to a single CSV file."""
	space = ReactionSpace(input_csv="", output_dir=output_dir)  # input_csv not needed for export
	space.export_to_csv(filename=filename)


@reaction_cli.command()
@click.option(
	"-o",
	"--output-dir",
	type=click.Path(file_okay=False),
	default="reaction_space_results",
	show_default=True,
	help="Directory containing the reaction databases.",
)
@click.option(
	"-f",
	"--filename",
	type=str,
	default="reaction_network.json",
	show_default=True,
	help="Output JSON filename for the graph.",
)
def export_graph(output_dir, filename):
	"""Step 3b: Generate and save the reaction network as a JSON file."""
	space = ReactionSpace(input_csv="", output_dir=output_dir)  # input_csv not needed for export
	space.generate_reaction_network_graph(filename=filename)


@reaction_cli.command()
@click.option(
	"-o",
	"--output-dir",
	type=click.Path(file_okay=False),
	default="reaction_space_results",
	show_default=True,
	help="Directory containing the reaction databases.",
)
@click.option(
	"--kuzu-dir",
	type=click.Path(file_okay=False),
	help="Directory to save KuzuDB database. Defaults to <output_dir>/kuzu_db",
)
@click.option(
	"--verified/--no-verified",
	default=True,
	show_default=True,
	help="Whether to export only verified reactions or all candidates.",
)
@click.option(
	"--model-path",
	type=click.Path(exists=True, dir_okay=False),
	default="model.script",
	show_default=True,
	help="Path to the TorchScript model file.",
)
def export_kuzu(output_dir, kuzu_dir, verified, model_path):
	"""Step 3d: Export reactions to KuzuDB graph database."""
	space = ReactionSpace(input_csv="", output_dir=output_dir)
	space.export_to_kuzu(kuzu_dir=kuzu_dir, use_verified=verified, model_path=model_path)


@reaction_cli.command()
@click.option(
	"-o",
	"--output-dir",
	type=click.Path(file_okay=False),
	default="reaction_space_results",
	show_default=True,
	help="Directory containing the reaction databases.",
)
@click.option(
	"--kuzu-dir",
	type=click.Path(file_okay=False),
	help="Directory to save KuzuDB database. Defaults to <output_dir>/kuzu_db",
)
@click.option(
	"--model-path",
	type=click.Path(exists=True, dir_okay=False),
	default="model.script",
	show_default=True,
	help="Path to the TorchScript model file.",
)
def update_kuzu_energies(output_dir, kuzu_dir, model_path):
	"""Step 3e: Update free energy predictions in an existing KuzuDB."""
	space = ReactionSpace(input_csv="", output_dir=output_dir)
	space.update_kuzu_energies(kuzu_dir=kuzu_dir, model_path=model_path)


@reaction_cli.command()
@click.option(
	"--kuzu-dir",
	type=click.Path(file_okay=True, dir_okay=True),
	default="reaction_space_results/kuzu_db",
	show_default=True,
	help="Path to the KuzuDB database.",
)
def calculate_importance(kuzu_dir):
	"""Offline step: Calculate node centrality (PageRank) for the visualizer."""
	from .calculate_importance import calculate_importance as calc

	calc(kuzu_dir=kuzu_dir)


def predict_energy_for_smiles(smiles, predictor):
	"""Predict free energy for a single SMILES string. Returns None if prediction fails."""
	if not smiles or not isinstance(smiles, str):
		return None
	try:
		return predictor.predict_free_energy(smiles)
	except Exception:
		return None


def calculate_reaction_delta_g(row, predictor):
	"""Calculate ΔG for a reaction: ΔG = G(products) - G(reactants)"""
	reactants_smiles = str(row["reactants"]).strip()
	products_smiles = str(row["products"]).strip()
	
	if not reactants_smiles or not products_smiles:
		return None
	
	try:
		# Split multi-molecule SMILES by dots and predict energy for each
		reactant_list = [s.strip() for s in reactants_smiles.split(".") if s.strip()]
		product_list = [s.strip() for s in products_smiles.split(".") if s.strip()]
		
		reactant_energies = [predict_energy_for_smiles(smi, predictor) for smi in reactant_list]
		product_energies = [predict_energy_for_smiles(smi, predictor) for smi in product_list]
		
		# If any prediction failed, return None
		if None in reactant_energies or None in product_energies:
			return None
		
		reactants_energy = sum(reactant_energies)
		products_energy = sum(product_energies)
		
		return products_energy - reactants_energy
	except Exception:
		return None


@reaction_cli.command()
@click.option(
	"-i",
	"--input-csv",
	type=click.Path(exists=True, dir_okay=False),
	required=True,
	help="Input CSV file with reactions (must have 'reactants' and 'products' columns).",
)
@click.option(
	"-o",
	"--output-csv",
	type=click.Path(),
	required=True,
	help="Output CSV file with predicted energies.",
)
@click.option(
	"--model-path",
	type=click.Path(exists=True, dir_okay=False),
	default="model.script",
	show_default=True,
	help="Path to the TorchScript model file for energy prediction.",
)
@click.option(
	"-w", "--workers", type=int, default=cpu_count(), show_default=True, help="Number of worker processes."
)
def predict_reaction_energies(input_csv, output_csv, model_path, workers):
	"""Predict free energy (ΔG) for all reactions in a CSV file."""
	click.secho(f"Loading reactions from {input_csv}...", fg="blue")
	df = pd.read_csv(input_csv)
	
	if "reactants" not in df.columns or "products" not in df.columns:
		click.secho("Error: CSV must have 'reactants' and 'products' columns.", fg="red")
		raise ValueError("Missing required columns: 'reactants' and/or 'products'")
	
	click.secho(f"Loaded {len(df)} reactions.", fg="green")
	
	# Initialize energy predictor
	predictor = EnergyPredictor(model_path=model_path)
	if predictor.model is None:
		click.secho(f"Warning: Could not load model from {model_path}. Using dummy predictions.", fg="yellow")
	
	click.secho("Predicting reaction energies...", fg="blue")
	
	# Predict delta G for each reaction
	delta_g_values = []
	for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting energies"):
		delta_g = calculate_reaction_delta_g(row, predictor)
		delta_g_values.append(delta_g)
	
	df["predicted_delta_g"] = delta_g_values
	
	# Save to output CSV
	df.to_csv(output_csv, index=False)
	click.secho(f"Results saved to {output_csv}", fg="green")
	
	# Print summary statistics
	valid_predictions = [x for x in delta_g_values if x is not None]
	if valid_predictions:
		click.secho(f"\nSummary Statistics:", fg="cyan")
		click.echo(f"  Total reactions: {len(df)}")
		click.echo(f"  Valid predictions: {len(valid_predictions)}")
		click.echo(f"  Failed predictions: {len(df) - len(valid_predictions)}")
		click.echo(f"  Mean ΔG: {sum(valid_predictions) / len(valid_predictions):.4f}")
		click.echo(f"  Min ΔG: {min(valid_predictions):.4f}")
		click.echo(f"  Max ΔG: {max(valid_predictions):.4f}")
	else:
		click.secho("No valid predictions could be made.", fg="yellow")
