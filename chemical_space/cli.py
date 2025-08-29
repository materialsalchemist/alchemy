from functools import wraps
import click
from os import cpu_count
import logging
from .chemical_space import ChemicalSpace


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def chemical_cli():
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
	DisableLog("rdApp.*")

def common_options(f):
	@wraps(f)
	@click.option("-n", "--max-atoms", type=int, default=3, show_default=True, help="Maximum number of heavy atoms.")
	@click.option("-a", "--atoms", multiple=True, default=['C', 'O', 'H'], show_default=True, help="Heavy atoms to include.")
	@click.option("-w", "--workers", type=int, default=cpu_count(), show_default=True, help="Number of worker processes.")
	@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="chemical_space_results", show_default=True, help="Directory to save all results.")
	def wrapper(*args, **kwargs):
		return f(*args, **kwargs)

	return wrapper

@chemical_cli.command()
@common_options
def explore(max_atoms, atoms, workers, output_dir):
	"""Run the full workflow: generate, build, and export."""
	space = ChemicalSpace(
		atoms=frozenset(atoms),
		max_atoms=max_atoms,
		n_workers=workers,
		output_dir=output_dir
	)
	space.explore()

@chemical_cli.command()
@common_options
def generate_compositions(max_atoms, atoms, workers, output_dir):
	"""Stage 1: Generate and save plausible compositions."""
	space = ChemicalSpace(
		atoms=frozenset(atoms),
		max_atoms=max_atoms,
		n_workers=workers,
		output_dir=output_dir
	)
	space.generate_compositions()

@chemical_cli.command()
@common_options
def build_molecules(max_atoms, atoms, workers, output_dir):
	"""Stage 2: Build molecules from existing compositions."""
	space = ChemicalSpace(
		atoms=frozenset(atoms),
		max_atoms=max_atoms,
		n_workers=workers,
		output_dir=output_dir
	)
	space.build_molecules()

@chemical_cli.command()
@click.option("-n", "--max-atoms", type=int, default=3, show_default=True, help="Maximum number of heavy atoms to check for DBs.")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="chemical_space_results", show_default=True, help="Directory containing results.")
@click.option("-f", "--filename", type=str, default="molecules.csv", show_default=True, help="Output CSV filename.")
def export_csv(max_atoms, output_dir, filename):
	"""Stage 3a: Export all found molecules to a single CSV file."""
	space = ChemicalSpace(max_atoms=max_atoms, output_dir=output_dir)
	space.export_to_csv(filename=filename)

@chemical_cli.command()
@click.option("-n", "--max-atoms", type=int, default=3, show_default=True, help="Maximum number of heavy atoms in molecules to export images for.")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="chemical_space_results", show_default=True, help="Directory containing LMDB results and to save images.")
def export_images(max_atoms, output_dir):
	"""Stage 3b: Export molecules from LMDB to images (RDKit PNG and OpenBabel PNG)."""
	space = ChemicalSpace(max_atoms=max_atoms, output_dir=output_dir)
	space.export_images()

@chemical_cli.command()
@click.option("-n", "--max-atoms", type=int, default=3, show_default=True, help="Maximum number of heavy atoms in molecules to export XYZ files for.")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default="chemical_space_results", show_default=True, help="Directory containing LMDB results and to save XYZ files.")
def export_xyz(max_atoms, output_dir):
	"""Stage 3c: Export molecules from LMDB to XYZ coordinate files using OpenBabel."""
	space = ChemicalSpace(max_atoms=max_atoms, output_dir=output_dir)
	space.export_xyz()
