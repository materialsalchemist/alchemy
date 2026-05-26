import click
import sys
import os
from multiprocessing import set_start_method, freeze_support

# Ensure the project root is in the path for top-level modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chemical_space.cli import chemical_cli
from reaction_space.cli import reaction_cli


@click.group()
def cli():
	"""Main CLI for alchemy."""
	pass


cli.add_command(chemical_cli, name="chemical")
cli.add_command(reaction_cli, name="reaction")


def main():
	cli()


if __name__ == "__main__":
	set_start_method("spawn")
	freeze_support()
	main()
