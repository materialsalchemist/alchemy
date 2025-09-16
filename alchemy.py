import click
from multiprocessing import set_start_method, freeze_support
from chemical_space.cli import chemical_cli
from reaction_space.cli import reaction_cli

@click.group()
def cli():
	"""Main CLI for alchemy."""
	pass

cli.add_command(chemical_cli, name="chemical")
cli.add_command(reaction_cli, name="reaction")

def main():
	set_start_method("spawn")
	cli()

if __name__ == "__main__":
	freeze_support()
	main()
