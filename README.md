# Alchemy: Chemical Reaction Network Explorer

A tool for exploring chemical space and generating chemical reaction networks.

## Features

- **Chemical Space Exploration**: Generate all plausible molecules up to a certain number of heavy atoms from a given set of elements.
- **Reaction Network Generation**: Discover chemical reactions from a set of molecules and build a reaction network.
- **Data Export**: Export molecules, reactions, and the reaction network graph in various formats (CSV, PNG, XYZ, JSON).

## Installation

### Prerequisites

- Python 3.11+
- [Open Babel](http://openbabel.org/wiki/Main_Page): Required for generating molecule images and XYZ coordinate files. Please follow the installation instructions for your operating system.

You can check if `obabel` is installed and in your PATH by running:
```bash
obabel -V
```

### Installation with pip

You can install the project and its dependencies using pip from the root of this repository:

```bash
pip install .
```

### Installation with uv

[`uv`](https://docs.astral.sh/uv/) is an extremely fast Python package installer and resolver, written in Rust. It can be used as a drop-in replacement for `pip`.

```bash
uv pip install .
```

This will install the `alchemy` command-line tool.

## Usage

The main command is `alchemy`, which has two subcommands: `chemical` and `reaction`.

### 1. Chemical Space Exploration

The `chemical` subcommand is used to generate a set of molecules.

#### Full Workflow (`explore`)

The easiest way to get started is to run the full workflow, which generates molecules and exports them.

```bash
alchemy chemical explore --max-atoms 3 -a C -a O -a H --output-dir chemical_space_results
```

This command will:
1. Generate all plausible molecular compositions up to 3 heavy atoms using C, O, and H.
2. Build unique molecule structures from these compositions.
3. Export the resulting molecules to `chemical_space_results/molecules.csv`.

#### Individual Stages

You can also run each stage of the chemical space exploration separately:

- **Stage 1: Generate Compositions**
  ```bash
  alchemy chemical generate-compositions --max-atoms 3
  ```

- **Stage 2: Build Molecules**
  ```bash
  alchemy chemical build-molecules --max-atoms 3
  ```

- **Stage 3: Export Data**
  ```bash
  # Export to CSV
  alchemy chemical export-csv --max-atoms 3

  # Export molecule images (requires Open Babel)
  alchemy chemical export-images --max-atoms 3

  # Export XYZ files (requires Open Babel)
  alchemy chemical export-xyz --max-atoms 3
  ```

For more options, use the `--help` flag: `alchemy chemical --help`.

### 2. Reaction Space Exploration

The `reaction` subcommand takes a list of molecules (e.g., from the `chemical explore` step) and generates a reaction network.

#### Full Workflow (`explore`)

```bash
alchemy reaction explore --input-csv chemical_space_results/molecules.csv --output-dir reaction_space_results
```
This command will:
1. Find reaction candidates from the molecules in `molecules.csv`.
2. Verify the chemical plausibility of these reactions using `rxnmapper`.
3. Export verified reactions to `reaction_space_results/reactions.csv`.
4. Export the reaction network graph to `reaction_space_results/reaction_network.json`.

#### Individual Stages

- **Stage 1: Find Reaction Candidates**
  ```bash
  alchemy reaction find-candidates --generations 2
  ```

- **Stage 2: Verify Reactions**
  ```bash
  alchemy reaction verify-reactions
  ```

- **Stage 3: Export Data**
  ```bash
  # Export reactions to CSV
  alchemy reaction export-csv

  # Export reaction network graph
  alchemy reaction export-graph
  ```

For more options, use the `--help` flag: `alchemy reaction --help`.

### Web Visualization

This project also includes a Flask web application for visualizing the generated molecules and reaction network.

To run the web app:
```bash
flask --app flask_app/app.py run
```
Then open your web browser to `http://127.0.0.1:5000`.

_Note: Make sure the `chemical_space_results` and `reaction_space_results` directories are in the project root for the web app to find the data._

## Development

If you want to contribute to the project, here’s how to set up a local development environment using `uv`.

Development Setup with `uv`

- **1: Clone the repository**
  ```bash
  git clone https://github.com/materialsalchemist/alchemy
  cd alchemy
  ```

- **2: Create and activate a virtual environment:**
  ```bash
  uv sync
  source .venv/bin/activate
  ```
