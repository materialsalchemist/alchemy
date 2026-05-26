# Alchemy: Chemical Reaction Network Explorer

A tool for exploring chemical space and generating chemical reaction networks.

## Features

- **Chemical Space Exploration**: Generate all plausible molecules up to a certain number of heavy atoms from a given set of elements, or discover fragments (cleavage products) from specific parent molecules.
- **Reaction Network Generation**: Discover chemical reactions from a set of molecules and build a reaction network using hierarchical generation (G0, G1, G2+).
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

The `chemical` subcommand is used to generate a set of molecules or fragments.

#### Full Workflow (`explore`)

The easiest way to get started is to run the full workflow.

**Generate from elements:**
```bash
alchemy chemical explore --max-atoms 3 -a C -a O -a H --output-dir chemical_space_results
```

**Generate fragments from a parent molecule:**
```bash
alchemy chemical explore --smiles "CCO" --max-atoms 3 --output-dir chemical_space_results
```

This command will:
1. Generate all plausible molecular compositions up to the atom limit.
2. Build unique molecule structures from these compositions.
3. If `--smiles` is provided, generate cleavage products (fragments) from the parent molecule.
4. Export the resulting molecules to `chemical_space_results/molecules.csv`.

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
1. Find reaction candidates (dissociations, transfers, rearrangements) from the molecules in `molecules.csv`.
2. Verify the chemical plausibility of these reactions.
3. Export verified reactions to `reaction_space_results/reactions.csv`.
4. Export the reaction network graph to `reaction_space_results/reaction_network.json`.

#### Individual Stages

- **Stage 1: Find Reaction Candidates**
  ```bash
  alchemy reaction find-candidates --generations 2 --max-complexity 3
  ```

- **Stage 2: Verify Reactions**
  ```bash
  alchemy reaction verify-reactions
  ```

- **Stage 3: Export Data**
  ```bash
  # Export reactions to CSV
  alchemy reaction export-csv

  # Export reaction network graph (NetworkX JSON)
  alchemy reaction export-graph

  # Export to KuzuDB Graph Database (Required for Web Visualization)
  alchemy reaction export-kuzu --output-dir reaction_space_results

  # Calculate node importance (PageRank) for visualization
  alchemy reaction calculate-importance --kuzu-dir reaction_space_results/kuzu_db
  ```

For more options, use the `--help` flag: `alchemy reaction --help`.

### Web Visualization

This project includes a high-performance **FastAPI** web application for visualizing the chemical reaction network using [Sigma.js](https://www.sigmajs.org/) and [KuzuDB](https://kuzudb.com/).

**1. Prepare the database:**
Ensure you have exported your reaction network to KuzuDB and calculated node importance:
```bash
alchemy reaction export-kuzu
alchemy reaction calculate-importance
```

**2. Run the web app:**
```bash
uvicorn app.main:app --reload
```
Then open your web browser to `http://127.0.0.1:8000`.

## Development

If you want to contribute to the project, here’s how to set up a local development environment using `uv`.

**Option 1**: Development Setup with `uv`

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

**Option 2:** Development Setup with `pip`

- **1: Clone the repository**
  ```bash
  git clone https://github.com/materialsalchemist/alchemy
  cd alchemy
  ```

- **2: Create and activate a virtual environment:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
