import os
import pandas as pd
import json
import lmdb
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
)
from werkzeug.routing import BaseConverter

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(APP_ROOT, "..")

CHEMICAL_SPACE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "chemical_space_results")
REACTION_SPACE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "reaction_space_results")

MOLECULES_CSV_PATH = os.path.join(CHEMICAL_SPACE_RESULTS_PATH, "molecules.csv")
RDKIT_IMAGES_PATH = os.path.join(CHEMICAL_SPACE_RESULTS_PATH, "molecule_images")
OBABEL_IMAGES_PATH = os.path.join(CHEMICAL_SPACE_RESULTS_PATH, "molecule_images_obabel")
REACTION_NETWORK_JSON_PATH = os.path.join(
    REACTION_SPACE_RESULTS_PATH, "reaction_network.json"
)
REACTION_IMAGES_PATH = os.path.join(REACTION_SPACE_RESULTS_PATH, "reaction_images")


app = Flask(__name__)
app.config["CHEMICAL_SPACE_RESULTS_PATH"] = CHEMICAL_SPACE_RESULTS_PATH
app.config["REACTION_SPACE_RESULTS_PATH"] = REACTION_SPACE_RESULTS_PATH

molecule_data_cache = {
    "df": None,
    "smiles_to_image_idx": None,
    "atom_columns": None,
    "all_smiles_sorted": None,
    "heavy_atom_columns": None,
}


def get_heavy_atom_columns(df_columns):
    """Identifies heavy atom columns (excluding SMILES and H if present)."""
    excluded = ["SMILES", "H"]
    return sorted([col for col in df_columns if col not in excluded and col.isalpha()])


def load_molecule_data():
    """Loads and processes molecule data from CSV."""
    if not os.path.exists(MOLECULES_CSV_PATH):
        app.logger.error(f"Molecules CSV not found at {MOLECULES_CSV_PATH}")
        return None, None, None, None, None

    try:
        df = pd.read_csv(MOLECULES_CSV_PATH)
        df = df.fillna(0)

        atom_columns = sorted([col for col in df.columns if col != "SMILES"])

        for col in atom_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(int)
                except ValueError:
                    app.logger.warning(
                        f"Could not convert column {col} to int. Skipping."
                    )

        all_smiles_sorted = sorted(list(df["SMILES"].unique()))
        smiles_to_image_idx = {smi: idx for idx, smi in enumerate(all_smiles_sorted)}

        heavy_atom_columns = get_heavy_atom_columns(df.columns)

        if heavy_atom_columns:
            df["num_heavy_atoms"] = df[heavy_atom_columns].sum(axis=1)
        else:
            df["num_heavy_atoms"] = 0

        molecule_data_cache["df"] = df
        molecule_data_cache["smiles_to_image_idx"] = smiles_to_image_idx
        molecule_data_cache["atom_columns"] = atom_columns
        molecule_data_cache["all_smiles_sorted"] = all_smiles_sorted
        molecule_data_cache["heavy_atom_columns"] = heavy_atom_columns

        app.logger.info(
            f"Loaded {len(df)} molecules. Heavy atoms: {heavy_atom_columns}"
        )
        return (
            df,
            smiles_to_image_idx,
            atom_columns,
            all_smiles_sorted,
            heavy_atom_columns,
        )
    except Exception as e:
        app.logger.error(f"Error loading molecule data: {e}")
        return None, None, None, None, None


with app.app_context():
    load_molecule_data()


class ListConverter(BaseConverter):
    def to_python(self, value):
        return value.split(",")

    def to_url(self, values):
        return ",".join(BaseConverter.to_url(value) for value in values)


app.url_map.converters["list"] = ListConverter


@app.route("/")
def index():
    return redirect(url_for("molecules_page"))


@app.route("/molecules")
def molecules_page():
    df = molecule_data_cache["df"]
    smiles_to_image_idx = molecule_data_cache["smiles_to_image_idx"]
    atom_columns = molecule_data_cache["atom_columns"]
    heavy_atom_columns = molecule_data_cache["heavy_atom_columns"]

    if df is None:
        return render_template(
            "molecules.html",
            error="Molecule data could not be loaded.",
            molecules=[],
            smiles_to_image_idx={},
            atom_columns=[],
            heavy_atom_columns=[],
            filters={},
        )

    filtered_df = df.copy()
    query_filters = {}

    search_smiles = request.args.get("search_smiles", "").strip()
    query_filters["search_smiles"] = search_smiles
    if search_smiles:
        filtered_df = filtered_df[
            filtered_df["SMILES"].str.contains(search_smiles, case=False, na=False)
        ]

    current_atom_filters = {}
    for atom in heavy_atom_columns:
        param_name = f"has_{atom}"
        if request.args.get(param_name):
            filtered_df = filtered_df[filtered_df[atom] > 0]
            current_atom_filters[param_name] = True
    query_filters["atom_filters"] = current_atom_filters

    exclusive_filter = request.args.get("exclusive_composition", "")
    query_filters["exclusive_composition"] = exclusive_filter
    if exclusive_filter:
        selected_atoms = exclusive_filter.split("_")  # e.g., "C_H" -> ["C", "H"]

        for atom_to_check in selected_atoms:
            if atom_to_check in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[atom_to_check] > 0]
            else:
                filtered_df = pd.DataFrame(columns=filtered_df.columns)
                break

        other_heavy_atoms = [a for a in heavy_atom_columns if a not in selected_atoms]
        for other_atom in other_heavy_atoms:
            if other_atom in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[other_atom] == 0]

    min_heavy_atoms = request.args.get("min_heavy_atoms", type=int)
    max_heavy_atoms = request.args.get("max_heavy_atoms", type=int)
    query_filters["min_heavy_atoms"] = min_heavy_atoms
    query_filters["max_heavy_atoms"] = max_heavy_atoms

    if "num_heavy_atoms" in filtered_df.columns:
        if min_heavy_atoms is not None:
            filtered_df = filtered_df[filtered_df["num_heavy_atoms"] >= min_heavy_atoms]
        if max_heavy_atoms is not None:
            filtered_df = filtered_df[filtered_df["num_heavy_atoms"] <= max_heavy_atoms]

    molecules_list = filtered_df.to_dict(orient="records")

    return render_template(
        "molecules.html",
        molecules=molecules_list,
        smiles_to_image_idx=smiles_to_image_idx,
        atom_columns=atom_columns,
        heavy_atom_columns=heavy_atom_columns,
        filters=query_filters,
    )


@app.route("/reactions")
def reactions_page():
    reactions_lmdb_path = os.path.join(REACTION_SPACE_RESULTS_PATH, "reactions.lmdb")
    reactions_csv_path = os.path.join(REACTION_SPACE_RESULTS_PATH, "reactions.csv")

    reactions = []
    if os.path.exists(reactions_lmdb_path):
        app.logger.info(f"Loading reactions from LMDB: {reactions_lmdb_path}")
        env = lmdb.open(reactions_lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            for i, (key, value) in enumerate(txn.cursor()):
                try:
                    value_data = json.loads(value.decode("utf-8"))
                    reaction_smarts = value_data["smi"]
                    reactants, products = reaction_smarts.split(">>")
                    
                    reaction_item = {
                        "reactants": reactants,
                        "products": products,
                        "image_path": f"reaction_images/reaction_{i}.png",
                    }
                    reactions.append(reaction_item)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    app.logger.warning(f"Skipping malformed reaction in LMDB: {e}")
                    continue
        env.close()
    elif os.path.exists(reactions_csv_path):
        app.logger.info(f"Loading reactions from CSV: {reactions_csv_path}")
        df_react = pd.read_csv(reactions_csv_path)
        reactions = df_react.to_dict('records')
        for i, reaction in enumerate(reactions):
            reaction['image_path'] = f"reaction_images/reaction_{i}.png"
    else:
        app.logger.warning("No reaction file (LMDB or CSV) found.")

    return render_template("reactions.html", reactions=reactions)


@app.route("/reaction_network_data")
def reaction_network_data():
    if not os.path.exists(REACTION_NETWORK_JSON_PATH):
        return jsonify({"error": "Reaction network data not found."}), 404

    return send_from_directory(
        os.path.dirname(REACTION_NETWORK_JSON_PATH),
        os.path.basename(REACTION_NETWORK_JSON_PATH),
    )


@app.route("/molecule_image/<image_type>/<filename>")
def molecule_image(image_type, filename):
    if image_type == "rdkit":
        directory = RDKIT_IMAGES_PATH
    elif image_type == "obabel":
        directory = OBABEL_IMAGES_PATH
    else:
        return "Invalid image type", 404

    if not os.path.exists(directory):
        app.logger.error(f"Image directory not found: {directory}")
        return "Image directory not found", 404

    return send_from_directory(directory, filename)


@app.route("/reaction_image/<filename>")
def reaction_image(filename):
    if not os.path.exists(REACTION_IMAGES_PATH):
        app.logger.error(f"Image directory not found: {REACTION_IMAGES_PATH}")
        return "Image directory not found", 404

    return send_from_directory(REACTION_IMAGES_PATH, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
