"""
reaction_space/kuzu_exporter.py

Export a ReactionSpace LMDB database into a KùzuDB property graph.

Schema
------
  Node tables:
    Molecule(id STRING PRIMARY KEY, smiles STRING, formula STRING,
             morgan_fp STRING, mw DOUBLE, num_rings INT64,
             num_rot_bonds INT64, tpsa DOUBLE, logp DOUBLE,
             is_radical BOOLEAN)

    Reaction(id STRING PRIMARY KEY, smiles STRING, gen STRING,
             num_reactants INT64, num_products INT64)

  Rel tables:
    REACTANT_OF(from Molecule, to Reaction)
    PRODUCT_OF(from Reaction, to Molecule)
    DERIVED_FROM(from Reaction, to Reaction)   -- G1 from G0 pair, etc.

Usage
-----
    from reaction_space.kuzu_exporter import export_to_kuzu

    export_to_kuzu(
        reaction_space=my_rs,           # ReactionSpace instance
        kuzu_dir="reaction_space_results/kuzu_db",
    )

Query examples (using kuzu Python client)
-----------------------------------------
    import kuzu

    db  = kuzu.Database("reaction_space_results/kuzu_db")
    con = db.connect()

    # All G1 reactions that involve oxygen
    con.execute(
        "MATCH (m:Molecule)-[:REACTANT_OF]->(r:Reaction) "
        "WHERE r.gen = 'G1' AND m.formula CONTAINS 'O' "
        "RETURN r.smiles, m.smiles LIMIT 20"
    ).get_as_df()

    # Two-hop paths from methane
    con.execute(
        "MATCH p = (m:Molecule)-[:REACTANT_OF]->(:Reaction)-[:PRODUCT_OF]->(m2:Molecule) "
        "WHERE m.smiles = 'C' RETURN m2.smiles"
    ).get_as_df()
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tqdm.auto import tqdm

from reaction_space.energy_predictor import EnergyPredictor

import kuzu

if TYPE_CHECKING:
	from .reaction_space import ReactionSpace

logger = logging.getLogger(__name__)


# ─── Molecular feature extraction ────────────────────────────────────────────


def _mol_features(smiles: str) -> Dict:
	"""Compute a small set of molecular descriptors for a SMILES string."""
	default = {
		"formula": "",
		"morgan_fp": "",
		"mw": 0.0,
		"num_rings": 0,
		"num_rot_bonds": 0,
		"tpsa": 0.0,
		"logp": 0.0,
		"is_radical": False,
	}

	if not smiles:
		return default

	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return default

	# Molecular formula
	mol_with_H = Chem.AddHs(mol)
	counts: Counter = Counter()
	for atom in mol_with_H.GetAtoms():
		counts[atom.GetSymbol()] += 1
	# Hill order: C first, H second, rest alphabetical
	formula_parts = []
	for sym in ["C", "H"] + sorted(s for s in counts if s not in ("C", "H")):
		if sym in counts:
			formula_parts.append(f"{sym}{counts[sym] if counts[sym] > 1 else ''}")
	formula = "".join(formula_parts)

	# Morgan fingerprint (radius 2, 512 bits) as hex string
	fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
	fp_hex = fp.ToBitString()

	# Check for radical electrons
	is_radical = any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms())

	return {
		"formula": formula,
		"morgan_fp": fp_hex,
		"mw": round(Descriptors.ExactMolWt(mol_with_H), 4),
		"num_rings": rdMolDescriptors.CalcNumRings(mol),
		"num_rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
		"tpsa": round(Descriptors.TPSA(mol), 4),
		"logp": round(Descriptors.MolLogP(mol), 4),
		"is_radical": is_radical,
	}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _canonicalize(smiles: str) -> str:
	"""Canonicalize a SMILES string using RDKit."""
	try:
		mol = Chem.MolFromSmiles(smiles)
		if mol:
			# We use isomericSmiles=True for high-fidelity canonicalization
			return Chem.MolToSmiles(mol, isomericSmiles=True)
	except Exception:
		pass
	return smiles


def _iter_lmdb(db_path: str):
	"""Yield (smiles, gen) tuples from an LMDB reaction database."""
	try:
		import lmdb
	except ImportError:
		raise ImportError("lmdb is required: pip install lmdb")

	if not os.path.exists(db_path):
		return

	env = lmdb.open(db_path, readonly=True, lock=False)
	with env.begin() as txn:
		for _, value in txn.cursor():
			try:
				data = json.loads(value.decode("utf-8"))
				yield data["smi"], data.get("gen", "G0")
			except Exception:
				continue
	env.close()


def _parse_reaction(smiles: str) -> Tuple[List[str], List[str]]:
	"""Return (reactant_smiles_list, product_smiles_list)."""
	try:
		left, right = smiles.split(">>")
		reactants = [s for s in left.split(".") if s]
		products = [s for s in right.split(".") if s]
		return reactants, products
	except ValueError:
		return [], []


# ─── Main exporter ────────────────────────────────────────────────────────────


def export_to_kuzu(
	reaction_space: "ReactionSpace",
	kuzu_dir: str = "kuzu_db",
	include_fingerprints: bool = True,
	batch_size: int = 2048,
	skip_if_exists: bool = True,
	use_verified: bool = True,
	model_path: str = "model.script",
) -> None:
	"""
	Export reactions from *reaction_space* into a KùzuDB graph.

	Parameters
	----------
	reaction_space:
	    An initialised ``ReactionSpace`` instance.
	kuzu_dir:
	    Directory where the KùzuDB will be created.
	include_fingerprints:
	    Compute and store Morgan fingerprints on Molecule nodes.
	batch_size:
	    Number of rows to insert per transaction.
	skip_if_exists:
	    If True and *kuzu_dir* already contains a database, skip export.
	use_verified:
	    If True, export from the verified database. Otherwise, export from candidates.
	model_path:
	    Path to the TorchScript model file for free energy prediction.
	"""

	if kuzu is None:
		raise ImportError("kuzu is required: pip install kuzu\nSee https://kuzudb.com for installation instructions.")

	if skip_if_exists and os.path.exists(kuzu_dir):
		logger.info("KùzuDB already exists at %s — skipping export.", kuzu_dir)
		print(f"[kuzu] Database already exists at {kuzu_dir}. Skipping.")
		return

	# If we are here, we are doing a fresh export.
	# Delete the old directory to avoid "duplicated primary key" errors during COPY.
	if os.path.exists(kuzu_dir):
		print(f"[kuzu] Removing existing database at {kuzu_dir} for fresh export…")
		if os.path.isdir(kuzu_dir):
			shutil.rmtree(kuzu_dir)
		else:
			os.remove(kuzu_dir)

	parent_dir = os.path.dirname(kuzu_dir)
	if parent_dir:
		os.makedirs(parent_dir, exist_ok=True)

	db = kuzu.Database(kuzu_dir)
	con = kuzu.Connection(db)

	# ── Schema ────────────────────────────────────────────────────────────────
	print("[kuzu] Creating schema…")

	con.execute(
		"CREATE NODE TABLE IF NOT EXISTS Molecule("
		"  id STRING,"
		"  smiles STRING,"
		"  formula STRING,"
		"  morgan_fp STRING,"
		"  mw DOUBLE,"
		"  num_rings INT64,"
		"  num_rot_bonds INT64,"
		"  tpsa DOUBLE,"
		"  logp DOUBLE,"
		"  is_radical BOOLEAN,"
		"  predicted_free_energy DOUBLE,"
		"  PRIMARY KEY (id)"
		")"
	)

	con.execute(
		"CREATE NODE TABLE IF NOT EXISTS Reaction("
		"  id STRING,"
		"  smiles STRING,"
		"  gen STRING,"
		"  num_reactants INT64,"
		"  num_products INT64,"
		"  PRIMARY KEY (id)"
		")"
	)

	con.execute("CREATE REL TABLE IF NOT EXISTS REACTANT_OF(  FROM Molecule TO Reaction)")

	con.execute("CREATE REL TABLE IF NOT EXISTS PRODUCT_OF(  FROM Reaction TO Molecule)")

	# ── Collect data ─────────────────────────────────────────────────────────
	if use_verified:
		print("[kuzu] Reading verified reactions…")
		db_paths = [reaction_space._db_paths.get("verified")]
	else:
		print("[kuzu] Reading candidate reactions…")
		db_paths = reaction_space._get_candidate_db_paths()

	db_paths = [p for p in db_paths if p and os.path.exists(p)]
	if not db_paths:
		print("[kuzu] WARNING: No reaction databases found.")
		return

	molecules: Dict[str, Dict] = {}  # canonical_smiles → feature dict
	reactions: Dict[str, Dict] = {}  # canonical_reaction_smiles → {smiles, gen, reactants, products}

	# Cache for canonical SMILES to avoid redundant RDKit calls
	smi_cache: Dict[str, str] = {}

	def get_canonical(s: str) -> str:
		if s not in smi_cache:
			smi_cache[s] = _canonicalize(s)
		return smi_cache[s]

	for db_path in db_paths:
		for rxn_smi, gen in _iter_lmdb(db_path):
			reactant_smis, product_smis = _parse_reaction(rxn_smi)
			if not reactant_smis or not product_smis:
				continue

			# Canonicalize reactants and products
			can_reactants = [get_canonical(s) for s in reactant_smis]
			can_products = [get_canonical(s) for s in product_smis]

			# Generate a canonical ID for the reaction itself
			can_rxn_smi = ".".join(sorted(can_reactants)) + ">>" + ".".join(sorted(can_products))

			if can_rxn_smi not in reactions:
				reactions[can_rxn_smi] = {
					"smiles": rxn_smi,
					"gen": gen,
					"reactants": can_reactants,
					"products": can_products,
				}

			for smi in can_reactants + can_products:
				if smi not in molecules:
					molecules[smi] = {}  # features computed below

	print(f"[kuzu] Found {len(molecules):,} unique molecules (canonicalized), {len(reactions):,} reactions.")

	# ── Compute molecule features ─────────────────────────────────────────────
	print(f"[kuzu] Computing molecular features using model: {model_path}…")
	energy_predictor = EnergyPredictor(model_path)
	for i, smi in enumerate(molecules):
		if i % 500 == 0:
			print(f"  {i}/{len(molecules)}…", end="\r")
		feats = _mol_features(smi) if include_fingerprints else _mol_features.__wrapped__(smi)
		feats["predicted_free_energy"] = energy_predictor.predict_free_energy(smi)
		molecules[smi] = feats
	print()

	# ── Insert Molecule nodes ─────────────────────────────────────────────────
	print("[kuzu] Bulk inserting Molecule nodes…")
	mol_df = pd.DataFrame(
		[
			{
				"id": smi,
				"smiles": smi,
				"formula": feats["formula"],
				"morgan_fp": feats["morgan_fp"] if include_fingerprints else "",
				"mw": feats["mw"],
				"num_rings": feats["num_rings"],
				"num_rot_bonds": feats["num_rot_bonds"],
				"tpsa": feats["tpsa"],
				"logp": feats["logp"],
				"is_radical": feats["is_radical"],
				"predicted_free_energy": feats["predicted_free_energy"],
			}
			for smi, feats in molecules.items()
		]
	)
	# Ensure correct column order for COPY and drop any rare internal duplicates
	mol_df = mol_df[
		[
			"id",
			"smiles",
			"formula",
			"morgan_fp",
			"mw",
			"num_rings",
			"num_rot_bonds",
			"tpsa",
			"logp",
			"is_radical",
			"predicted_free_energy",
		]
	]
	mol_df.drop_duplicates(subset=["id"], inplace=True)

	load_dir = os.path.dirname(kuzu_dir) or "."
	os.makedirs(load_dir, exist_ok=True)

	mol_csv = os.path.join(load_dir, "molecules_load.csv")
	mol_df.to_csv(mol_csv, index=False)
	con.execute(f'COPY Molecule FROM "{mol_csv}"')
	os.remove(mol_csv)
	print(f"  Inserted {len(molecules):,} molecules.")

	# ── Insert Reaction nodes ─────────────────────────────────────────────────
	print("[kuzu] Bulk inserting Reaction nodes…")
	rxn_df = pd.DataFrame(
		[
			{
				"id": smi,
				"smiles": meta.get("smiles", smi),
				"gen": meta["gen"],
				"num_reactants": len(meta["reactants"]),
				"num_products": len(meta["products"]),
			}
			for smi, meta in reactions.items()
		]
	)
	rxn_df = rxn_df[["id", "smiles", "gen", "num_reactants", "num_products"]]
	rxn_df.drop_duplicates(subset=["id"], inplace=True)

	rxn_csv = os.path.join(load_dir, "reactions_load.csv")
	rxn_df.to_csv(rxn_csv, index=False)
	con.execute(f'COPY Reaction FROM "{rxn_csv}"')
	os.remove(rxn_csv)
	print(f"  Inserted {len(reactions):,} reactions.")

	# ── Insert REACTANT_OF edges ──────────────────────────────────────────────
	print("[kuzu] Bulk inserting REACTANT_OF edges…")
	reactant_edges_set = set()
	for rxn_smi, meta in reactions.items():
		for mol_smi in meta["reactants"]:
			if mol_smi in molecules:
				reactant_edges_set.add((mol_smi, rxn_smi))

	if reactant_edges_set:
		re_df = pd.DataFrame(list(reactant_edges_set), columns=["from", "to"])
		re_csv = os.path.join(load_dir, "reactant_edges_load.csv")
		re_df.to_csv(re_csv, index=False, header=False)
		con.execute(f'COPY REACTANT_OF FROM "{re_csv}"')
		os.remove(re_csv)
		print(f"  Inserted {len(reactant_edges_set):,} REACTANT_OF edges.")

	# ── Insert PRODUCT_OF edges ───────────────────────────────────────────────
	print("[kuzu] Bulk inserting PRODUCT_OF edges…")
	product_edges_set = set()
	for rxn_smi, meta in reactions.items():
		for mol_smi in meta["products"]:
			if mol_smi in molecules:
				product_edges_set.add((rxn_smi, mol_smi))

	if product_edges_set:
		po_df = pd.DataFrame(list(product_edges_set), columns=["from", "to"])
		po_csv = os.path.join(load_dir, "product_edges_load.csv")
		po_df.to_csv(po_csv, index=False, header=False)
		con.execute(f'COPY PRODUCT_OF FROM "{po_csv}"')
		os.remove(po_csv)
		print(f"  Inserted {len(product_edges_set):,} PRODUCT_OF edges.")

	# ── Summary ───────────────────────────────────────────────────────────────
	mol_count = con.execute("MATCH (m:Molecule) RETURN count(m)").get_next()[0]
	rxn_count = con.execute("MATCH (r:Reaction) RETURN count(r)").get_next()[0]
	re_count = con.execute("MATCH ()-[e:REACTANT_OF]->() RETURN count(e)").get_next()[0]
	po_count = con.execute("MATCH ()-[e:PRODUCT_OF]->() RETURN count(e)").get_next()[0]

	print("\n[kuzu] Export complete!")
	print(f"  Molecule nodes : {mol_count:,}")
	print(f"  Reaction nodes : {rxn_count:,}")
	print(f"  REACTANT_OF    : {re_count:,}")
	print(f"  PRODUCT_OF     : {po_count:,}")
	print(f"  Database dir   : {kuzu_dir}")


# ─── JSON export for the graph visualizer ─────────────────────────────────────


def export_to_visualizer_json(
	reaction_space: "ReactionSpace",
	output_path: str = "reaction_graph.json",
	max_nodes: int = 2000,
) -> None:
	"""
	Export the reaction network to a JSON format consumable by the
	Reaction Space Explorer HTML visualizer.

	The format is::

	    {
	      "nodes": [{"id": ..., "label": ..., "type": "molecule"|"reaction",
	                 "smiles": ..., "gen": ..., "formula": ...}, ...],
	      "links": [{"source": ..., "target": ..., "role": "reactant"|"product"}, ...]
	    }
	"""
	verified_db = reaction_space._db_paths.get("verified")
	if not verified_db or not os.path.exists(verified_db):
		print("WARNING: verified reactions DB not found.")
		return

	molecules: Dict[str, Dict] = {}
	reactions_data = []

	for rxn_smi, gen in _iter_lmdb(verified_db):
		reactant_smis, product_smis = _parse_reaction(rxn_smi)
		if not reactant_smis or not product_smis:
			continue

		# Canonicalize reactants and products
		can_reactants = [_canonicalize(s) for s in reactant_smis]
		can_products = [_canonicalize(s) for s in product_smis]

		# Canonical ID for the reaction
		can_rxn_smi = ".".join(sorted(can_reactants)) + ">>" + ".".join(sorted(can_products))

		rxn_id = f"rxn_{abs(hash(can_rxn_smi)) % 10**9}"
		reactions_data.append(
			{
				"id": rxn_id,
				"smiles": rxn_smi,
				"gen": gen,
				"label": f"{gen}: {rxn_smi[:30]}…" if len(rxn_smi) > 30 else f"{gen}: {rxn_smi}",
				"type": "reaction",
				"reactants": can_reactants,
				"products": can_products,
			}
		)

		for smi in can_reactants + can_products:
			if smi not in molecules:
				feats = _mol_features(smi)
				molecules[smi] = {
					"id": f"mol_{abs(hash(smi)) % 10**9}",
					"smiles": smi,
					"label": feats["formula"] or smi[:15],
					"type": "molecule",
					"formula": feats["formula"],
				}

		if len(molecules) + len(reactions_data) >= max_nodes:
			print(f"[export] Reached max_nodes={max_nodes}, truncating.")
			break

	nodes = list(molecules.values()) + [
		{k: v for k, v in r.items() if k not in ("reactants", "products")} for r in reactions_data
	]

	links = []
	mol_smi_to_id = {v["smiles"]: v["id"] for v in molecules.values()}
	for rxn in reactions_data:
		for smi in rxn["reactants"]:
			if smi in mol_smi_to_id:
				links.append({"source": mol_smi_to_id[smi], "target": rxn["id"], "role": "reactant"})
		for smi in rxn["products"]:
			if smi in mol_smi_to_id:
				links.append({"source": rxn["id"], "target": mol_smi_to_id[smi], "role": "product"})

	import json as _json

	with open(output_path, "w") as f:
		_json.dump({"nodes": nodes, "links": links}, f)

	print(f"[export] Wrote {len(nodes):,} nodes, {len(links):,} edges → {output_path}")


# ─── Update predictions ───────────────────────────────────────────────────────


def update_kuzu_predictions(
	kuzu_dir: str = "kuzu_db",
	model_path: str = "model.script",
	batch_size: int = 2048,
) -> None:
	"""
	Update the predicted_free_energy field for all Molecule nodes in an
	existing KùzuDB.
	"""
	if kuzu is None:
		raise ImportError("kuzu is required.")

	db = kuzu.Database(kuzu_dir)
	con = kuzu.Connection(db)

	print(f"[kuzu] Updating predictions using model: {model_path}…")
	energy_predictor = EnergyPredictor(model_path)

	if energy_predictor.model is None:
		print(f"Warning: Model file not found or failed to load: {model_path}")
		print("Molecules will be updated with NULL energy.")

	# 1. Fetch all molecule SMILES and IDs
	res = con.execute("MATCH (m:Molecule) RETURN m.id, m.smiles")
	all_mols = []
	while res.has_next():
		row = res.get_next()
		all_mols.append((row[0], row[1]))

	if not all_mols:
		print("No molecules found in database.")
		return

	print(f"Found {len(all_mols):,} molecules to update.")

	# 2. Update in batches
	for start in range(0, len(all_mols), batch_size):
		chunk = all_mols[start : start + batch_size]
		params = []
		for mid, smi in chunk:
			energy = energy_predictor.predict_free_energy(smi)
			params.append({"id": mid, "energy": energy})

		con.execute(
			"UNWIND $rows AS r MATCH (m:Molecule {id: r.id}) SET m.predicted_free_energy = CAST(r.energy, 'DOUBLE')",
			{"rows": params},
		)
		print(f"  Updated {min(start + batch_size, len(all_mols)):,}/{len(all_mols)}…", end="\r")

	print(f"\n[kuzu] Update complete for {len(all_mols):,} molecules.")


# ─── Convenience integration into ReactionSpace ───────────────────────────────
# Monkey-patch or call standalone — your choice.
#
# Example:
#   from reaction_space.kuzu_exporter import export_to_kuzu, export_to_visualizer_json
#
#   rs = ReactionSpace(input_csv="molecules.csv", num_generations=2)
#   rs.explore()
#   export_to_kuzu(rs, kuzu_dir="results/kuzu_db")
#   export_to_visualizer_json(rs, output_path="results/reaction_graph.json")
