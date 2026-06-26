"""
reaction_space/calculate_importance.py

Offline script to calculate node centrality (Degree, PageRank)
and store it in KuzuDB for importance-based sampling in the visualizer.
"""

import kuzu
import networkx as nx
import os
import argparse

def calculate_importance(kuzu_dir: str):
	if not os.path.exists(kuzu_dir):
		print(f"Error: KuzuDB not found at {kuzu_dir}")
		return

	db = kuzu.Database(kuzu_dir)
	conn = kuzu.Connection(db)

	print("[importance] Loading graph from Kuzu into NetworkX…", flush=True)

	G = nx.Graph()

	for rel_label in ("REACTANT_OF", "PRODUCT_OF"):
		res = conn.execute(f"MATCH (a)-[r:{rel_label}]->(b) RETURN a.id, b.id")
		while res.has_next():
			u, v = res.get_next()
			G.add_edge(u, v, label=rel_label)

	print(f"[importance] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.", flush=True)

	if G.number_of_nodes() == 0:
		print("[importance] Graph is empty.")
		return

	print("[importance] Calculating PageRank…", flush=True)
	pagerank = nx.pagerank(G)

	print("[importance] Calculating Degree Centrality…", flush=True)
	degree = nx.degree_centrality(G)

	print("[importance] Updating Kuzu schema…", flush=True)
	try:
		conn.execute("ALTER TABLE Molecule ADD centrality DOUBLE DEFAULT 0.0")
	except Exception:
		pass  # Already exists

	try:
		conn.execute("ALTER TABLE Reaction ADD centrality DOUBLE DEFAULT 0.0")
	except Exception:
		pass  # Already exists

	# Write back to Kuzu in batches
	print("[importance] Writing scores back to Kuzu…", flush=True)

	# Group by node type if possible, or just update by ID
	# PageRank scores are typically small, so we'll store them as centrality

	scores = [{"id": node_id, "score": score} for node_id, score in pagerank.items()]

	batch_size = 2048
	for i in range(0, len(scores), batch_size):
		chunk = scores[i : i + batch_size]
		# Update Molecules
		conn.execute("UNWIND $rows AS r MATCH (m:Molecule {id: r.id}) SET m.centrality = r.score", {"rows": chunk})

		# Update Reactions
		conn.execute("UNWIND $rows AS r MATCH (rx:Reaction {id: r.id}) SET rx.centrality = r.score", {"rows": chunk})

		print(f"  Updated {min(i + batch_size, len(scores))}/{len(scores)}...", end="\r")

	print("\n[importance] Done.", flush=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--kuzu-dir", default="reaction_space_results/kuzu_db")
	args = parser.parse_args()

	calculate_importance(args.kuzu_dir)
