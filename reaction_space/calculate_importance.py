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
	conn = db.connect()

	print("[importance] Loading graph from Kuzu into NetworkX…")

	# Fetch all edges to build NetworkX graph
	G = nx.Graph()

	res = conn.execute("MATCH (a)-[r]->(b) RETURN a.id, b.id, r._label")
	while res.has_next():
		u, v, label = res.get_next()
		G.add_edge(u, v, label=label)

	print(f"[importance] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

	if G.number_of_nodes() == 0:
		print("[importance] Graph is empty.")
		return

	# 1. Calculate PageRank
	print("[importance] Calculating PageRank…")
	pagerank = nx.pagerank(G)

	# 2. Calculate Degree Centrality
	print("[importance] Calculating Degree Centrality…")
	degree = nx.degree_centrality(G)

	# 3. Update Kuzu schema to include centrality
	print("[importance] Updating Kuzu schema…")
	try:
		conn.execute("ALTER TABLE Molecule ADD centrality DOUBLE DEFAULT 0.0")
	except Exception:
		pass  # Already exists

	try:
		conn.execute("ALTER TABLE Reaction ADD centrality DOUBLE DEFAULT 0.0")
	except Exception:
		pass  # Already exists

	# 4. Write back to Kuzu in batches
	print("[importance] Writing scores back to Kuzu…")

	# Group by node type if possible, or just update by ID
	# PageRank scores are typically small, so we'll store them as centrality

	scores = [{"id": node_id, "score": score} for node_id, score in pagerank.items()]

	batch_size = 2048
	for i in range(0, len(scores), batch_size):
		chunk = scores[i : i + batch_size]

		# We need to know if it's a Molecule or Reaction to use the right table
		# Or we can try to MATCH (n) if Kuzu supports it efficiently across tables
		# In this schema, we have Molecule and Reaction tables.

		# Update Molecules
		conn.execute("UNWIND $rows AS r MATCH (m:Molecule {id: r.id}) SET m.centrality = r.score", {"rows": chunk})

		# Update Reactions
		conn.execute("UNWIND $rows AS r MATCH (rx:Reaction {id: r.id}) SET rx.centrality = r.score", {"rows": chunk})

		print(f"  Updated {min(i + batch_size, len(scores))}/{len(scores)}...", end="\r")

	print("\n[importance] Done.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--kuzu-dir", default="reaction_space_results/kuzu_db")
	args = parser.parse_args()

	calculate_importance(args.kuzu_dir)
