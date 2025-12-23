#!/usr/bin/env python3
import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt


def load_digraph_manual(json_path: str | Path) -> nx.DiGraph:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    G = nx.DiGraph()

    # --- add nodes ---
    for node in data.get("nodes", []):
        node_id = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node_id, **attrs)

    # --- add edges (from "links") ---
    for link in data.get("links", []):
        src = link["source"]
        tgt = link["target"]
        attrs = {k: v for k, v in link.items() if k not in ("source", "target")}
        G.add_edge(src, tgt, **attrs)

    return G


def draw_png(G: nx.DiGraph, out_png: str | Path):
    pos = nx.spring_layout(G, seed=0)

    # Shorten long hash IDs for readability
    labels = {
        n: (n[:6] + "…" + n[-6:] if isinstance(n, str) and len(n) > 14 else n)
        for n in G.nodes()
    }

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.6,
        connectionstyle="arc3,rad=0.15",
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("out_png")
    args = ap.parse_args()

    G = load_digraph_manual(args.json_path)

    # Optional: remove self-loops if undesired
    # G.remove_edges_from(nx.selfloop_edges(G))

    draw_png(G, args.out_png)
    print("Saved:", args.out_png)
