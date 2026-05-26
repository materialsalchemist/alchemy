from fastapi import APIRouter, Query, HTTPException
from app.database import get_conn
from app.utils import format_node, check_thermo_validity, filter_graph

router = APIRouter(prefix="/graph")


@router.get("/config")
async def get_config():
	conn = get_conn()
	if not conn:
		return {"has_thermo": False}
	try:
		res = conn.execute("MATCH (m:Molecule) RETURN m.predicted_free_energy LIMIT 1")
		return {"has_thermo": res.has_next() and res.get_next()[0] is not None}
	except:
		return {"has_thermo": False}


@router.get("/overview")
async def get_overview(limit: int = 50, filter_thermo: bool = False):
	conn = get_conn()
	if not conn:
		raise HTTPException(status_code=500, detail="Database connection not initialized")
	try:
		# Cap limit at 200 for stability in birds-eye view
		safe_limit = min(limit, 200)

		has_centrality = False
		try:
			check_res = conn.execute("MATCH (m:Molecule) RETURN m.centrality LIMIT 1")
			if check_res.has_next():
				has_centrality = True
		except:
			pass
		query = (
			"MATCH (m:Molecule) RETURN m.id ORDER BY m.centrality DESC LIMIT $limit"
			if has_centrality
			else "MATCH (m:Molecule)-[r]->() RETURN m.id, count(r) AS degree ORDER BY degree DESC LIMIT $limit"
		)

		top_ids = [row[0] for row in conn.execute(query, {"limit": safe_limit}).get_as_df().values]
		if not top_ids:
			return {"nodes": [], "links": []}

		nodes, links_raw, rxn_ids, internal_to_id = {}, [], set(), {}

		# Fetch in smaller chunks to avoid Kuzu Buffer Manager overflow
		chunk_size = 50
		for i in range(0, len(top_ids), chunk_size):
			chunk = top_ids[i : i + chunk_size]
			res = conn.execute("MATCH (n)-[e]-(m) WHERE n.id IN $ids RETURN n, e, m LIMIT 500", {"ids": chunk})

			while res.has_next():
				row = res.get_next()
				for node_val in [row[0], row[2]]:
					if node_val is None:
						continue
					node = format_node({"n": node_val})
					nodes[node["id"]] = node
					internal_to_id[str(node_val["_id"])] = node["id"]
					if node["type"] == "reaction":
						rxn_ids.add(node["id"])
				links_raw.append(row[1])

		valid_rxns, thermo_details = check_thermo_validity(conn, rxn_ids)

		links = []
		for e in links_raw:
			s_id, d_id = str(e["_src"]), str(e["_dst"])
			if s_id in internal_to_id and d_id in internal_to_id:
				links.append(
					{
						"source": internal_to_id[s_id],
						"target": internal_to_id[d_id],
						"role": e["_label"].split("_")[0].lower(),
					}
				)

		# Calculate degrees for visual scaling
		for node in nodes.values():
			node["degree"] = 0
		for link in links:
			if link["source"] in nodes:
				nodes[link["source"]]["degree"] += 1
			if link["target"] in nodes:
				nodes[link["target"]]["degree"] += 1

		final_nodes, final_links = filter_graph(nodes, links, valid_rxns, thermo_details, filter_thermo)
		return {"nodes": final_nodes, "links": final_links}
	except Exception as e:
		print(f"Overview error: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/neighborhood")
async def get_neighborhood(node_id: str, hops: int = Query(1, ge=1, le=3), filter_thermo: bool = False):
	conn = get_conn()
	if not conn:
		raise HTTPException(status_code=500, detail="Database connection not initialized")
	try:
		# Deterministic neighborhood query with higher limit
		query = "MATCH (n {id: $node_id})-[e]-(m) RETURN n, e, m ORDER BY m.id LIMIT 1000"
		res = conn.execute(query, {"node_id": node_id})
		nodes, links_raw, rxn_ids, internal_to_id = {}, [], set(), {}
		while res.has_next():
			row = res.get_next()
			for node_val in [row[0], row[2]]:
				node = format_node({"n": node_val})
				nodes[node["id"]] = node
				internal_to_id[str(node_val["_id"])] = node["id"]
				if node["type"] == "reaction":
					rxn_ids.add(node["id"])
			links_raw.append(row[1])

		valid_rxns, thermo_details = check_thermo_validity(conn, rxn_ids)
		seen_links = set()
		links = []
		for e in links_raw:
			edge_key = (str(e["_src"]), str(e["_dst"]), e["_label"])
			if edge_key not in seen_links:
				links.append(
					{
						"source": internal_to_id[str(e["_src"])],
						"target": internal_to_id[str(e["_dst"])],
						"role": e["_label"].split("_")[0].lower(),
					}
				)
				seen_links.add(edge_key)

		# Calculate degrees for visual scaling
		for node in nodes.values():
			node["degree"] = 0
		for link in links:
			if link["source"] in nodes:
				nodes[link["source"]]["degree"] += 1
			if link["target"] in nodes:
				nodes[link["target"]]["degree"] += 1

		final_nodes, final_links = filter_graph(nodes, links, valid_rxns, thermo_details, filter_thermo)
		return {"nodes": final_nodes, "links": final_links}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_nodes(q: str = Query(..., min_length=2)):
	conn = get_conn()
	if not conn:
		raise HTTPException(status_code=500, detail="Database connection not initialized")
	try:
		res = conn.execute(
			"MATCH (n) WHERE n.smiles CONTAINS $q OR n.formula CONTAINS $q OR n.id CONTAINS $q RETURN n LIMIT 20",
			{"q": q},
		)
		results = []
		while res.has_next():
			results.append(format_node({"n": res.get_next()[0]}))
		return results
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
