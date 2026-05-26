def format_node(row_dict):
	n = row_dict["n"]
	node_type = "molecule" if "Molecule" in n["_label"] else "reaction"
	props = {k: v for k, v in n.items() if not k.startswith("_")}

	node_id = n["id"]

	if node_type == "molecule":
		# Descriptive label for molecules: Formula (SMILES)
		formula = props.get("formula", "")
		smiles = props.get("smiles", node_id)
		label = f"{formula} [{smiles}]" if formula else smiles
	else:
		# Descriptive label for reactions: Gen: SMILES
		gen = props.get("gen", "??")
		smiles = props.get("smiles", node_id)
		# Truncate long reaction smiles for the label
		s_short = (smiles[:30] + "...") if len(smiles) > 30 else smiles
		label = f"{gen}: {s_short}"

	return {"id": node_id, "type": node_type, "label": label, **props}


def check_thermo_validity(conn, rxn_ids):
	if not rxn_ids:
		return set(), {}

	reactant_res = conn.execute(
		"MATCH (m:Molecule)-[r:REACTANT_OF]->(rxn:Reaction) WHERE rxn.id IN $ids RETURN rxn.id, m.id, m.predicted_free_energy AS e, m.formula",
		{"ids": list(rxn_ids)},
	)
	product_res = conn.execute(
		"MATCH (rxn:Reaction)-[p:PRODUCT_OF]->(m:Molecule) WHERE rxn.id IN $ids RETURN rxn.id, m.id, m.predicted_free_energy AS e, m.formula",
		{"ids": list(rxn_ids)},
	)

	# details = { rxn_id: { "reactants": [{"id", "energy", "label"}], "products": [...] } }
	details = {rid: {"reactants": [], "products": []} for rid in rxn_ids}

	while reactant_res.has_next():
		row = reactant_res.get_next()
		rid, mid, energy, formula = row[0], row[1], row[2], row[3]
		if rid in details:
			details[rid]["reactants"].append({"id": mid, "energy": energy, "label": formula or mid})

	while product_res.has_next():
		row = product_res.get_next()
		rid, mid, energy, formula = row[0], row[1], row[2], row[3]
		if rid in details:
			details[rid]["products"].append({"id": mid, "energy": energy, "label": formula or mid})

	valid_rxns = set()
	for rid in rxn_ids:
		r_list = details[rid]["reactants"]
		p_list = details[rid]["products"]

		r_es = [x["energy"] for x in r_list]
		p_es = [x["energy"] for x in p_list]

		# If all participants have energy data, compare sums
		if r_es and p_es and all(e is not None for e in r_es) and all(e is not None for e in p_es):
			if sum(p_es) < sum(r_es):
				valid_rxns.add(rid)
		else:
			# Default to valid if data is missing or it's a source/sink
			valid_rxns.add(rid)

	return valid_rxns, details


def filter_graph(nodes, links, valid_rxns, thermo_details, filter_thermo):
	# Always tag nodes with validity and details so the UI can use them
	for node in nodes.values():
		if node["type"] == "reaction":
			node["thermodynamically_valid"] = node["id"] in valid_rxns
			node["thermo_details"] = thermo_details.get(node["id"], {"reactants": [], "products": []})

	if not filter_thermo:
		return list(nodes.values()), links

	filtered_nodes, filtered_links = {}, []
	for link in links:
		s_node, t_node = nodes[link["source"]], nodes[link["target"]]

		# Skip links involving invalid reactions if filtering is ON
		if (s_node["type"] == "reaction" and s_node["id"] not in valid_rxns) or (
			t_node["type"] == "reaction" and t_node["id"] not in valid_rxns
		):
			continue

		filtered_nodes[link["source"]], filtered_nodes[link["target"]] = s_node, t_node
		filtered_links.append(link)

	return list(filtered_nodes.values()), filtered_links
