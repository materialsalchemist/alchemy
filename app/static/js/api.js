const API_BASE = "";

async function fetchConfig() {
	const res = await fetch(`${API_BASE}/graph/config`);
	return await res.json();
}

async function fetchOverview(limit = 40, filterThermo = false) {
	const res = await fetch(`${API_BASE}/graph/overview?limit=${limit}&filter_thermo=${filterThermo}`);
	return await res.json();
}

async function fetchNeighborhood(nodeId, hops = 1, filterThermo = false) {
	const res = await fetch(
		`${API_BASE}/graph/neighborhood?node_id=${encodeURIComponent(nodeId)}&hops=${hops}&filter_thermo=${filterThermo}`
	);
	return await res.json();
}

async function searchNodes(query) {
	const res = await fetch(`${API_BASE}/graph/search?q=${encodeURIComponent(query)}`);
	return await res.json();
}

window.api = { fetchConfig, fetchOverview, fetchNeighborhood, searchNodes };
