function updateStats(nodes, edges) {
	document.getElementById("stat-nodes").innerHTML = `nodes: <b>${nodes}</b>`;
	document.getElementById("stat-edges").innerHTML = `edges: <b>${edges}</b>`;
	document.getElementById("empty-state").style.display = nodes > 0 ? "none" : "block";
}
function showTooltip(d) {
	const tt = document.getElementById("tooltip");
	tt.querySelector("#tt-type").textContent = d.type;
	tt.querySelector("#tt-label").textContent = d.label || d.smiles || d.id;
	tt.querySelector("#tt-id").textContent = d.smiles || d.id;
	tt.classList.add("visible");
}
function hideTooltip() {
	document.getElementById("tooltip").classList.remove("visible");
}
function openDrawer(nodeId, d) {
	const drawer = document.getElementById("detail-drawer");
	drawer.classList.add("open");
	const inner = document.getElementById("drawer-inner");
	const graph = window.graphState.getGraph();

	const inEdges = graph.inEdges(nodeId);
	const outEdges = graph.outEdges(nodeId);

	let thermoHtml = "";
	if (d.type === "reaction") {
		// Use the comprehensive details provided by the server
		const reactants = d.thermo_details?.reactants || [];
		const products = d.thermo_details?.products || [];

		const sumR = reactants.reduce((acc, curr) => acc + (curr.energy || 0), 0);
		const sumP = products.reduce((acc, curr) => acc + (curr.energy || 0), 0);
		const deltaG = sumP - sumR;

		thermoHtml = `
      <div style="margin-top:20px">
        <b style="color:var(--accent);font-size:12px;text-transform:uppercase">Thermodynamics</b>
        <div style="margin-top:8px;font-size:12px">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span>Reactants:</span>
            <span>${sumR.toFixed(2)} eV</span>
          </div>
          ${reactants.map((r) => `<div style="padding-left:10px;color:var(--text2);font-size:11px">• ${r.label}: ${r.energy !== undefined && r.energy !== null ? r.energy.toFixed(3) : "N/A"}</div>`).join("")}
          
          <div style="display:flex;justify-content:space-between;margin-top:8px;margin-bottom:4px">
            <span>Products:</span>
            <span>${sumP.toFixed(2)} eV</span>
          </div>
          ${products.map((p) => `<div style="padding-left:10px;color:var(--text2);font-size:11px">• ${p.label}: ${p.energy !== undefined && p.energy !== null ? p.energy.toFixed(3) : "N/A"}</div>`).join("")}
          
          <div style="display:flex;justify-content:space-between;margin-top:12px;padding-top:8px;border-top:1px solid #3d4466;font-weight:bold">
            <span>ΔG:</span>
            <span style="color:${deltaG <= 0 ? "#1D9E75" : "#ff4d4f"}">${deltaG.toFixed(3)} eV</span>
          </div>
        </div>
      </div>
    `;
	}

	inner.innerHTML = `
    <div style="display:flex;justify-content:space-between">
      <b style="color:var(--accent)">${d.type.toUpperCase()}</b>
      <div class="drawer-close" onclick="window.ui.closeDrawer()">✕</div>
    </div>
    <div style="font-size:18px;margin-top:8px">${d.label}</div>
    <div class="smiles-box" style="margin-top:10px">${d.smiles || d.id}</div>
    <table class="prop-table" style="margin-top:15px">
      <tr><td>ID</td><td>${nodeId}</td></tr>
      <tr><td>Gen</td><td>${d.gen || "N/A"}</td></tr>
      <tr><td>Degree</td><td>${graph.degree(nodeId)} (In:${inEdges.length}, Out:${outEdges.length})</td></tr>
      ${d.predicted_free_energy !== undefined && d.predicted_free_energy !== null ? `<tr><td>Free Energy</td><td>${d.predicted_free_energy.toFixed(4)} eV</td></tr>` : ""}
    </table>
    ${thermoHtml}
  `;
}
function closeDrawer() {
	document.getElementById("detail-drawer").classList.remove("open");
	window.graphState.setSelectedNode(null);
	const renderer = window.graphState.getRenderer();
	if (renderer) renderer.refresh();
}
function handleSearchInput(e) {
	const q = e.target.value.trim();

	// Filter nodes in the visual graph
	if (window.graphState && window.graphState.setSearchQuery) {
		window.graphState.setSearchQuery(q.length >= 2 ? q : "");
	}

	const searchResults = document.getElementById("search-results");
	searchResults.innerHTML = "";
	if (q.length < 2) return;
	window.api
		.searchNodes(q)
		.then((nodes) => {
			nodes.forEach((n) => {
				const div = document.createElement("div");
				div.className = "mol-item";

				const typeIcon = n.type === "molecule" ? "⬡" : "⇄";
				const typeColor = n.type === "molecule" ? "var(--mol-color)" : "var(--rxn-color)";

				div.innerHTML = `<span style="color:${typeColor};margin-right:6px">${typeIcon}</span> ${n.label}`;
				div.title = n.id; // Show full ID on hover

				div.onclick = async () => {
					const graph = window.graphState.getGraph();
					if (!graph.hasNode(n.id)) await window.graphOps.expandNeighborhood(n.id);
					const renderer = window.graphState.getRenderer();
					const pos = renderer.getNodeDisplayData(n.id);
					if (pos && typeof pos.x === "number" && typeof pos.y === "number") {
						renderer.getCamera().animate(pos, { duration: 500 });
					}
					window.graphState.setSelectedNode(n.id);

					// Clear search UI and visual filter
					document.getElementById("search-input").value = "";
					document.getElementById("search-results").innerHTML = "";
					if (window.graphState.setSearchQuery) window.graphState.setSearchQuery("");

					openDrawer(n.id, graph.getNodeAttributes(n.id));
					renderer.refresh();
				};
				searchResults.appendChild(div);
			});
		})
		.catch((e) => {});
}
function showToast(message, type = "info") {
	const container = document.getElementById("toast-container");
	if (!container) return;

	const toast = document.createElement("div");
	toast.style.background = "var(--surface)";
	toast.style.color = "var(--text)";
	toast.style.border = "1px solid var(--accent)";
	toast.style.padding = "10px 16px";
	toast.style.borderRadius = "8px";
	toast.style.marginBottom = "10px";
	toast.style.fontSize = "12px";
	toast.style.fontFamily = "var(--font)";
	toast.style.boxShadow = "0 4px 12px rgba(0,0,0,0.5)";
	toast.style.opacity = "0";
	toast.style.transition = "opacity 0.3s ease";
	toast.style.pointerEvents = "auto";

	toast.innerHTML = `<b style="color:var(--accent)">SYSTEM:</b> ${message}`;

	container.appendChild(toast);

	// Fade in
	setTimeout(() => {
		toast.style.opacity = "1";
	}, 10);

	// Fade out and remove
	setTimeout(() => {
		toast.style.opacity = "0";
		setTimeout(() => {
			toast.remove();
		}, 300);
	}, 3000);
}

window.ui = { updateStats, showTooltip, hideTooltip, openDrawer, closeDrawer, handleSearchInput, showToast };
