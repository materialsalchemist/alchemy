document.addEventListener("DOMContentLoaded", async () => {
	await window.graphOps.initSigma();
	const config = await window.api.fetchConfig();
	if (config.has_thermo) {
		const thermoSect = document.getElementById("thermo-section");
		if (thermoSect) thermoSect.style.display = "block";
	}

	const connectBtn = document.getElementById("btn-connect-kuzu");
	if (connectBtn) {
		connectBtn.onclick = async () => {
			const filterThermo = document.getElementById("btn-thermo-filter")?.classList.contains("active") || false;
			const limit = 100; // Default limit since slider is removed
			window.ui?.showToast(`Connecting to Kuzu...`);
			window.graphOps.loadKuzuOverview(limit, filterThermo);
		};
	}

	const thermoBtn = document.getElementById("btn-thermo-filter");
	if (thermoBtn) {
		thermoBtn.onclick = (e) => {
			thermoBtn.classList.toggle("active");
			const filterThermo = thermoBtn.classList.contains("active");
			if (window.graphState && window.graphState.setThermoFilter) {
				window.graphState.setThermoFilter(filterThermo);
			}
		};
	}

	const renderer = window.graphState?.getRenderer();
	if (renderer) {
		document.getElementById("zoom-in")?.addEventListener("click", () => renderer.getCamera().animatedZoom(1.5));
		document.getElementById("zoom-out")?.addEventListener("click", () => renderer.getCamera().animatedZoom(0.7));
		document.getElementById("zoom-fit")?.addEventListener("click", () => renderer.getCamera().animatedReset());
	}

	const resetBtn = document.getElementById("btn-reset-view");
	if (resetBtn) {
		resetBtn.onclick = () => {
			window.graphOps.unfixNodes();
			window.graphState?.getRenderer()?.getCamera().animatedReset();
		};
	}

	document.querySelectorAll(".gen-toggle").forEach((el) => {
		if (!el.dataset.gen) return;
		el.onclick = () => {
			el.classList.toggle("active");
			const gen = el.dataset.gen;
			const activeGens = window.graphState.activeGens;
			if (el.classList.contains("active")) activeGens.add(gen);
			else activeGens.delete(gen);
			window.graphState?.getRenderer()?.refresh();
		};
	});

	const searchInput = document.getElementById("search-input");
	if (searchInput) searchInput.oninput = (e) => window.ui?.handleSearchInput(e);

	const demoBtn = document.getElementById("btn-load-demo");
	if (demoBtn) {
		demoBtn.onclick = () => {
			window.ui?.showToast("Loading demo graph...");
			const demo = {
				nodes: [
					{ id: "C", label: "Methane", type: "molecule", gen: "G0", smiles: "C" },
					{ id: "O2", label: "Oxygen", type: "molecule", gen: "G0", smiles: "[O][O]" },
					{ id: "RXN1", label: "Oxidation", type: "reaction", gen: "G1", smiles: "C.O2>>CO2.H2O" },
					{ id: "CO2", label: "CO2", type: "molecule", gen: "G1", smiles: "C(=O)=O" },
					{ id: "H2O", label: "Water", type: "molecule", gen: "G1", smiles: "O" },
				],
				links: [
					{ source: "C", target: "RXN1", role: "reactant" },
					{ source: "O2", target: "RXN1", role: "reactant" },
					{ source: "RXN1", target: "CO2", role: "product" },
					{ source: "RXN1", target: "H2O", role: "product" },
				],
			};
			const graph = window.graphState?.getGraph();
			if (graph) {
				graph.clear();
				window.graphOps.addDataToGraph(demo);
			}
		};
	}
});
