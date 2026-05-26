const graph = new graphology.Graph({ multi: false, type: "directed" });
let renderer = null;
let hoveredNode = null;
let selectedNode = null;
let activeGens = new Set(["G0", "G1", "G2", "G3"]);
let activeTypes = new Set(["molecule", "reaction"]);
let searchQuery = "";
let visibleNodesFromSearch = null;
let thermoFilterActive = false;

let layoutRunner = null;
let draggedNode = null;
let isDragging = false;

async function initSigma() {
	const container = document.getElementById("sigma-container");

	let SigmaRenderer, NodeSquareProgram;
	try {
		// Dynamically import Sigma and the NodeSquare plugin via jsDelivr ESM
		const SigmaModule = await import("https://cdn.jsdelivr.net/npm/sigma@3.0.3/+esm");
		const SquareModule = await import("https://cdn.jsdelivr.net/npm/@sigma/node-square@3.0.0/+esm");

		SigmaRenderer = SigmaModule.default;
		NodeSquareProgram = SquareModule.NodeSquareProgram;
	} catch (e) {
		console.error("Failed to load Sigma modules via ESM:", e);
		return;
	}

	try {
		renderer = new SigmaRenderer(graph, container, {
			renderEdgeLabels: false,
			labelFont: "IBM Plex Mono",
			labelSize: 10,
			labelColor: { color: "#8b91b0" },
			defaultEdgeType: "arrow",
			defaultNodeType: "circle",
			labelRenderedSizeThreshold: 6,

			// Register the custom square program
			nodeProgramClasses: {
				square: NodeSquareProgram,
			},

			nodeReducer: (node, data) => {
				const res = { ...data };

				// Dynamically assign the rendering shape based on the semantic type
				res.type = data.type === "reaction" ? "square" : "circle";

				// Hide labels for reactions as per user request (shape/color is enough), unless hovered
				if (data.type === "reaction") {
					res.label = hoveredNode === node ? data.smiles || data.id : "";
				} else if (hoveredNode === node) {
					// Show full string for hovered molecules as well
					res.label = data.smiles || data.id;
				}

				if (hoveredNode && node !== hoveredNode && !graph.neighbors(node).includes(hoveredNode)) {
					res.label = "";
					res.color = "#1c2030";
				}

				if (selectedNode && node !== selectedNode && !graph.neighbors(node).includes(selectedNode)) {
					res.color = "#1c2030";
					res.label = "";
				}

				if (!activeGens.has(data.gen) && data.type === "reaction") {
					res.hidden = true;
				}

				if (!activeTypes.has(data.type)) {
					res.hidden = true;
				}

				if (visibleNodesFromSearch && !visibleNodesFromSearch.has(node)) {
					res.hidden = true;
				}

				if (thermoFilterActive && data.type === "reaction" && data.thermodynamically_valid === false) {
					res.hidden = true;
				}

				return res;
			},
			edgeReducer: (edge, data) => {
				const res = { ...data };

				// Lower opacity for non-active edges (reduce hairball effect)
				res.color = "rgba(61, 68, 102, 0.15)";

				if (hoveredNode && !graph.hasExtremity(edge, hoveredNode)) res.hidden = true;
				if (selectedNode && !graph.hasExtremity(edge, selectedNode)) res.hidden = true;

				// Highlight active edges
				if (hoveredNode && graph.hasExtremity(edge, hoveredNode)) res.color = "rgba(91, 142, 255, 0.8)";
				if (selectedNode && graph.hasExtremity(edge, selectedNode)) res.color = "rgba(91, 142, 255, 0.8)";

				return res;
			},
		});
		console.log("Sigma v3 initialized successfully with Square nodes.");
	} catch (e) {
		console.error("Sigma initialization failed:", e);
		return;
	}

	// --- Drag & Drop Logic ---
	renderer.on("downNode", (e) => {
		isDragging = true;
		draggedNode = e.node;
		graph.setNodeAttribute(draggedNode, "fixed", true);
		renderer.getCamera().disable(); // Disable camera pan while dragging
	});

	renderer.getMouseCaptor().on("mousemove", (e) => {
		if (!isDragging || !draggedNode) return;

		// Get new position of node from mouse cursor
		const pos = renderer.viewportToGraph(e);

		graph.setNodeAttribute(draggedNode, "x", pos.x);
		graph.setNodeAttribute(draggedNode, "y", pos.y);

		// Prevent Sigma internal refresh from debouncing
		e.preventSigmaDefault();
		e.original.preventDefault();
		e.original.stopPropagation();
	});

	renderer.getMouseCaptor().on("mouseup", () => {
		if (draggedNode) {
			// Obsidian style: Leave the node 'fixed' after dragging so it stays in the new place
			graph.setNodeAttribute(draggedNode, "fixed", true);
		}
		isDragging = false;
		draggedNode = null;
		renderer.getCamera().enable();
	});

	// --- Interaction Events ---
	renderer.on("enterNode", ({ node }) => {
		hoveredNode = node;
		const data = graph.getNodeAttributes(node);
		window.ui?.showTooltip(data);
		renderer.refresh();
	});
	renderer.on("leaveNode", () => {
		hoveredNode = null;
		window.ui?.hideTooltip();
		renderer.refresh();
	});
	renderer.on("clickNode", ({ node }) => {
		selectedNode = node;
		const data = graph.getNodeAttributes(node);
		window.ui?.openDrawer(node, data);
		window.graphOps?.expandNeighborhood(node);
		renderer.refresh();
	});
	renderer.on("clickStage", () => {
		selectedNode = null;
		window.ui?.closeDrawer();
		renderer.refresh();
	});
}

function runLayout(duration = 0) {
	const btn = document.getElementById("btn-run-layout");
	const fa2 =
		window.graphologyLibrary?.layoutForceAtlas2 || (typeof forceAtlas2 !== "undefined" ? forceAtlas2 : null);

	if (!fa2) {
		console.error("ForceAtlas2 layout not found.");
		return;
	}

	// Normalize duration (if called from event, duration is the event object)
	const isPulse = typeof duration === "number" && duration > 0;

	// If we are already running a layout, stop it (unless it's a timed pulse)
	if (layoutRunner && !isPulse) {
		layoutRunner.stop();
		layoutRunner = null;
		btn.textContent = "Start Layout";
		btn.classList.remove("active");
		return;
	}

	if (isPulse && layoutRunner) {
		// Don't start a pulse if continuous layout is already running
		return;
	}

	const gravity = parseFloat(document.getElementById("sl-gravity")?.value ?? 1);
	const scaling = parseFloat(document.getElementById("sl-scaling")?.value ?? 10);

	const FA2Layout = window.graphologyLibrary?.FA2Layout;

	if (FA2Layout) {
		const settings = {
			gravity: gravity * 1.5, // Stronger gravity for "Obsidian" feel
			scalingRatio: scaling * 10, // Dramatically increase repulsion to spread out nodes
			strongGravityMode: true,
			adjustSizes: true,
			outboundAttractionDistribution: true,
			edgeWeightInfluence: 0.1, // Reduce link attraction to prevent clustering
			barnesHutOptimize: graph.order > 500,
		};

		if (duration > 0) {
			// Pulse mode: run for X ms then stop
			fa2.assign(graph, { iterations: duration / 50, settings });
			renderer?.refresh();
		} else {
			// Toggle mode
			layoutRunner = new FA2Layout(graph, { settings });
			layoutRunner.start();
			btn.textContent = "Stop Layout";
			btn.classList.add("active");
		}
	} else {
		// Fallback
		fa2.assign(graph, {
			iterations: duration > 0 ? 50 : 100,
			settings: { gravity, scalingRatio: scaling * 2, strongGravityMode: true, adjustSizes: true },
		});
		renderer?.refresh();
	}
}

function addDataToGraph(data) {
	if (!data?.nodes) {
		return 0;
	}

	let newNodesCount = 0;
	data.nodes.forEach((n) => {
		if (!graph.hasNode(n.id)) {
			newNodesCount++;
			// Calculate logarithmic size based on degree
			const degree = n.degree || 1;
			// Reduce baseSize significantly for reaction squares so they don't dominate
			const baseSize = n.type === "molecule" ? 4 : 3;
			const scaledSize = Math.max(baseSize, Math.min(baseSize + Math.log(degree + 1) * 3, 15));

			// Obsidian style labels: hide full smiles strings for reactions
			let displayLabel = n.id;
			if (n.type === "reaction") displayLabel = ""; // No text cluster for reactions
			if (displayLabel.length > 15) displayLabel = displayLabel.substring(0, 12) + "...";

			graph.addNode(n.id, {
				...n,
				x: Math.random() * 10,
				y: Math.random() * 10,
				size: scaledSize,
				color: n.type === "molecule" ? "#378ADD" : n.gen === "G0" ? "#f5a623" : "#1D9E75",
				label: displayLabel,
			});
		}
	});

	(data.links ?? []).forEach((l) => {
		const s = typeof l.source === "object" ? l.source.id : l.source;
		const t = typeof l.target === "object" ? l.target.id : l.target;

		if (graph.hasNode(s) && graph.hasNode(t) && !graph.hasEdge(s, t)) {
			graph.addEdge(s, t, { size: 1, color: "#3d4466", type: "arrow", role: l.role });
		}
	});

	window.ui?.updateStats(graph.order, graph.size);
	if (newNodesCount > 0) {
		runLayout(2000); // 2 second physics pulse for feedback
	}
	return newNodesCount;
}

async function loadKuzuOverview(limit = 40, filterThermo = false) {
	try {
		const data = await window.api.fetchOverview(limit, filterThermo);
		if (data?.nodes) {
			graph.clear();
			addDataToGraph(data);
			window.ui?.showToast(`Loaded ${data.nodes.length} nodes from Kuzu.`);
		}
	} catch (err) {
		console.error("API fetch failed:", err);
		window.ui?.showToast("Failed to fetch graph from Kuzu.", "error");
	}
}

async function expandNeighborhood(nodeId) {
	try {
		const filterThermo = document.getElementById("btn-thermo-filter")?.classList.contains("active") ?? false;
		const data = await window.api.fetchNeighborhood(nodeId, 1, filterThermo);
		if (data?.nodes) {
			const added = addDataToGraph(data);
			if (added > 0) {
				window.ui?.showToast(`Added ${added} new neighboring reactions/molecules.`);
			} else {
				window.ui?.showToast("Neighborhood already fully loaded.");
			}
		}
	} catch (err) {
		console.error("expandNeighborhood failed:", err);
		window.ui?.showToast("Failed to expand neighborhood.", "error");
	}
}

function unfixNodes() {
	graph.forEachNode((node) => {
		graph.removeNodeAttribute(node, "fixed");
	});
	renderer?.refresh();
}

window.graphOps = { initSigma, runLayout, unfixNodes, addDataToGraph, loadKuzuOverview, expandNeighborhood };
window.graphState = {
	getGraph: () => graph,
	getRenderer: () => renderer,
	setSelectedNode: (v) => {
		selectedNode = v;
	},
	setSearchQuery: (v) => {
		searchQuery = v;
		if (!v) {
			visibleNodesFromSearch = null;
		} else {
			const q = v.toLowerCase();
			const matches = new Set();
			graph.forEachNode((node, attrs) => {
				const lbl = (attrs.label || "").toLowerCase();
				const fullId = (attrs.id || "").toLowerCase();
				const smiles = (attrs.smiles || "").toLowerCase();
				const formula = (attrs.formula || "").toLowerCase();
				if (
					lbl === q ||
					fullId === q ||
					smiles === q ||
					formula === q ||
					lbl.includes(q) ||
					fullId.includes(q) ||
					smiles.includes(q) ||
					formula.includes(q)
				) {
					matches.add(node);
				}
			});
			visibleNodesFromSearch = new Set(matches);
			// Also show immediate neighbors for context
			matches.forEach((node) => {
				graph.forEachNeighbor(node, (neighbor) => {
					visibleNodesFromSearch.add(neighbor);
				});
			});
		}
		renderer?.refresh();
	},
	setThermoFilter: (v) => {
		thermoFilterActive = v;
		renderer?.refresh();
	},
	activeGens,
	activeTypes,
};
