document.addEventListener('DOMContentLoaded', function() {
	const graphContainer = document.getElementById('reaction-graph');

	if (typeof reactionDataUrl !== 'undefined' && graphContainer) {
		console.log('Attempting to load reaction data from:', reactionDataUrl);
		
		fetch(reactionDataUrl)
			.then(response => {
				if (!response.ok) {
					throw new Error(`HTTP error! status: ${response.status}`);
				}
				return response.json();
			})
			.then(data => {
				if (data.error) {
					console.error('Error loading reaction data:', data.error);
					graphContainer.innerHTML = `<p>Error loading reaction data: ${data.error}</p>`;
					return;
				}
				console.log('Reaction data loaded:', data);
				// Basic D3 rendering
				if (data.nodes && data.links) {
					renderGraph(data);
				} else {
					graphContainer.innerHTML = '<p>Reaction data is missing nodes or links.</p>';
				}
			})
			.catch(error => {
				console.error('Error fetching or parsing reaction data:', error);
				graphContainer.innerHTML = `<p>Could not load reaction data: ${error.message}. Check console for details.</p>`;
			});
	} else {
		if (graphContainer) {
			graphContainer.innerHTML = '<p>Reaction data URL not defined or graph container not found.</p>';
		}
		console.error('Reaction data URL (reactionDataUrl) is not defined or graph container not found.');
	}
});

function renderGraph(graph) {
	const width = 800;
	const height = 600;
	const graphContainer = d3.select("#reaction-graph");
	graphContainer.html("");

	const svg = graphContainer.append("svg")
		.attr("width", width)
		.attr("height", height)
		.style("border", "1px solid #ccc")
		.on("keydown", (event) => {
			switch (event.key) {
				case "+":
				case "=": // Numpad plus and standard plus
					if (event.ctrlKey) {
						event.preventDefault();
						svg.transition().duration(250).call(zoom.scaleBy, 1.2);
					}
					break;
				case "-":
				case "_":
					if (event.ctrlKey) {
						event.preventDefault();
						svg.transition().duration(250).call(zoom.scaleBy, 0.8);
					}
					break;
			}
		})
		.attr("tabindex", 0); // Make SVG focusable to receive key events

	const g = svg.append("g");

	const simulation = d3.forceSimulation(graph.nodes)
		.force("link", d3.forceLink(graph.links).id(d => d.id).distance(100))
		.force("charge", d3.forceManyBody().strength(-150))
		.force("center", d3.forceCenter(width / 2, height / 2));

	const link = g.append("g")
		.attr("stroke", "#999")
		.attr("stroke-opacity", 0.6)
		.selectAll("line")
		.data(graph.links)
		.join("line")
		.attr("stroke-width", d => Math.sqrt(d.value || 1));

	const node = g.append("g")
		.attr("stroke", "#fff")
		.attr("stroke-width", 1.5)
		.selectAll("circle")
		.data(graph.nodes)
		.join("circle")
		.attr("r", d => d.type === 'reaction' ? 8 : 5)
		.attr("fill", d => d.type === 'reaction' ? "#ff7f0e" : "#1f77b4")
		.call(drag(simulation));

	node.append("title")
		.text(d => d.id);

	const label = g.append("g")
		.attr("class", "labels")
		.selectAll("text")
		.data(graph.nodes)
		.join("text")
		.attr("dx", 10)
		.attr("dy", ".35em")
		.style("font-size", "10px")
		.text(d => d.id.length > 20 ? d.id.substring(0, 17) + "..." : d.id);

	// Update positions on each tick of the simulation
	simulation.on("tick", () => {
		link
			.attr("x1", d => d.source.x)
			.attr("y1", d => d.source.y)
			.attr("x2", d => d.target.x)
			.attr("y2", d => d.target.y);

		node
			.attr("cx", d => d.x)
			.attr("cy", d => d.y);
		
		label
			.attr("x", d => d.x)
			.attr("y", d => d.y);
	});

	// --- ZOOM & PAN LOGIC ---

	const zoom = d3.zoom()
		.scaleExtent([0.1, 8])
		.filter(event => {
			if (event.type === 'mousedown' || event.type === 'touchstart') {
				// Allow panning with middle mouse button or if spacebar is pressed
				if (event.button === 1 || event.buttons === 4 || event.ctrlKey) {
					return true;
				}
	
				// Prevent panning on left-click unless it's on the background
				return event.target.tagName.toLowerCase() === 'svg';
			}
			return true;
		})
		.on("zoom", (event) => {
			g.attr("transform", event.transform);
		});

	svg.call(zoom);

	d3.select(window).on("keydown", (event) => {
		if (event.code === "Space") {
			svg.style("cursor", "grab");
		}
	}).on("keyup", (event) => {
		if (event.code === "Space") {
			svg.style("cursor", "auto");
		}
	});

	// --- DRAG LOGIC ---
	function drag(simulation) {
		function dragstarted(event, d) {
			if (!event.active) simulation.alphaTarget(0.3).restart();
			d.fx = d.x;
			d.fy = d.y;
		}
		function dragged(event, d) {
			d.fx = event.x;
			d.fy = event.y;
		}
		function dragended(event, d) {
			if (!event.active) simulation.alphaTarget(0);
			d.fx = null;
			d.fy = null;
		}
		return d3.drag()
			.on("start", dragstarted)
			.on("drag", dragged)
			.on("end", dragended);
	}
}
