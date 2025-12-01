// Table sorting functionality
function sortTable(n) {
    const table = document.getElementById("mainTable");
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.rows);
    const header = table.querySelectorAll('th')[n];
    let dir = header.getAttribute('data-dir') === 'asc' ? 'desc' : 'asc';

    table.querySelectorAll('th').forEach(th => {
        th.setAttribute('data-dir', '');
        th.classList.remove('asc', 'desc');
        th.classList.add('sort-icon');
    });

    header.setAttribute('data-dir', dir);
    header.classList.remove('sort-icon');
    header.classList.add(dir);

    rows.sort((rowA, rowB) => {
        const cellA = rowA.cells[n];
        const cellB = rowB.cells[n];
        let a = cellA.getAttribute('data-sort') || cellA.innerText.trim();
        let b = cellB.getAttribute('data-sort') || cellB.innerText.trim();

        const parseCurrency = (str) => {
            if (!str) return 0;
            str = str.replace(/[$,]/g, '');
            if (str.includes('M')) return parseFloat(str) * 1000000;
            if (str.includes('k')) return parseFloat(str) * 1000;
            return parseFloat(str);
        };

        let aNum = parseFloat(a);
        let bNum = parseFloat(b);

        if (cellA.innerText.includes('$')) { aNum = parseCurrency(cellA.innerText); }
        if (cellB.innerText.includes('$')) { bNum = parseCurrency(cellB.innerText); }

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return dir === 'asc' ? aNum - bNum : bNum - aNum;
        }

        return dir === 'asc' ? a.localeCompare(b) : b.localeCompare(a);
    });

    rows.forEach(row => tbody.appendChild(row));
}

// Table filtering functionality
function filterTable() {
    const input = document.getElementById("searchInput");
    const filter = input.value.toUpperCase();
    const table = document.getElementById("mainTable");
    const tr = table.getElementsByTagName("tr");

    for (let i = 1; i < tr.length; i++) {
        const td = tr[i].getElementsByTagName("td")[0];
        if (td) {
            const txtValue = td.textContent || td.innerText;
            if (txtValue.toUpperCase().indexOf(filter) > -1) {
                tr[i].style.display = "";
                if (filter.length > 0) {
                    tr[i].classList.add("bg-yellow-50", "text-gray-900");
                } else {
                    tr[i].classList.remove("bg-yellow-50", "text-gray-900");
                }
            } else {
                tr[i].style.display = "none";
                tr[i].classList.remove("bg-yellow-50");
            }
        }
    }
}

// Tab switching functionality
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.getElementById('tab-' + tabName).classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('btn-' + tabName).classList.add('active');
}

// Hamburger menu functionality
function toggleHamburger() {
    const menu = document.getElementById('hamburger-menu');
    menu.classList.toggle('hidden');
}

// Close hamburger menu when clicking outside
document.addEventListener('click', function(event) {
    const menu = document.getElementById('hamburger-menu');
    const btn = document.getElementById('hamburger-btn');
    if (menu && !menu.contains(event.target) && !btn.contains(event.target)) {
        menu.classList.add('hidden');
    }
});

// About modal functionality
function showAboutModal() {
    document.getElementById('about-modal').classList.remove('hidden');
    document.getElementById('hamburger-menu').classList.add('hidden');
}

function closeAboutModal() {
    document.getElementById('about-modal').classList.add('hidden');
}

// Help modal functionality
function showHelpModal() {
    document.getElementById('help-modal').classList.remove('hidden');
    document.getElementById('hamburger-menu').classList.add('hidden');
}

function closeHelpModal() {
    document.getElementById('help-modal').classList.add('hidden');
}

// Logout functionality
function logout() {
    fetch('/logout', { method: 'POST' })
        .then(() => window.location.href = '/login')
        .catch(() => window.location.href = '/login');
}

// Close modals with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        document.getElementById('modal')?.classList.add('hidden');
        document.getElementById('about-modal')?.classList.add('hidden');
        document.getElementById('help-modal')?.classList.add('hidden');
        document.getElementById('smart-search-modal')?.classList.add('hidden');
    }
});

// Smart Search functionality
function showSmartSearchModal() {
    document.getElementById('smart-search-modal').classList.remove('hidden');
    document.getElementById('smart-query').focus();
    // Reset form
    document.getElementById('smart-search-form').reset();
    document.getElementById('search-results').classList.add('hidden');
    document.getElementById('search-loading').classList.add('hidden');
}

function closeSmartSearchModal() {
    document.getElementById('smart-search-modal').classList.add('hidden');
}

async function performSmartSearch(event) {
    event.preventDefault();

    const query = document.getElementById('smart-query').value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }

    // Show loading, hide results
    document.getElementById('search-loading').classList.remove('hidden');
    document.getElementById('search-results').classList.add('hidden');
    document.getElementById('search-btn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('query', query);

        const response = await fetch('/smart_search', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Hide loading
        document.getElementById('search-loading').classList.add('hidden');
        document.getElementById('search-btn').disabled = false;

        if (data.error) {
            alert('Search error: ' + data.error);
            return;
        }

        if (data.message) {
            alert(data.message);
            return;
        }

        // Display results
        displaySmartSearchResults(data.results);

    } catch (error) {
        console.error('Search error:', error);
        document.getElementById('search-loading').classList.add('hidden');
        document.getElementById('search-btn').disabled = false;
        alert('Search failed: ' + error.message);
    }
}

function displaySmartSearchResults(results) {
    const resultsList = document.getElementById('results-list');

    if (!results || results.length === 0) {
        resultsList.innerHTML = '<p class="text-sm text-gray-500 text-center py-4">No matching investigators found.</p>';
        document.getElementById('search-results').classList.remove('hidden');
        return;
    }

    let html = '<div class="space-y-3">';

    results.forEach((inv, index) => {
        const technologies = inv.technologies || [];
        const populations = inv.populations || [];

        // Debug logging
        console.log('Investigator:', inv.name, 'Match %:', inv.match_percentage, 'Final score:', inv.final_score);

        // Use match_percentage from backend (already computed as percentage)
        let similarity;
        if (typeof inv.match_percentage === 'number') {
            similarity = inv.match_percentage.toFixed(1);
        } else if (typeof inv.final_score === 'number') {
            similarity = (inv.final_score * 100).toFixed(1);
        } else {
            similarity = '0.0';
        }

        html += `
            <div class="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition">
                <div class="flex justify-between items-start mb-2">
                    <h5 class="font-semibold text-gray-900 text-sm">${index + 1}. ${inv.name}</h5>
                    <span class="text-xs bg-amber-100 text-amber-800 px-2 py-1 rounded font-medium">${similarity}% match</span>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                    <div>
                        <p class="text-xs font-semibold text-gray-500 uppercase mb-1">Technologies & Methods</p>
                        <div class="flex flex-wrap gap-1">
                            ${technologies.length > 0
                                ? technologies.map(t => `<span class="bg-indigo-50 text-indigo-700 text-xs px-2 py-0.5 rounded">${t}</span>`).join('')
                                : '<span class="text-xs text-gray-400 italic">Not specified</span>'}
                        </div>
                    </div>

                    <div>
                        <p class="text-xs font-semibold text-gray-500 uppercase mb-1">Study Populations</p>
                        <div class="flex flex-wrap gap-1">
                            ${populations.length > 0
                                ? populations.map(p => `<span class="bg-emerald-50 text-emerald-700 text-xs px-2 py-0.5 rounded">${p}</span>`).join('')
                                : '<span class="text-xs text-gray-400 italic">Not specified</span>'}
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';

    resultsList.innerHTML = html;
    document.getElementById('search-results').classList.remove('hidden');
}

// =============================================================================
// 3D UMAP VISUALIZATION WITH THREE.JS
// =============================================================================

let viz3D = {
    scene: null,
    camera: null,
    renderer: null,
    labelRenderer: null,
    controls: null,
    nodes: [],
    spheres: [],
    animationId: null,
    raycaster: new THREE.Raycaster(),
    mouse: new THREE.Vector2(),
    selectedSphere: null,
    currentLabel: null,
    clusterNames: {}  // AIDEV-NOTE: Maps cluster_id to descriptive cluster name
};

// Cluster colors (8 distinct colors)
const CLUSTER_COLORS = [
    0x3B82F6, // blue
    0xEF4444, // red
    0x10B981, // green
    0xF59E0B, // amber
    0x8B5CF6, // purple
    0xEC4899, // pink
    0x06B6D4, // cyan
    0xF97316  // orange
];

function show3DVisualization() {
    document.getElementById('viz-3d-modal').classList.remove('hidden');
    load3DData();
}

function close3DVisualization() {
    document.getElementById('viz-3d-modal').classList.add('hidden');
    if (viz3D.animationId) {
        cancelAnimationFrame(viz3D.animationId);
    }
    // Clean up Three.js resources
    if (viz3D.renderer) {
        viz3D.renderer.dispose();
    }
    if (viz3D.labelRenderer) {
        viz3D.labelRenderer.domElement.remove();
    }
}

async function load3DData() {
    try {
        const response = await fetch('/umap_data');
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            close3DVisualization();
            return;
        }

        viz3D.nodes = data.nodes;
        viz3D.clusterNames = data.cluster_names || {};  // AIDEV-NOTE: Store cluster names from backend
        document.getElementById('viz-info').textContent = `${data.nodes.length} investigators`;

        // Initialize 3D scene
        init3DScene();

        // Hide loading indicator
        document.getElementById('viz-loading').classList.add('hidden');

    } catch (error) {
        console.error('Failed to load 3D data:', error);
        alert('Failed to load visualization data: ' + error.message);
        close3DVisualization();
    }
}

function init3DScene() {
    const container = document.getElementById('viz-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    viz3D.scene = new THREE.Scene();
    viz3D.scene.background = new THREE.Color(0x0a0a0a);

    // Camera
    viz3D.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    viz3D.camera.position.set(0, 0, 50);

    // Renderer
    viz3D.renderer = new THREE.WebGLRenderer({ antialias: true });
    viz3D.renderer.setSize(width, height);
    container.appendChild(viz3D.renderer.domElement);

    // CSS2D Label Renderer
    viz3D.labelRenderer = new THREE.CSS2DRenderer();
    viz3D.labelRenderer.setSize(width, height);
    viz3D.labelRenderer.domElement.style.position = 'absolute';
    viz3D.labelRenderer.domElement.style.top = '0px';
    viz3D.labelRenderer.domElement.style.pointerEvents = 'none';
    container.appendChild(viz3D.labelRenderer.domElement);

    // OrbitControls
    viz3D.controls = new THREE.OrbitControls(viz3D.camera, viz3D.renderer.domElement);
    viz3D.controls.enableDamping = true;
    viz3D.controls.dampingFactor = 0.05;

    // Remove labels when user starts panning/zooming
    viz3D.controls.addEventListener('start', function() {
        if (viz3D.currentLabel) {
            viz3D.currentLabel.parent.remove(viz3D.currentLabel);
            viz3D.currentLabel = null;
        }
        if (viz3D.selectedSphere) {
            const prevColor = CLUSTER_COLORS[viz3D.selectedSphere.userData.cluster % CLUSTER_COLORS.length];
            viz3D.selectedSphere.material.color.setHex(prevColor);
            viz3D.selectedSphere = null;
        }
    });

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    viz3D.scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(10, 10, 10);
    viz3D.scene.add(pointLight);

    // Create spheres for each investigator
    viz3D.spheres = [];
    viz3D.nodes.forEach(node => {
        // Size sphere by funding (min 0.2, max 2.0)
        const fundingMil = node.funding / 1000000;
        const radius = Math.max(0.2, Math.min(2.0, 0.2 + fundingMil * 0.3));

        // Color by cluster
        const color = CLUSTER_COLORS[node.cluster % CLUSTER_COLORS.length];

        const geometry = new THREE.SphereGeometry(radius, 16, 16);
        const material = new THREE.MeshPhongMaterial({ color: color, shininess: 30 });
        const sphere = new THREE.Mesh(geometry, material);

        // Position
        sphere.position.set(node.x * 10, node.y * 10, node.z * 10);

        // Store reference to node data
        sphere.userData = node;

        viz3D.scene.add(sphere);
        viz3D.spheres.push(sphere);
    });

    // Calculate bounding box and center camera on the cloud
    if (viz3D.spheres.length > 0) {
        const box = new THREE.Box3();
        viz3D.spheres.forEach(sphere => box.expandByObject(sphere));

        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

        // Position camera to see the whole cloud
        const cameraDistance = maxDim * 1.5;
        viz3D.camera.position.set(center.x, center.y, center.z + cameraDistance);
        viz3D.controls.target.copy(center);
        viz3D.controls.update();
    }

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Handle sphere clicks
    viz3D.renderer.domElement.addEventListener('click', onSphereClick, false);

    // Start animation loop
    animate();
}

function animate() {
    viz3D.animationId = requestAnimationFrame(animate);
    viz3D.controls.update();
    viz3D.renderer.render(viz3D.scene, viz3D.camera);
    viz3D.labelRenderer.render(viz3D.scene, viz3D.camera);
}

function onWindowResize() {
    const container = document.getElementById('viz-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    viz3D.camera.aspect = width / height;
    viz3D.camera.updateProjectionMatrix();
    viz3D.renderer.setSize(width, height);
    viz3D.labelRenderer.setSize(width, height);
}

function searchInvestigator(event) {
    const query = event.target.value.toLowerCase().trim();

    if (!query) {
        // Reset all sphere colors
        viz3D.spheres.forEach(sphere => {
            const color = CLUSTER_COLORS[sphere.userData.cluster % CLUSTER_COLORS.length];
            sphere.material.color.setHex(color);
        });
        return;
    }

    // Find matching investigators
    let found = false;
    viz3D.spheres.forEach(sphere => {
        const name = sphere.userData.name.toLowerCase();

        if (name.includes(query)) {
            // Highlight matching sphere
            sphere.material.color.setHex(0xFFFF00); // Yellow highlight

            if (!found) {
                // Center camera on first match
                const pos = sphere.position;
                viz3D.controls.target.set(pos.x, pos.y, pos.z);
                viz3D.camera.position.set(pos.x, pos.y, pos.z + 20);
                found = true;
            }
        } else {
            // Dim non-matching spheres
            sphere.material.color.setHex(0x333333);
        }
    });
}

function onSphereClick(event) {
    // Calculate mouse position in normalized device coordinates (-1 to +1)
    const rect = viz3D.renderer.domElement.getBoundingClientRect();
    viz3D.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    viz3D.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Update the picking ray with the camera and mouse position
    viz3D.raycaster.setFromCamera(viz3D.mouse, viz3D.camera);

    // Calculate objects intersecting the picking ray
    const intersects = viz3D.raycaster.intersectObjects(viz3D.spheres);

    if (intersects.length > 0) {
        const clickedSphere = intersects[0].object;
        const investigator = clickedSphere.userData;

        // If clicking the same sphere again, toggle it off
        if (viz3D.selectedSphere === clickedSphere) {
            // Remove label
            if (viz3D.currentLabel) {
                viz3D.currentLabel.parent.remove(viz3D.currentLabel);
                viz3D.currentLabel = null;
            }
            // Reset color
            const prevColor = CLUSTER_COLORS[clickedSphere.userData.cluster % CLUSTER_COLORS.length];
            clickedSphere.material.color.setHex(prevColor);
            viz3D.selectedSphere = null;
            return;
        }

        // Reset previously selected sphere
        if (viz3D.selectedSphere) {
            const prevColor = CLUSTER_COLORS[viz3D.selectedSphere.userData.cluster % CLUSTER_COLORS.length];
            viz3D.selectedSphere.material.color.setHex(prevColor);
        }

        // Remove previous label if exists
        if (viz3D.currentLabel) {
            viz3D.currentLabel.parent.remove(viz3D.currentLabel);
            viz3D.currentLabel = null;
        }

        // Highlight selected sphere
        clickedSphere.material.color.setHex(0xFFFFFF); // White highlight
        viz3D.selectedSphere = clickedSphere;

        // Get cluster name (or fall back to "Cluster N" if not available)
        const clusterName = viz3D.clusterNames[investigator.cluster] || `Cluster ${investigator.cluster}`;

        // Create label element
        const labelDiv = document.createElement('div');
        labelDiv.className = 'investigator-label';
        labelDiv.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
        labelDiv.style.color = 'white';
        labelDiv.style.padding = '8px 12px';
        labelDiv.style.borderRadius = '6px';
        labelDiv.style.fontSize = '14px';
        labelDiv.style.fontFamily = 'sans-serif';
        labelDiv.style.whiteSpace = 'nowrap';
        labelDiv.style.border = '2px solid white';
        labelDiv.innerHTML = `
            <strong>${investigator.name}</strong><br>
            <span style="font-size: 12px; opacity: 0.9;">
                ${clusterName} | Funding: $${(investigator.funding / 1000000).toFixed(2)}M
            </span>
        `;

        // Create CSS2D object
        const label = new THREE.CSS2DObject(labelDiv);

        // Position label next to the sphere (offset by sphere radius + 2 units)
        const radius = clickedSphere.geometry.parameters.radius;
        label.position.set(radius + 3, radius + 1, 0);

        // Add label to sphere so it moves with it
        clickedSphere.add(label);

        viz3D.currentLabel = label;
    }
}
