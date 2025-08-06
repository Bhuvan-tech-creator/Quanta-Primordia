// Global variables
let map;
let zones = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing application...');
    
    // Add a small delay to ensure all elements are ready
    setTimeout(() => {
        initializeMap();
        setupEventListeners();
        loadZones();
    }, 100);
});

// Initialize the map
function initializeMap() {
    console.log('Initializing map...');
    // Initialize map centered on NYC
    map = L.map('map').setView([40.7128, -74.0060], 11);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Add NYC borough boundaries (simplified)
    const nycBounds = [
        [40.4774, -74.2591], // Southwest
        [40.9176, -73.7004]  // Northeast
    ];
    map.fitBounds(nycBounds);
    console.log('Map initialized successfully');
}

// Setup event listeners
function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    const optimizeBtn = document.getElementById('optimizeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const startZone = document.getElementById('startZone');
    const endZone = document.getElementById('endZone');
    
    if (optimizeBtn) {
        optimizeBtn.addEventListener('click', optimizeRoute);
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetPage);
    }
    
    if (startZone) {
        startZone.addEventListener('change', validateRouteSelection);
    }
    
    if (endZone) {
        endZone.addEventListener('change', validateRouteSelection);
    }
    
    console.log('Event listeners set up successfully');
}

// Load zones from API
async function loadZones() {
    console.log('Loading zones from API...');
    try {
        const response = await fetch('/api/zones');
        console.log('API response received:', response);
        console.log('Response status:', response.status);
        
        const data = await response.json();
        console.log('Parsed response data:', data);
        
        if (data.status === 'success') {
            zones = data.zones;
            console.log('Zones loaded successfully:', zones.length, 'zones');
            if (zones.length > 0) {
                console.log('First zone:', zones[0]);
                console.log('Last zone:', zones[zones.length - 1]);
            }
            populateZoneDropdowns();
        } else {
            console.error('Error in API response:', data.message);
            showAlert('Error loading zones: ' + data.message, 'danger');
        }
    } catch (error) {
        console.error('Error loading zones:', error);
        showAlert('Error loading zones: ' + error.message, 'danger');
    }
}

// Populate zone dropdowns
function populateZoneDropdowns() {
    console.log('Populating zone dropdowns...');
    const startSelect = document.getElementById('startZone');
    const endSelect = document.getElementById('endZone');
    
    console.log('Start select element:', startSelect);
    console.log('End select element:', endSelect);
    
    if (!startSelect || !endSelect) {
        console.error('Dropdown elements not found!');
        // Try again after a short delay
        setTimeout(populateZoneDropdowns, 500);
        return;
    }
    
    // Clear existing options
    startSelect.innerHTML = '<option value="">Select start zone...</option>';
    endSelect.innerHTML = '<option value="">Select end zone...</option>';
    
    console.log('Adding', zones.length, 'zones to dropdowns');
    
    // Add zone options
    zones.forEach((zone, index) => {
        const option = document.createElement('option');
        option.value = zone.LocationID;
        option.textContent = `${zone.Zone} (${zone.Borough})`;
        
        startSelect.appendChild(option.cloneNode(true));
        endSelect.appendChild(option);
        
        if (index < 3) { // Log first 3 zones for debugging
            console.log(`Added zone ${index + 1}:`, zone);
        }
    });
    
    console.log('Dropdowns populated successfully');
    console.log('Start dropdown options:', startSelect.options.length);
    console.log('End dropdown options:', endSelect.options.length);
    
    // Force a reflow to ensure the dropdowns are updated
    startSelect.style.display = 'none';
    startSelect.offsetHeight; // Trigger reflow
    startSelect.style.display = '';
    
    endSelect.style.display = 'none';
    endSelect.offsetHeight; // Trigger reflow
    endSelect.style.display = '';
}

// Validate route selection
function validateRouteSelection() {
    const startZone = document.getElementById('startZone').value;
    const endZone = document.getElementById('endZone').value;
    const optimizeBtn = document.getElementById('optimizeBtn');
    
    optimizeBtn.disabled = !startZone || !endZone;
}

// Optimize route
async function optimizeRoute() {
    const startZone = document.getElementById('startZone').value;
    const endZone = document.getElementById('endZone').value;
    
    if (!startZone || !endZone) {
        showAlert('Please select both start and end zones', 'warning');
        return;
    }
    
    if (startZone === endZone) {
        showAlert('Start and end zones cannot be the same', 'warning');
        return;
    }
    
    const optimizeBtn = document.getElementById('optimizeBtn');
    const routeLoading = document.getElementById('routeLoading');
    
    // Show loading with progress bar
    optimizeBtn.disabled = true;
    routeLoading.style.display = 'block';
    
    const { progressBar, interval } = showLoadingBar();
    
    try {
        const response = await fetch('/api/optimize_route', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                start_zone: parseInt(startZone),
                end_zone: parseInt(endZone)
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Complete progress bar
            const progressBarInner = progressBar.querySelector('.progress-bar');
            progressBarInner.style.width = '100%';
            progressBarInner.textContent = 'Quantum Optimization Complete!';
            
            setTimeout(() => {
                // Display results
                displayRouteResults(data.route);
                showRouteOnMap(startZone, endZone, data.coordinates);
                showAlert('Advanced quantum route optimization completed successfully!', 'success');
                hideLoadingBar(progressBar, interval);
            }, 500);
        } else {
            showAlert('Error: ' + data.message, 'danger');
            hideLoadingBar(progressBar, interval);
        }
    } catch (error) {
        showAlert('Error connecting to server: ' + error.message, 'danger');
        hideLoadingBar(progressBar, interval);
    } finally {
        optimizeBtn.disabled = false;
    }
}

// Display route results
function displayRouteResults(route) {
    // Classical route
    document.getElementById('classicalDistance').textContent = route.classical.distance;
    document.getElementById('classicalTime').textContent = route.classical.time;
    document.getElementById('classicalCO2').textContent = route.classical.co2;
    
    // Quantum route
    document.getElementById('quantumDistance').textContent = route.quantum.distance;
    document.getElementById('quantumTime').textContent = route.quantum.time;
    document.getElementById('quantumCO2').textContent = route.quantum.co2;
    
    // Improvements
    document.getElementById('distanceImprovement').textContent = route.improvements.distance_improvement;
    document.getElementById('timeImprovement').textContent = route.improvements.time_improvement;
    document.getElementById('co2Saved').textContent = route.improvements.co2_saved;
    
    // Add quantum efficiency display if available
    if (route.improvements.efficiency_gain) {
        const efficiencyElement = document.getElementById('quantumEfficiency');
        if (efficiencyElement) {
            efficiencyElement.textContent = route.improvements.efficiency_gain.toFixed(1) + '%';
        }
    }
    
    document.getElementById('routeResults').style.display = 'block';
}

// Show route on map
function showRouteOnMap(startZone, endZone, coordinates) {
    // Clear existing markers and routes
    map.eachLayer((layer) => {
        if (layer instanceof L.Marker || layer instanceof L.Polyline) {
            map.removeLayer(layer);
        }
    });
    
    const startZoneInfo = zones.find(z => z.LocationID == startZone);
    const endZoneInfo = zones.find(z => z.LocationID == endZone);
    
    // Create GPS icon for start and end markers
    const gpsIcon = L.divIcon({
        html: '<i class="fas fa-map-marker-alt" style="color: #e74c3c; font-size: 24px;"></i>',
        className: 'gps-marker',
        iconSize: [24, 24],
        iconAnchor: [12, 24]
    });
    
    // Add GPS markers
    const startMarker = L.marker(coordinates.start, {icon: gpsIcon}).addTo(map);
    startMarker.bindPopup(`<b>Start:</b> ${startZoneInfo.Zone}<br><b>Borough:</b> ${startZoneInfo.Borough}`);
    
    const endMarker = L.marker(coordinates.end, {icon: gpsIcon}).addTo(map);
    endMarker.bindPopup(`<b>End:</b> ${endZoneInfo.Zone}<br><b>Borough:</b> ${endZoneInfo.Borough}`);
    
    // Add classical route (real road route) - PURPLE DASHED
    if (coordinates.classical_route && coordinates.classical_route.length > 0) {
        const classicalRoute = L.polyline(coordinates.classical_route, {
            color: '#9b59b6',
            weight: 6,
            opacity: 0.9,
            dashArray: '15, 10'
        }).addTo(map);
        classicalRoute.bindPopup('Classical Route (Real Road Network)');
    }
    
    // Add quantum route (optimized path) - GREEN SOLID
    if (coordinates.quantum_route && coordinates.quantum_route.length > 0) {
        const quantumRoute = L.polyline(coordinates.quantum_route, {
            color: '#27ae60',
            weight: 8,
            opacity: 0.9
        }).addTo(map);
        quantumRoute.bindPopup('Quantum Optimized Route (Advanced Algorithm)');
    }
    
    // Fit map to show both markers
    map.fitBounds([coordinates.start, coordinates.end]);
}

// Add loading bar functionality
function showLoadingBar() {
    const loadingDiv = document.getElementById('routeLoading');
    const progressBar = document.createElement('div');
    progressBar.className = 'progress';
    progressBar.style.height = '20px';
    progressBar.style.marginTop = '10px';
    
    const progressBarInner = document.createElement('div');
    progressBarInner.className = 'progress-bar progress-bar-striped progress-bar-animated';
    progressBarInner.style.width = '0%';
    progressBarInner.textContent = 'Quantum Optimization in Progress...';
    
    progressBar.appendChild(progressBarInner);
    loadingDiv.appendChild(progressBar);
    
    // Animate progress bar with quantum-themed messages
    let progress = 0;
    const quantumMessages = [
        'Initializing quantum circuits...',
        'Applying quantum superposition...',
        'Measuring quantum states...',
        'Optimizing route variations...',
        'Calculating quantum efficiency...',
        'Finalizing optimization...'
    ];
    let messageIndex = 0;
    
    const interval = setInterval(() => {
        progress += Math.random() * 12;
        if (progress > 90) progress = 90;
        progressBarInner.style.width = progress + '%';
        
        // Update message every 15%
        if (progress > messageIndex * 15 && messageIndex < quantumMessages.length - 1) {
            messageIndex++;
            progressBarInner.textContent = quantumMessages[messageIndex];
        }
    }, 300);
    
    return { progressBar, interval };
}

function hideLoadingBar(progressBar, interval) {
    clearInterval(interval);
    if (progressBar) {
        progressBar.remove();
    }
    document.getElementById('routeLoading').style.display = 'none';
}

// Show alert message
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the main container
    const mainContainer = document.querySelector('.main-container');
    mainContainer.insertBefore(alertDiv, mainContainer.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Reset page function
function resetPage() {
    // Clear dropdowns
    document.getElementById('startZone').value = '';
    document.getElementById('endZone').value = '';
    
    // Disable optimize button
    document.getElementById('optimizeBtn').disabled = true;
    
    // Hide route results
    document.getElementById('routeResults').style.display = 'none';
    
    // Clear map
    map.eachLayer((layer) => {
        if (layer instanceof L.Marker || layer instanceof L.Polyline) {
            map.removeLayer(layer);
        }
    });
    
    // Reset map view to NYC
    map.setView([40.7128, -74.0060], 11);
    
    // Hide loading
    document.getElementById('routeLoading').style.display = 'none';
    
    showAlert('Page reset successfully!', 'success');
} 