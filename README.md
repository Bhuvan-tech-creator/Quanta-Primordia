# Quantum Traffic Optimization - NYC

This project explores the use of quantum computing to optimize traffic routes in New York City. It leverages real-world data, advanced algorithms, and quantum-inspired techniques to provide efficient and eco-friendly route suggestions.

## Features

-   **Interactive Map:** Visualizes traffic patterns and optimized routes on an interactive map of NYC.
-   **Quantum-Enhanced Route Optimization:** Utilizes quantum computing principles to find the most efficient routes, considering both time and distance.
-   **Data Integration:** Incorporates real-time traffic data to provide accurate and up-to-date route analysis.
-   **User-Friendly Interface:** Allows users to select start and end zones and view detailed route comparisons.
-   **Detailed Route Analysis:** Provides comprehensive information on distance, time, and CO2 emissions for classical and quantum-optimized routes.
-   **Performance Improvements:** Includes optimized code for faster route calculation and analysis.

## Technologies Used

-   **Frontend:**
    -   HTML, CSS, JavaScript
    -   Leaflet for interactive maps
    -   Bootstrap for responsive design
    -   Font Awesome for icons
-   **Backend:**
    -   Python
    -   Flask for web framework
    -   pandas for data manipulation
    -   NumPy for numerical computations
    -   SciKit-learn for machine learning
    -   Plotly for data visualization
    -   geopy for geolocation
    -   OSRM for road routing

## External Items Needed
    - A .parquet file of traffic data from NYC TLC Trip Record Data
    - A .csv file of Traffic Lookup Zone From NYC TLC Trip Record Data

## Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd QuantaTravel
    ```

2. **Create Virtual Environment**

    Creae a virtual environemtn in a version before python 3.12
   
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    -   Download the NYC taxi trip data (e.g., `yellow_tripdata_2025-06.parquet`) and place it in the project root directory.
    -   Ensure the `taxi_zone_lookup.csv` file is also in the project root directory.

5.  **Run the application:**

    ```bash
    python app.py
    ```

6.  **Access the application:**

    Open your web browser and navigate to `http://localhost:5000`.

## Key Components

### `app.py`

-   The main Flask application that handles routing and API endpoints.
-   Initializes the [`QuantumTrafficOptimizer`](quantum_traffic_optimizer.py) and loads zone data.
-   Defines routes for the main page (`/`), testing (`/test`), zone data (`/api/zones`), and route optimization (`/api/optimize_route`).

### `data_processor.py`

-   Handles loading, preprocessing, and analyzing NYC taxi data.
-   The [`NYCDataProcessor`](data_processor.py) class performs tasks such as:
    -   Loading trip and zone data.
    -   Preprocessing trip data (converting data types, adding time-based features, filtering invalid trips).
    -   Creating traffic patterns and congestion analysis.
    -   Calculating zone statistics and optimization scenarios.

### `quantum_analysis.py`

-   Provides quantum-enhanced traffic analysis and insights.
-   The [`QuantumAnalysis`](quantum_analysis.py) class performs tasks such as:
    -   Preprocessing data.
    -   Creating a traffic matrix using FAST vectorized operations (`_create_traffic_matrix_fast`).
    -   Calculating traffic hotspots and time-based traffic patterns.
    -   Running quantum optimization on traffic data.
    -   Providing quantum-optimized routes between two zones.

### `quantum_traffic_optimizer.py`

-   The main quantum traffic optimizer that orchestrates the quantum optimization process.
-   The [`QuantumTrafficOptimizer`](quantum_traffic_optimizer.py) class manages:
    -   Loading and preparing taxi trip and zone data.
    -   Initializing quantum analysis and optimizer components.
    -   Optimizing traffic routes using advanced quantum computing.
    -   Providing optimized routes between two zones using quantum results.
    -   Delivering comprehensive traffic analysis with quantum insights.

### `route_service.py`

-   Provides advanced route services with real road routing and quantum optimization.
-   The [`RouteService`](route_service.py) class offers functionalities such as:
    -   Generating realistic routes between two zones using real road data.
    -   Generating traffic data using real NYC taxi trip data.
    -   Calculating route analysis using real road data and quantum optimization.

### `static/js/app.js`

-   The main JavaScript file for the frontend application.
-   Handles:
    -   Initializing the map using Leaflet.
    -   Setting up event listeners for user interactions.
    -   Loading zone data from the API and populating dropdowns.
    -   Validating route selections.
    -   Optimizing routes by sending requests to the backend API.
    -   Displaying route results and visualizing routes on the map.

# Thank You So Much!
# Hope This Helps!
