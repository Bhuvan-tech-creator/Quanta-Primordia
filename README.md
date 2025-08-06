# Quantum Traffic Optimization - NYC

A sophisticated quantum computing-based traffic optimization system for New York City taxi routes. This application uses real road data from OpenStreetMap and advanced quantum algorithms to optimize routes between different NYC zones.

## Features

- **Real Road Routing**: Uses OpenStreetMap (OSRM) for actual road network routing
- **Quantum Optimization**: Advanced quantum algorithms for route optimization
- **Interactive Web Interface**: Beautiful, responsive web UI with real-time route visualization
- **Traffic Analysis**: Comprehensive traffic pattern analysis with quantum insights
- **Multi-Objective Optimization**: Optimizes for distance, time, and CO2 emissions simultaneously

## Project Structure

```
QuantaTravel/
├── app.py                          # Main Flask application
├── route_service.py                # OpenStreetMap routing service
├── quantum_circuit.py             # Quantum circuit implementation
├── quantum_optimizer.py           # Quantum optimization algorithms
├── quantum_analysis.py            # Traffic analysis with quantum insights
├── quantum_traffic_optimizer.py   # Main quantum traffic optimizer
├── requirements.txt               # Python dependencies
├── static/                        # Static files (CSS, JS)
├── templates/                     # HTML templates
└── README.md                     # This file
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd QuantaTravel
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download required data files**
   - You'll need to obtain the NYC taxi trip data and zone lookup files
   - Place them in the project root directory:
     - `yellow_tripdata_2025-06.parquet` (or similar taxi trip data)
     - `taxi_zone_lookup.csv` (NYC taxi zone data)

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`
   - The application will be available at this address

3. **Use the application**
   - Select start and end zones from the dropdown menus
   - Click "Optimize Route" to run quantum optimization
   - View the results on the interactive map
   - Compare classical vs quantum route metrics

## Quantum Computing Components

### Quantum Circuit (`quantum_circuit.py`)
- Implements quantum circuits using PennyLane
- Features sophisticated entanglement patterns
- Multi-layer quantum operations for optimization

### Quantum Optimizer (`quantum_optimizer.py`)
- Advanced optimization algorithms
- Multi-objective cost functions
- Quantum parameter optimization

### Quantum Analysis (`quantum_analysis.py`)
- Traffic pattern analysis
- Quantum-enhanced insights
- Historical data processing

## API Endpoints

- `GET /api/zones` - Get all available zones
- `POST /api/optimize_route` - Optimize route between two zones

## Configuration

The application can be configured by modifying the following parameters:

- **Quantum Qubits**: Number of qubits for optimization (default: 12)
- **Quantum Layers**: Number of quantum circuit layers (default: 4)
- **Optimization Iterations**: Number of optimization iterations (default: 150)

## Dependencies

- **Flask**: Web framework
- **PennyLane**: Quantum computing library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Geopy**: Geocoding and distance calculations
- **Requests**: HTTP library for API calls
- **Folium**: Map visualization
- **Plotly**: Interactive plotting

## Data Requirements

The application requires:
- NYC taxi trip data (Parquet format)
- NYC taxi zone lookup data (CSV format)

These files should be placed in the project root directory.

## Performance Notes

- The quantum optimization can take several seconds to complete
- Large datasets may require significant memory
- The application uses caching to improve performance

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Data File Errors**: Ensure required data files are present
   - `yellow_tripdata_*.parquet`
   - `taxi_zone_lookup.csv`

3. **Port Already in Use**: Change the port in `app.py`
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
   ```

4. **Memory Issues**: For large datasets, consider:
   - Using a smaller subset of data
   - Increasing system memory
   - Running on a more powerful machine

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NYC Taxi and Limousine Commission for trip data
- OpenStreetMap contributors for road data
- PennyLane team for quantum computing framework
- Flask team for the web framework 