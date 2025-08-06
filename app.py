from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
from quantum_traffic_optimizer import QuantumTrafficOptimizer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from geopy.geocoders import Nominatim
import os
from route_service import get_route_service

app = Flask(__name__)
CORS(app)

# Initialize the quantum traffic optimizer
optimizer = None
zone_data = None

def load_zone_data():
    """Load zone data from CSV"""
    global zone_data
    if zone_data is None:
        zone_data = pd.read_csv('taxi_zone_lookup.csv')

def initialize_optimizer():
    """Initialize the quantum traffic optimizer with advanced algorithms"""
    global optimizer
    if optimizer is None:
        print("Quantum optimizer initialization skipped for faster startup")
        # Uncomment the following lines for full quantum optimization
        # print("Initializing advanced quantum traffic optimizer...")
        # try:
        #     optimizer = QuantumTrafficOptimizer(
        #         trip_data_path='yellow_tripdata_2025-06.parquet',
        #         zone_data_path='taxi_zone_lookup.csv',
        #         num_qubits=12  # Increased for better optimization
        #     )
        #     # Run advanced quantum optimization
        #     print("Running quantum optimization...")
        #     optimizer.optimize_traffic_routes(num_iterations=100)
        #     print("Quantum optimization completed!")
        # except Exception as e:
        #     print(f"Error initializing quantum optimizer: {e}")
        #     optimizer = None
        optimizer = None

@app.route('/')
def index():
    """Main page with interactive map and controls"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Test page for dropdown functionality"""
    return render_template('test_dropdown.html')

@app.route('/api/zones', methods=['GET'])
def get_zones():
    """Get all available zones for dropdown selection"""
    try:
        if zone_data is None:
            load_zone_data()
        
        zones = []
        for _, zone in zone_data.iterrows():
            # Handle NaN values
            borough = zone['Borough'] if pd.notna(zone['Borough']) else 'Unknown'
            zone_name = zone['Zone'] if pd.notna(zone['Zone']) else 'Unknown Zone'
            
            zones.append({
                'LocationID': int(zone['LocationID']),
                'Zone': zone_name,
                'Borough': borough
            })
        
        return jsonify({'status': 'success', 'zones': zones})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/optimize_route', methods=['POST'])
def optimize_route():
    """Optimize route between two zones using real road routing and quantum optimization"""
    try:
        data = request.get_json()
        start_zone = int(data['start_zone'])
        end_zone = int(data['end_zone'])
        
        # Get zone information
        if zone_data is None:
            load_zone_data()
        
        start_zone_info = zone_data[zone_data['LocationID'] == start_zone].iloc[0]
        end_zone_info = zone_data[zone_data['LocationID'] == end_zone].iloc[0]
        
        # Initialize quantum optimizer if not already done
        if optimizer is None:
            initialize_optimizer()
        
        # Get route analysis using real road routing and quantum optimization
        route_service = get_route_service()
        route_analysis = route_service.get_route_analysis(start_zone, end_zone)
        
        if route_analysis is None:
            return jsonify({'status': 'error', 'message': 'Unable to calculate route'})
        
        result = {
            'start_zone': start_zone_info['Zone'],
            'end_zone': end_zone_info['Zone'],
            'start_borough': start_zone_info['Borough'],
            'end_borough': end_zone_info['Borough'],
            'classical': route_analysis['classical'],
            'quantum': route_analysis['quantum'],
            'improvements': route_analysis['improvements']
        }
        
        return jsonify({
            'status': 'success',
            'route': result,
            'coordinates': route_analysis['coordinates']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("Starting Advanced Quantum Traffic Optimization Web Application...")
    print("Initializing quantum optimizer...")
    initialize_optimizer()
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 