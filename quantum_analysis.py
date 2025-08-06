import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from quantum_optimizer import QuantumOptimizer

class QuantumAnalysis:
    """Quantum-enhanced traffic analysis and insights"""
    
    def __init__(self, trip_data: pd.DataFrame, zone_data: pd.DataFrame):
        """
        Initialize quantum traffic analysis
        
        Args:
            trip_data: Trip data DataFrame
            zone_data: Zone data DataFrame
        """
        self.trip_data = trip_data
        self.zone_data = zone_data
        self.traffic_matrix = None
        self.zone_mapping = None
        self.quantum_optimizer = None
        
        # Preprocess data
        self._preprocess_data()
        self._create_traffic_matrix()
        
    def _preprocess_data(self):
        """Preprocess the trip data for analysis"""
        # Convert datetime columns
        self.trip_data['tpep_pickup_datetime'] = pd.to_datetime(self.trip_data['tpep_pickup_datetime'])
        self.trip_data['tpep_dropoff_datetime'] = pd.to_datetime(self.trip_data['tpep_dropoff_datetime'])
        
        # Add time-based features
        self.trip_data['pickup_hour'] = self.trip_data['tpep_pickup_datetime'].dt.hour
        self.trip_data['pickup_day'] = self.trip_data['tpep_pickup_datetime'].dt.dayofweek
        self.trip_data['trip_duration'] = (
            self.trip_data['tpep_dropoff_datetime'] - self.trip_data['tpep_pickup_datetime']
        ).dt.total_seconds() / 60  # minutes
        
        # Filter out invalid trips
        self.trip_data = self.trip_data[
            (self.trip_data['trip_distance'] > 0) &
            (self.trip_data['trip_duration'] > 0) &
            (self.trip_data['trip_duration'] < 180)  # Less than 3 hours
        ]
        
        print(f"After preprocessing: {len(self.trip_data)} valid trips")
        
    def _create_traffic_matrix(self):
        """Create a traffic congestion matrix based on trip data"""
        print("Creating traffic matrix...")
        
        # Get unique zones
        zones = sorted(self.zone_data['LocationID'].unique())
        zone_to_idx = {zone: idx for idx, zone in enumerate(zones)}
        self.zone_mapping = zone_to_idx
        
        # Initialize traffic matrix
        self.traffic_matrix = np.zeros((len(zones), len(zones)))
        
        # Calculate traffic between zones with time-based weighting
        for _, trip in self.trip_data.iterrows():
            pickup_zone = trip['PULocationID']
            dropoff_zone = trip['DOLocationID']
            
            if pickup_zone in zone_to_idx and dropoff_zone in zone_to_idx:
                pickup_idx = zone_to_idx[pickup_zone]
                dropoff_idx = zone_to_idx[dropoff_zone]
                
                # Enhanced weighting based on distance, duration, and time of day
                base_weight = trip['trip_distance'] * (trip['trip_duration'] / 60)
                
                # Time-based traffic weighting
                hour = trip['pickup_hour']
                if hour in [7, 8, 9, 17, 18, 19]:  # Peak hours
                    time_weight = 2.0
                elif 10 <= hour <= 16:  # Business hours
                    time_weight = 1.5
                else:  # Off-peak hours
                    time_weight = 0.8
                
                final_weight = base_weight * time_weight
                self.traffic_matrix[pickup_idx][dropoff_idx] += final_weight
        
        # Normalize the traffic matrix
        max_traffic = np.max(self.traffic_matrix)
        if max_traffic > 0:
            self.traffic_matrix = self.traffic_matrix / max_traffic
            
        print(f"Created traffic matrix with shape: {self.traffic_matrix.shape}")
        
    def initialize_quantum_optimizer(self, num_qubits: int = 12, num_layers: int = 4):
        """Initialize the quantum optimizer"""
        self.quantum_optimizer = QuantumOptimizer(num_qubits, num_layers)
        
    def get_traffic_analysis(self) -> Dict:
        """Get comprehensive traffic analysis with quantum insights"""
        # Calculate traffic hotspots using quantum-enhanced clustering
        zone_traffic = np.sum(self.traffic_matrix, axis=1)
        hotspot_zones = np.argsort(zone_traffic)[-10:]  # Top 10 congested zones
        
        # Calculate time-based traffic patterns with quantum weighting
        hourly_traffic = self.trip_data.groupby('pickup_hour').size()
        
        # Calculate borough-wise traffic distribution
        zone_traffic_df = pd.DataFrame({
            'LocationID': list(self.zone_mapping.keys()),
            'TrafficVolume': zone_traffic
        })
        zone_traffic_df = zone_traffic_df.merge(self.zone_data, on='LocationID')
        borough_traffic = zone_traffic_df.groupby('Borough')['TrafficVolume'].sum()
        
        # Quantum-enhanced analysis
        quantum_insights = self._calculate_quantum_insights()
        
        return {
            'hotspot_zones': hotspot_zones.tolist(),
            'hourly_traffic': hourly_traffic.to_dict(),
            'borough_traffic': borough_traffic.to_dict(),
            'total_trips': len(self.trip_data),
            'avg_trip_distance': self.trip_data['trip_distance'].mean(),
            'avg_trip_duration': self.trip_data['trip_duration'].mean(),
            'quantum_insights': quantum_insights
        }
    
    def _calculate_quantum_insights(self) -> Dict:
        """Calculate quantum-enhanced traffic insights"""
        if self.quantum_optimizer is None:
            return {}
        
        circuit_info = self.quantum_optimizer.quantum_circuit.get_circuit_info()
        
        return {
            'quantum_circuit_depth': circuit_info['circuit_depth'],
            'entanglement_patterns': circuit_info['entanglement_patterns'],
            'num_qubits': circuit_info['num_qubits'],
            'num_layers': circuit_info['num_layers']
        }
    
    def get_optimization_results(self) -> Optional[Dict]:
        """Get quantum optimization results"""
        if self.quantum_optimizer is None:
            return None
        
        return self.quantum_optimizer.get_optimization_results()
    
    def get_zones(self) -> List[Dict]:
        """Get all available zones for dropdown selection"""
        zones = []
        for _, zone in self.zone_data.iterrows():
            zones.append({
                'LocationID': int(zone['LocationID']),
                'Zone': str(zone['Zone']),
                'Borough': str(zone['Borough'])
            })
        return zones
    
    def run_quantum_optimization(self, num_iterations: int = 100) -> np.ndarray:
        """Run quantum optimization on traffic data"""
        if self.quantum_optimizer is None:
            self.initialize_quantum_optimizer()
        
        return self.quantum_optimizer.optimize_traffic_routes(num_iterations)
    
    def get_quantum_route_optimization(self, start_zone: int, end_zone: int) -> Optional[Dict]:
        """Get quantum-optimized route between two zones"""
        if self.quantum_optimizer is None:
            return None
        
        return self.quantum_optimizer.quantum_optimize_route(start_zone, end_zone, self.traffic_matrix) 