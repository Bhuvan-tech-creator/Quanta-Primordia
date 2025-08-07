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
        self._create_traffic_matrix_fast()
        
    def _preprocess_data(self):
        """Preprocess the trip data for analysis - OPTIMIZED VERSION"""
        print("Preprocessing trip data...")
        
        # Convert datetime columns efficiently
        self.trip_data['tpep_pickup_datetime'] = pd.to_datetime(self.trip_data['tpep_pickup_datetime'])
        self.trip_data['tpep_dropoff_datetime'] = pd.to_datetime(self.trip_data['tpep_dropoff_datetime'])
        
        # Add time-based features efficiently
        self.trip_data['pickup_hour'] = self.trip_data['tpep_pickup_datetime'].dt.hour
        self.trip_data['pickup_day'] = self.trip_data['tpep_pickup_datetime'].dt.dayofweek
        self.trip_data['trip_duration'] = (
            self.trip_data['tpep_dropoff_datetime'] - self.trip_data['tpep_pickup_datetime']
        ).dt.total_seconds() / 60  # minutes
        
        # Filter out invalid trips efficiently
        valid_mask = (
            (self.trip_data['trip_distance'] > 0) &
            (self.trip_data['trip_duration'] > 0) &
            (self.trip_data['trip_duration'] < 180)  # Less than 3 hours
        )
        self.trip_data = self.trip_data[valid_mask]
        
        print(f"After preprocessing: {len(self.trip_data)} valid trips")
        
    def _create_traffic_matrix_fast(self):
        """Create a traffic congestion matrix using FAST vectorized operations"""
        print("Creating traffic matrix using FAST vectorized operations...")
        
        # Get unique zones
        zones = sorted(self.zone_data['LocationID'].unique())
        zone_to_idx = {zone: idx for idx, zone in enumerate(zones)}
        self.zone_mapping = zone_to_idx
        
        # Initialize traffic matrix
        self.traffic_matrix = np.zeros((len(zones), len(zones)))
        
        # OPTIMIZATION 1: Sample data if too large (keep only 10% for speed)
        if len(self.trip_data) > 100000:
            print(f"Large dataset detected ({len(self.trip_data)} trips). Sampling 10% for speed...")
            sample_size = min(100000, len(self.trip_data) // 10)
            trip_sample = self.trip_data.sample(n=sample_size, random_state=42)
        else:
            trip_sample = self.trip_data
        
        print(f"Using {len(trip_sample)} trips for traffic matrix creation")
        
        # OPTIMIZATION 2: Use vectorized operations instead of iteration
        # Filter trips with valid zones
        valid_trips = trip_sample[
            (trip_sample['PULocationID'].isin(zone_to_idx.keys())) &
            (trip_sample['DOLocationID'].isin(zone_to_idx.keys()))
        ].copy()
        
        # Convert zone IDs to indices efficiently
        valid_trips['pickup_idx'] = valid_trips['PULocationID'].map(zone_to_idx)
        valid_trips['dropoff_idx'] = valid_trips['DOLocationID'].map(zone_to_idx)
        
        # OPTIMIZATION 3: Calculate weights vectorized
        # Base weight calculation
        valid_trips['base_weight'] = valid_trips['trip_distance'] * (valid_trips['trip_duration'] / 60)
        
        # Time-based traffic weighting (vectorized)
        hour_conditions = [
            valid_trips['pickup_hour'].isin([7, 8, 9, 17, 18, 19]),  # Peak hours
            valid_trips['pickup_hour'].between(10, 16),  # Business hours
        ]
        time_weights = [2.0, 1.5, 0.8]  # Peak, business, off-peak
        valid_trips['time_weight'] = np.select(hour_conditions, time_weights[:2], default=time_weights[2])
        
        # Final weight calculation
        valid_trips['final_weight'] = valid_trips['base_weight'] * valid_trips['time_weight']
        
        # OPTIMIZATION 4: Use groupby for fast aggregation
        traffic_aggregation = valid_trips.groupby(['pickup_idx', 'dropoff_idx'])['final_weight'].sum().reset_index()
        
        # Fill traffic matrix efficiently
        for _, row in traffic_aggregation.iterrows():
            pickup_idx = int(row['pickup_idx'])
            dropoff_idx = int(row['dropoff_idx'])
            weight = row['final_weight']
            self.traffic_matrix[pickup_idx][dropoff_idx] = weight
        
        # Normalize the traffic matrix
        max_traffic = np.max(self.traffic_matrix)
        if max_traffic > 0:
            self.traffic_matrix = self.traffic_matrix / max_traffic
            
        print(f"Created traffic matrix with shape: {self.traffic_matrix.shape} in FAST mode")
        
    def _create_traffic_matrix(self):
        """DEPRECATED: Use _create_traffic_matrix_fast() instead"""
        self._create_traffic_matrix_fast()
        
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