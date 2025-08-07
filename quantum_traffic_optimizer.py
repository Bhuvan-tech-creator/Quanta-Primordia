import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
from quantum_analysis import QuantumAnalysis
from quantum_optimizer import QuantumOptimizer

warnings.filterwarnings('ignore')

class QuantumTrafficOptimizer:
    def __init__(self, trip_data_path: str, zone_data_path: str, num_qubits: int = 12):
        
        self.num_qubits = num_qubits
        self.trip_data = None
        self.zone_data = None
        self.quantum_analysis = None
        self.quantum_optimizer = None
        
        
        self._load_data(trip_data_path, zone_data_path)
        self._initialize_quantum_components()
        
    def _load_data(self, trip_data_path: str, zone_data_path: str):
        
        print("Loading trip data...")
        self.trip_data = pd.read_parquet(trip_data_path)
        
        print("Loading zone data...")
        self.zone_data = pd.read_csv(zone_data_path)
        
        print(f"Loaded {len(self.trip_data)} trips and {len(self.zone_data)} zones")
        
    def _initialize_quantum_components(self):
        
        print("Initializing quantum components...")
        
        
        self.quantum_analysis = QuantumAnalysis(self.trip_data, self.zone_data)
        
        
        self.quantum_optimizer = QuantumOptimizer(self.num_qubits, 4)  
        
        print("Quantum components initialized successfully")
        
    def optimize_traffic_routes(self, num_iterations: int = 150) -> np.ndarray:
        
        print("Starting advanced quantum traffic optimization...")
        
        
        optimized_params = self.quantum_optimizer.optimize_traffic_routes(num_iterations)
        
        print("Advanced quantum optimization completed!")
        
        return optimized_params
    
    def get_optimized_route(self, start_zone: int, end_zone: int) -> Tuple[float, float]:
        
        if start_zone not in self.quantum_analysis.zone_mapping or end_zone not in self.quantum_analysis.zone_mapping:
            return None, None
        
        
        quantum_distance, quantum_time = self.quantum_optimizer.get_optimized_route(
            start_zone, end_zone, self.quantum_analysis.traffic_matrix
        )
        
        return quantum_distance, quantum_time
    
    def get_traffic_analysis(self) -> Dict:
        
        return self.quantum_analysis.get_traffic_analysis()
    
    def get_optimization_results(self) -> Optional[Dict]:
        
        return self.quantum_optimizer.get_optimization_results()
    
    def get_zones(self) -> List[Dict]:
        
        return self.quantum_analysis.get_zones()
    
    def quantum_optimize_route(self, start_zone: int, end_zone: int) -> Optional[Dict]:
        
        return self.quantum_analysis.get_quantum_route_optimization(start_zone, end_zone)

if __name__ == "__main__":
    
    optimizer = QuantumTrafficOptimizer(
        trip_data_path='yellow_tripdata_2025-06.parquet',
        zone_data_path='taxi_zone_lookup.csv',
        num_qubits=12
    )
    
    
    optimizer.optimize_traffic_routes(num_iterations=100)
    
    
    traffic_analysis = optimizer.get_traffic_analysis()
    optimization_results = optimizer.get_optimization_results()
    
    print("Traffic Analysis:", traffic_analysis)
    print("Optimization Results:", optimization_results) 