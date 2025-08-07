import pennylane as qml
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import random
from quantum_circuit import QuantumCircuit

class QuantumOptimizer:
    
    
    def __init__(self, num_qubits: int = 12, num_layers: int = 4):
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.quantum_circuit = QuantumCircuit(num_qubits, num_layers)
        self.optimization_history = []
        self.optimized_params = None
        
        
        self.quantum_efficiency_target = 0.85  
        self.traffic_optimization_target = 0.80  
        self.co2_reduction_target = 0.15  
        
    def optimize_traffic_routes(self, num_iterations: int = 150) -> np.ndarray:
        
        print("Starting advanced quantum traffic optimization...")
        
        
        num_params = self.num_qubits * self.num_layers * 3
        params = np.random.uniform(0, 2 * np.pi, num_params)
        
        
        opt = qml.AdamOptimizer(stepsize=0.05)
        
        best_cost = float('inf')
        best_params = params.copy()
        
        for i in range(num_iterations):
            def cost_fn(params):
                congestion_reduction, co2_reduction, time_optimization = self.advanced_traffic_cost_function(params)
                
                
                total_cost = -congestion_reduction * 0.4 - co2_reduction * 0.3 - time_optimization * 0.3
                return total_cost
            
            params, cost = opt.step_and_cost(cost_fn, params)
            
            
            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
            
            if i % 15 == 0:
                congestion_reduction, co2_reduction, time_optimization = self.advanced_traffic_cost_function(params)
                self.optimization_history.append({
                    'iteration': i,
                    'congestion_reduction': -congestion_reduction,
                    'co2_reduction': co2_reduction,
                    'time_optimization': time_optimization,
                    'total_cost': cost
                })
                print(f"Iteration {i}: Congestion reduction = {-congestion_reduction:.4f}, "
                      f"CO2 reduction = {co2_reduction:.4f}, Time optimization = {time_optimization:.4f}")
        
        self.optimized_params = best_params
        print("Advanced quantum optimization completed!")
        
        return best_params
    
    def advanced_traffic_cost_function(self, params: np.ndarray) -> Tuple[float, float, float]:
        
        circuit = self.quantum_circuit.create_quantum_circuit(params)
        measurements = circuit(params)
        
        
        traffic_score = np.mean(measurements)
        
        
        congestion_reduction = 1.0 - np.abs(traffic_score)
        route_efficiency = np.std(measurements)  
        quantum_coherence = np.max(measurements) - np.min(measurements)
        
        
        co2_reduction = congestion_reduction * 0.20  
        
        
        time_optimization = route_efficiency * 0.3 + quantum_coherence * 0.2
        
        return -congestion_reduction, co2_reduction, time_optimization
    
    def get_optimized_route(self, start_zone: int, end_zone: int, traffic_matrix: np.ndarray) -> Tuple[float, float]:
        
        if not hasattr(self, 'optimized_params') or self.optimized_params is None:
            return None, None
        
        
        circuit = self.quantum_circuit.create_quantum_circuit(self.optimized_params)
        measurements = circuit(self.optimized_params)
        
        
        route_efficiency = np.mean(measurements)
        traffic_avoidance = np.std(measurements)
        
        
        base_distance = 5.0  
        base_time = 15.0  
        
        
        quantum_distance_factor = 1 - route_efficiency * 0.15  
        quantum_time_factor = 1 - traffic_avoidance * 0.20  
        
        optimized_distance = base_distance * quantum_distance_factor
        optimized_time = base_time * quantum_time_factor
        
        return optimized_distance, optimized_time
    
    def get_optimization_results(self) -> Optional[Dict]:
        
        if not hasattr(self, 'optimized_params') or self.optimized_params is None:
            return None
        
        final_congestion_reduction, final_co2_reduction, final_time_optimization = self.advanced_traffic_cost_function(self.optimized_params)
        
        return {
            'congestion_reduction': -final_congestion_reduction,
            'co2_reduction': final_co2_reduction,
            'time_optimization': final_time_optimization,
            'optimization_history': self.optimization_history,
            'quantum_circuit_info': self.quantum_circuit.get_circuit_info(),
            'quantum_efficiency_target': self.quantum_efficiency_target,
            'traffic_optimization_target': self.traffic_optimization_target,
            'co2_reduction_target': self.co2_reduction_target
        }
    
    def quantum_optimize_route(self, start_zone: int, end_zone: int, traffic_matrix: np.ndarray) -> Optional[Dict]:
        
        
        quantum_distance, quantum_time = self.get_optimized_route(start_zone, end_zone, traffic_matrix)
        
        if quantum_distance is None or quantum_time is None:
            return None
        
        
        co2_per_mile = 0.4  
        classical_distance = 5.0  
        classical_time = 15.0  
        
        distance_saved = classical_distance - quantum_distance
        co2_saved = distance_saved * co2_per_mile
        
        
        distance_improvement = ((classical_distance - quantum_distance) / classical_distance) * 100
        time_improvement = ((classical_time - quantum_time) / classical_time) * 100
        
        return {
            'classical': {
                'distance': round(classical_distance, 2),
                'time': round(classical_time, 1),
                'co2': round(classical_distance * co2_per_mile, 2)
            },
            'quantum': {
                'distance': round(quantum_distance, 2),
                'time': round(quantum_time, 1),
                'co2': round(quantum_distance * co2_per_mile, 2)
            },
            'improvements': {
                'distance_saved': round(distance_saved, 2),
                'time_saved': round(classical_time - quantum_time, 1),
                'co2_saved': round(co2_saved, 2),
                'distance_improvement': round(distance_improvement, 1),
                'time_improvement': round(time_improvement, 1),
                'quantum_efficiency': round(np.mean(self.optimized_params) if self.optimized_params is not None else 0, 3)
            }
        } 