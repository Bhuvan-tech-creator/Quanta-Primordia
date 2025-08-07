#!/usr/bin/env python3
"""
Test script to verify optimized performance of the quantum traffic optimization system.
This script tests the key optimizations made to improve speed and functionality.
"""

import time
import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_traffic_matrix_creation():
    """Test the optimized traffic matrix creation"""
    print("=== Testing Optimized Traffic Matrix Creation ===")
    
    try:
        from quantum_analysis import QuantumAnalysis
        
        # Create a small test dataset
        print("Creating test dataset...")
        test_trips = pd.DataFrame({
            'PULocationID': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'DOLocationID': [2, 3, 1, 3, 1, 2, 2, 3, 1],
            'trip_distance': [2.5, 3.1, 1.8, 4.2, 2.9, 3.5, 2.1, 3.8, 2.7],
            'tpep_pickup_datetime': pd.date_range('2025-01-01', periods=9, freq='H'),
            'tpep_dropoff_datetime': pd.date_range('2025-01-01 01:00:00', periods=9, freq='H'),
            'pickup_hour': [8, 9, 10, 17, 18, 19, 7, 8, 9],
            'trip_duration': [15, 20, 12, 25, 18, 22, 14, 19, 16]
        })
        
        test_zones = pd.DataFrame({
            'LocationID': [1, 2, 3],
            'Zone': ['Test Zone 1', 'Test Zone 2', 'Test Zone 3'],
            'Borough': ['Test Borough', 'Test Borough', 'Test Borough']
        })
        
        print("Testing optimized traffic matrix creation...")
        start_time = time.time()
        
        # Test the optimized version
        quantum_analysis = QuantumAnalysis(test_trips, test_zones)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        print(f"✓ Traffic matrix created successfully in {creation_time:.2f} seconds")
        print(f"✓ Matrix shape: {quantum_analysis.traffic_matrix.shape}")
        print(f"✓ Zone mapping: {len(quantum_analysis.zone_mapping)} zones")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in traffic matrix creation test: {e}")
        return False

def test_quantum_optimization():
    """Test the optimized quantum optimization"""
    print("\n=== Testing Optimized Quantum Optimization ===")
    
    try:
        from quantum_advanced import AdvancedQuantumOptimizer
        
        print("Initializing optimized quantum optimizer...")
        start_time = time.time()
        
        # Test with reduced parameters for speed
        optimizer = AdvancedQuantumOptimizer(num_qubits=8, num_layers=3)
        
        init_time = time.time() - start_time
        print(f"✓ Quantum optimizer initialized in {init_time:.2f} seconds")
        
        # Test optimization
        print("Testing quantum optimization...")
        opt_start_time = time.time()
        
        traffic_data = {'intensity': 1.5}
        optimized_params = optimizer.optimize_traffic_routes_advanced(
            num_iterations=20,  # Reduced for testing
            traffic_data=traffic_data
        )
        
        opt_time = time.time() - opt_start_time
        print(f"✓ Quantum optimization completed in {opt_time:.2f} seconds")
        print(f"✓ Optimized parameters shape: {optimized_params.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in quantum optimization test: {e}")
        return False

def test_route_service():
    """Test the optimized route service"""
    print("\n=== Testing Optimized Route Service ===")
    
    try:
        from route_service import get_route_service
        
        print("Initializing route service...")
        start_time = time.time()
        
        route_service = get_route_service()
        
        init_time = time.time() - start_time
        print(f"✓ Route service initialized in {init_time:.2f} seconds")
        
        # Test route analysis
        print("Testing route analysis...")
        analysis_start_time = time.time()
        
        route_analysis = route_service.get_route_analysis(1, 2)
        
        analysis_time = time.time() - analysis_start_time
        print(f"✓ Route analysis completed in {analysis_time:.2f} seconds")
        
        if route_analysis:
            print(f"✓ Classical distance: {route_analysis['classical']['distance']} miles")
            print(f"✓ Quantum distance: {route_analysis['quantum']['distance']} miles")
            print(f"✓ Time savings: {route_analysis['improvements']['time_savings']}%")
            print(f"✓ Efficiency gain: {route_analysis['improvements']['efficiency_gain']}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in route service test: {e}")
        return False

def test_quantum_traffic_optimizer():
    """Test the optimized quantum traffic optimizer"""
    print("\n=== Testing Optimized Quantum Traffic Optimizer ===")
    
    try:
        from quantum_traffic_optimizer import QuantumTrafficOptimizer
        
        # Create test data files if they don't exist
        if not os.path.exists('yellow_tripdata_2025-06.parquet'):
            print("Creating test parquet file...")
            test_data = pd.DataFrame({
                'PULocationID': [1, 2, 3] * 100,
                'DOLocationID': [2, 3, 1] * 100,
                'trip_distance': np.random.uniform(1, 10, 300),
                'tpep_pickup_datetime': pd.date_range('2025-01-01', periods=300, freq='H'),
                'tpep_dropoff_datetime': pd.date_range('2025-01-01 01:00:00', periods=300, freq='H')
            })
            test_data.to_parquet('yellow_tripdata_2025-06.parquet')
        
        if not os.path.exists('taxi_zone_lookup.csv'):
            print("Creating test CSV file...")
            test_zones = pd.DataFrame({
                'LocationID': range(1, 11),
                'Zone': [f'Test Zone {i}' for i in range(1, 11)],
                'Borough': ['Test Borough'] * 10
            })
            test_zones.to_csv('taxi_zone_lookup.csv', index=False)
        
        print("Initializing quantum traffic optimizer...")
        start_time = time.time()
        
        optimizer = QuantumTrafficOptimizer(
            trip_data_path='yellow_tripdata_2025-06.parquet',
            zone_data_path='taxi_zone_lookup.csv',
            num_qubits=8  # Reduced for testing
        )
        
        init_time = time.time() - start_time
        print(f"✓ Quantum traffic optimizer initialized in {init_time:.2f} seconds")
        
        # Test optimization
        print("Testing traffic route optimization...")
        opt_start_time = time.time()
        
        optimized_params = optimizer.optimize_traffic_routes(num_iterations=30)  # Reduced for testing
        
        opt_time = time.time() - opt_start_time
        print(f"✓ Traffic route optimization completed in {opt_time:.2f} seconds")
        
        # Test getting zones
        zones = optimizer.get_zones()
        print(f"✓ Retrieved {len(zones)} zones")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in quantum traffic optimizer test: {e}")
        return False

def run_performance_tests():
    """Run all performance tests"""
    print("🚀 Starting Optimized Performance Tests")
    print("=" * 50)
    
    tests = [
        ("Traffic Matrix Creation", test_traffic_matrix_creation),
        ("Quantum Optimization", test_quantum_optimization),
        ("Route Service", test_route_service),
        ("Quantum Traffic Optimizer", test_quantum_traffic_optimizer)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optimizations working correctly!")
        return True
    else:
        print("⚠️  Some optimizations need attention.")
        return False

if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1) 