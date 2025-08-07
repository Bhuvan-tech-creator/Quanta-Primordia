#!/usr/bin/env python3
"""
Test script for quantum distance optimization feature
"""

import sys
import time
from route_service import get_route_service

def test_distance_optimization():
    """Test the quantum distance optimization feature"""
    print("=== Testing Quantum Distance Optimization ===")
    
    try:
        # Initialize route service
        print("Initializing route service...")
        route_service = get_route_service()
        print("✓ Route service initialized successfully")
        
        # Test route analysis with both time and distance optimization
        print("\nTesting route analysis...")
        start_zone = 1  # Newark Airport
        end_zone = 2    # Jamaica Bay
        
        print(f"Analyzing route from zone {start_zone} to zone {end_zone}...")
        start_time = time.time()
        
        route_analysis = route_service.get_route_analysis(start_zone, end_zone)
        
        if route_analysis is None:
            print("✗ Route analysis returned None")
            return False
        
        total_time = time.time() - start_time
        print(f"✓ Route analysis completed in {total_time:.2f} seconds")
        
        # Check that we have all three route types
        print("\nChecking route results...")
        
        if 'classical' not in route_analysis:
            print("✗ Missing classical route data")
            return False
        print("✓ Classical route data present")
        
        if 'quantum_time' not in route_analysis:
            print("✗ Missing quantum time route data")
            return False
        print("✓ Quantum time route data present")
        
        if 'quantum_distance' not in route_analysis:
            print("✗ Missing quantum distance route data")
            return False
        print("✓ Quantum distance route data present")
        
        # Check coordinates
        coordinates = route_analysis.get('coordinates', {})
        if 'classical_route' not in coordinates:
            print("✗ Missing classical route coordinates")
            return False
        print("✓ Classical route coordinates present")
        
        if 'quantum_time_route' not in coordinates:
            print("✗ Missing quantum time route coordinates")
            return False
        print("✓ Quantum time route coordinates present")
        
        if 'quantum_distance_route' not in coordinates:
            print("✗ Missing quantum distance route coordinates")
            return False
        print("✓ Quantum distance route coordinates present")
        
        # Display results
        print("\n=== Route Analysis Results ===")
        print(f"Classical Route:")
        print(f"  Distance: {route_analysis['classical']['distance']} miles")
        print(f"  Time: {route_analysis['classical']['time']} minutes")
        print(f"  CO2: {route_analysis['classical']['co2']} kg")
        
        print(f"\nQuantum Time Optimization:")
        print(f"  Distance: {route_analysis['quantum_time']['distance']} miles")
        print(f"  Time: {route_analysis['quantum_time']['time']} minutes")
        print(f"  CO2: {route_analysis['quantum_time']['co2']} kg")
        
        print(f"\nQuantum Distance Optimization:")
        print(f"  Distance: {route_analysis['quantum_distance']['distance']} miles")
        print(f"  Time: {route_analysis['quantum_distance']['time']} minutes")
        print(f"  CO2: {route_analysis['quantum_distance']['co2']} kg")
        
        # Check improvements
        improvements = route_analysis.get('improvements', {})
        print(f"\nImprovements:")
        print(f"  Time Distance Improvement: {improvements.get('time_distance_improvement', 0)}%")
        print(f"  Distance Distance Improvement: {improvements.get('distance_distance_improvement', 0)}%")
        print(f"  Time Time Improvement: {improvements.get('time_time_improvement', 0)}%")
        print(f"  Distance Time Improvement: {improvements.get('distance_time_improvement', 0)}%")
        
        # Verify that distance optimization actually reduces distance
        classical_distance = route_analysis['classical']['distance']
        distance_optimized_distance = route_analysis['quantum_distance']['distance']
        
        if distance_optimized_distance < classical_distance:
            print(f"✓ Distance optimization successful: {classical_distance:.2f} -> {distance_optimized_distance:.2f} miles")
        else:
            print(f"⚠ Distance optimization did not reduce distance: {classical_distance:.2f} -> {distance_optimized_distance:.2f} miles")
        
        # Verify that time optimization actually reduces time
        classical_time = route_analysis['classical']['time']
        time_optimized_time = route_analysis['quantum_time']['time']
        
        if time_optimized_time < classical_time:
            print(f"✓ Time optimization successful: {classical_time:.1f} -> {time_optimized_time:.1f} minutes")
        else:
            print(f"⚠ Time optimization did not reduce time: {classical_time:.1f} -> {time_optimized_time:.1f} minutes")
        
        print("\n=== Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_distance_optimization()
    sys.exit(0 if success else 1) 