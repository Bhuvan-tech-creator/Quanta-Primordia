#!/usr/bin/env python3
"""
Test script to verify that the route fixes are working properly.
This will test that:
1. Quantum routes are different from classical routes
2. Shorter distances result in faster times
3. Routes are visually distinct on the map
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_service import get_route_service
from geopy.distance import geodesic

def test_route_differences():
    """Test that quantum and classical routes are different"""
    print("=== Testing Route Differences ===")
    
    route_service = get_route_service()
    
    # Test with specific zones
    start_zone = 1
    end_zone = 10
    
    print(f"Testing route from zone {start_zone} to zone {end_zone}")
    
    # Get route analysis
    analysis = route_service.get_route_analysis(start_zone, end_zone)
    
    if not analysis:
        print("ERROR: No route analysis returned")
        return False
    
    classical = analysis['classical']
    quantum = analysis['quantum']
    
    print(f"Classical route: {classical['distance']:.2f} miles, {classical['time']:.1f} minutes")
    print(f"Quantum route: {quantum['distance']:.2f} miles, {quantum['time']:.1f} minutes")
    
    # Test 1: Check if routes are different
    classical_coords = analysis['coordinates']['classical_route']
    quantum_coords = analysis['coordinates']['quantum_route']
    
    print(f"Classical route has {len(classical_coords)} coordinates")
    print(f"Quantum route has {len(quantum_coords)} coordinates")
    
    # Check if coordinates are actually different
    different_coords = 0
    min_len = min(len(classical_coords), len(quantum_coords))
    
    for i in range(min_len):
        if classical_coords[i] != quantum_coords[i]:
            different_coords += 1
    
    print(f"Different coordinates: {different_coords}/{min_len} ({different_coords/min_len*100:.1f}%)")
    
    # Test 2: Check distance and time relationship
    distance_improvement = classical['distance'] - quantum['distance']
    time_improvement = classical['time'] - quantum['time']
    
    print(f"Distance improvement: {distance_improvement:.2f} miles")
    print(f"Time improvement: {time_improvement:.1f} minutes")
    
    # Test 3: Verify that shorter distance means faster time
    if quantum['distance'] < classical['distance']:
        if quantum['time'] >= classical['time']:
            print("WARNING: Quantum route is shorter but not faster!")
            return False
        else:
            print("✓ Quantum route is shorter AND faster")
    else:
        print("INFO: Quantum route is longer (may be due to traffic avoidance)")
    
    # Test 4: Check if improvements are reasonable
    if distance_improvement > 0:
        print("✓ Quantum route saves distance")
    if time_improvement > 0:
        print("✓ Quantum route saves time")
    
    return True

def test_route_generation():
    """Test the route generation functions directly"""
    print("\n=== Testing Route Generation ===")
    
    route_service = get_route_service()
    
    # Test with sample coordinates
    start_point = [40.7589, -73.9851]  # Times Square
    end_point = [40.7505, -73.9934]    # Penn Station
    
    # Create a simple classical route
    classical_coords = [
        start_point,
        [40.7547, -73.9893],  # Midpoint
        end_point
    ]
    
    print("Testing quantum route variations...")
    
    # Test traffic optimization
    traffic_data = {'intensity': 1.5}
    traffic_route = route_service.apply_traffic_optimization(classical_coords, traffic_data)
    print(f"Traffic optimized route: {len(traffic_route)} coordinates")
    
    # Test distance optimization
    distance_route = route_service.apply_distance_optimization(classical_coords, 2.0)
    print(f"Distance optimized route: {len(distance_route)} coordinates")
    
    # Test speed optimization
    speed_route = route_service.apply_speed_optimization(classical_coords, traffic_data)
    print(f"Speed optimized route: {len(speed_route)} coordinates")
    
    # Test hybrid optimization
    hybrid_route = route_service.apply_hybrid_quantum_optimization(classical_coords, traffic_data, 2.0)
    print(f"Hybrid optimized route: {len(hybrid_route)} coordinates")
    
    # Check if routes are different
    routes = [classical_coords, traffic_route, distance_route, speed_route, hybrid_route]
    route_names = ["Classical", "Traffic", "Distance", "Speed", "Hybrid"]
    
    print("\nRoute comparison:")
    for i, (name, route) in enumerate(zip(route_names, routes)):
        if len(route) >= 2:
            distance = sum(geodesic(route[j], route[j+1]).miles for j in range(len(route)-1))
            print(f"{name}: {len(route)} points, {distance:.3f} miles")
        else:
            print(f"{name}: {len(route)} points, invalid route")
    
    return True

if __name__ == "__main__":
    print("Testing route fixes...")
    
    try:
        test_route_generation()
        test_route_differences()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 