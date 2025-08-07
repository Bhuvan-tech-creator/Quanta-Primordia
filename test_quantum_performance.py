#!/usr/bin/env python3
"""
Quantum Performance Test Script
Tests the quantum optimization performance and ensures it stays within 2-minute time limit
"""

import time
import sys
import traceback
from route_service import get_route_service

def test_quantum_performance():
    """Test quantum optimization performance with time monitoring"""
    print("=== Quantum Performance Test ===")
    print("Testing quantum optimization with 2-minute time limit...")
    
    start_time = time.time()
    max_time = 120  # 2 minutes
    
    try:
        # Initialize route service
        print("Initializing route service...")
        route_service = get_route_service()
        
        # Test route optimization
        print("Testing route optimization from zone 1 to zone 10...")
        route_analysis = route_service.get_route_analysis(1, 10)
        
        if route_analysis is None:
            print("❌ ERROR: Route analysis returned None")
            return False
        
        total_time = time.time() - start_time
        
        print(f"✅ Route optimization completed in {total_time:.2f} seconds")
        
        # Check time limit
        if total_time > max_time:
            print(f"❌ WARNING: Exceeded {max_time}s time limit by {total_time - max_time:.2f}s")
            return False
        else:
            print(f"✅ Time limit check passed ({total_time:.2f}s < {max_time}s)")
        
        # Check quantum optimization results
        if 'quantum' in route_analysis:
            quantum_data = route_analysis['quantum']
            classical_data = route_analysis['classical']
            
            print(f"Classical route: {classical_data['distance']:.2f} miles, {classical_data['time']:.1f} minutes")
            print(f"Quantum route: {quantum_data['distance']:.2f} miles, {quantum_data['time']:.1f} minutes")
            
            # Check for improvements
            distance_improvement = ((classical_data['distance'] - quantum_data['distance']) / classical_data['distance']) * 100
            time_improvement = ((classical_data['time'] - quantum_data['time']) / classical_data['time']) * 100
            
            print(f"Distance improvement: {distance_improvement:.1f}%")
            print(f"Time improvement: {time_improvement:.1f}%")
            
            if quantum_data['distance'] < classical_data['distance'] and quantum_data['time'] < classical_data['time']:
                print("✅ Quantum optimization provided improvements")
                return True
            else:
                print("⚠️ Quantum optimization did not provide expected improvements")
                return False
        else:
            print("❌ ERROR: No quantum data in route analysis")
            return False
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ ERROR: Test failed after {total_time:.2f} seconds")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_multiple_routes():
    """Test multiple route optimizations to ensure consistent performance"""
    print("\n=== Multiple Route Performance Test ===")
    
    test_routes = [
        (1, 10),
        (5, 15),
        (10, 20),
        (15, 25),
        (20, 30)
    ]
    
    route_service = get_route_service()
    successful_tests = 0
    
    for i, (start_zone, end_zone) in enumerate(test_routes, 1):
        print(f"\nTest {i}: Zone {start_zone} to Zone {end_zone}")
        start_time = time.time()
        
        try:
            route_analysis = route_service.get_route_analysis(start_zone, end_zone)
            test_time = time.time() - start_time
            
            if route_analysis and test_time <= 120:
                print(f"✅ Test {i} completed in {test_time:.2f}s")
                successful_tests += 1
            else:
                print(f"❌ Test {i} failed or exceeded time limit ({test_time:.2f}s)")
                
        except Exception as e:
            test_time = time.time() - start_time
            print(f"❌ Test {i} failed after {test_time:.2f}s: {e}")
    
    print(f"\n=== Multiple Route Test Results ===")
    print(f"Successful tests: {successful_tests}/{len(test_routes)}")
    print(f"Success rate: {(successful_tests/len(test_routes))*100:.1f}%")
    
    return successful_tests == len(test_routes)

if __name__ == "__main__":
    print("Starting Quantum Performance Tests...")
    
    # Test 1: Single route optimization
    test1_passed = test_quantum_performance()
    
    # Test 2: Multiple route optimizations
    test2_passed = test_multiple_routes()
    
    print("\n=== Final Results ===")
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED - Quantum optimization working within time limits")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Quantum optimization needs adjustment")
        sys.exit(1) 