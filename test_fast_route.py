#!/usr/bin/env python3
"""
Fast test script to verify route service functionality without slow geocoding
"""

import time
from route_service import get_route_service

def test_fast_route_service():
    """Test the route service with fast coordinates"""
    print("=== Testing Fast Route Service ===")
    
    try:
        # Get route service
        print("1. Getting route service...")
        start_time = time.time()
        route_service = get_route_service()
        init_time = time.time() - start_time
        print(f"✓ Route service initialized successfully in {init_time:.2f} seconds")
        
        # Test with two zones
        start_zone = 1  # Manhattan
        end_zone = 2    # Manhattan
        
        print(f"2. Testing route from zone {start_zone} to zone {end_zone}...")
        start_time = time.time()
        
        # Get route analysis
        route_analysis = route_service.get_route_analysis(start_zone, end_zone)
        
        if route_analysis is None:
            print("✗ Route analysis failed - returned None")
            return False
        
        total_time = time.time() - start_time
        print(f"✓ Route analysis completed in {total_time:.2f} seconds")
        
        # Check results
        print("3. Checking results...")
        if 'classical' in route_analysis and 'quantum' in route_analysis:
            print("✓ Route analysis contains classical and quantum data")
            print(f"  Classical: {route_analysis['classical']['distance']} miles, {route_analysis['classical']['time']} minutes")
            print(f"  Quantum: {route_analysis['quantum']['distance']} miles, {route_analysis['quantum']['time']} minutes")
        else:
            print("✗ Route analysis missing required data")
            return False
        
        if 'coordinates' in route_analysis:
            print("✓ Route analysis contains coordinates")
            print(f"  Start: {route_analysis['coordinates']['start']}")
            print(f"  End: {route_analysis['coordinates']['end']}")
        else:
            print("✗ Route analysis missing coordinates")
            return False
        
        print("=== All tests passed! ===")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fast_route_service()
    if success:
        print("Fast route service is working correctly!")
    else:
        print("Fast route service has issues that need to be fixed.") 