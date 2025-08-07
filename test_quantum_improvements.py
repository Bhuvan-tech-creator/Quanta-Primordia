#!/usr/bin/env python3
"""
Test script to verify that quantum optimization now provides better results.
"""

from route_service import get_route_service

def test_quantum_improvements():
    """Test that quantum optimization provides better results than classical"""
    print("Testing quantum optimization improvements...")
    
    # Get route service
    service = get_route_service()
    
    # Test route analysis
    result = service.get_route_analysis(1, 2)
    
    if result:
        classical = result['classical']
        quantum = result['quantum']
        improvements = result['improvements']
        
        print(f"\nClassical Route:")
        print(f"  Distance: {classical['distance']} miles")
        print(f"  Time: {classical['time']} minutes")
        print(f"  CO2: {classical['co2']} kg")
        
        print(f"\nQuantum Route:")
        print(f"  Distance: {quantum['distance']} miles")
        print(f"  Time: {quantum['time']} minutes")
        print(f"  CO2: {quantum['co2']} kg")
        
        print(f"\nImprovements:")
        print(f"  Distance saved: {improvements['distance_saved']} miles")
        print(f"  Time saved: {improvements['time_saved']} minutes")
        print(f"  CO2 saved: {improvements['co2_saved']} kg")
        print(f"  Distance improvement: {improvements['distance_improvement']}%")
        print(f"  Time improvement: {improvements['time_improvement']}%")
        
        # Check if quantum route is actually better
        distance_better = quantum['distance'] < classical['distance']
        time_better = quantum['time'] < classical['time']
        co2_better = quantum['co2'] < classical['co2']
        
        print(f"\nAnalysis:")
        print(f"  Distance better: {'✓' if distance_better else '✗'}")
        print(f"  Time better: {'✓' if time_better else '✗'}")
        print(f"  CO2 better: {'✓' if co2_better else '✗'}")
        
        overall_better = distance_better and time_better
        print(f"  Overall better: {'✓' if overall_better else '✗'}")
        
        if overall_better:
            print("🎉 Quantum optimization is working correctly!")
            return True
        else:
            print("⚠️  Quantum optimization needs improvement.")
            return False
    else:
        print("✗ Failed to get route analysis")
        return False

if __name__ == "__main__":
    success = test_quantum_improvements()
    exit(0 if success else 1) 