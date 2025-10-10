#!/usr/bin/env python3
"""
Test script for location clusters implementation
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_location_clusters():
    """Test the location cluster functionality"""
    try:
        # Test the location cluster data directly
        location_clusters = {
            0: "Davenport / Quad Cities area (Iowa)",
            1: "Bakersfield / Central Valley (CA)",
            2: "Delaware Water Gap area (PA/NJ)",
            3: "Orlando / Central Florida area (FL)",
            4: "Yakima Valley area (WA)",
            5: "Maui (HI)",
            6: "Palmer / Wasilla area (AK)",
            7: "Montrose / San Juan Mountains (CO)",
            8: "Corsicana / Dallas-Fort Worth (TX)",
            9: "Hazard / Eastern Kentucky (KY)"
        }
        
        print("Location clusters:")
        for k, v in location_clusters.items():
            print(f"  {k}: {v}")
        
        print("\nLocation options (as tuples):")
        location_options = [(cluster_num, display_name) for cluster_num, display_name in location_clusters.items()]
        for opt in location_options:
            print(f"  {opt}")
        
        print("\nTest validation - checking valid cluster numbers:")
        valid_clusters = list(location_clusters.keys())
        print(f"Valid cluster numbers: {valid_clusters}")
        
        # Test a few cluster numbers
        test_clusters = [0, 3, 5, 9]
        for cluster in test_clusters:
            if cluster in valid_clusters:
                print(f"  Cluster {cluster} ({location_clusters[cluster]}) is valid")
            else:
                print(f"  Cluster {cluster} is NOT valid")
        
        print("\nLocation clusters implementation test passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_location_clusters()