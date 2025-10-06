#!/usr/bin/env python3
"""
Debug script to test different inputs and see what's happening
"""
import sys
import os
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor

def test_different_inputs():
    """Test with different inputs to see if predictions change"""
    print("Testing Different Inputs for Prediction Variation")
    print("=" * 60)
    
    predictor = VehicleConditionPredictor()
    
    # Test cases with very different inputs
    test_cases = [
        {
            "name": "Expensive New Car",
            "data": {
                "price": 50000,
                "year": 2023,
                "manufacturer": "bmw",
                "model": "x5",
                "cylinders": 6,
                "fuel": "gas",
                "odometer": 5000,
                "title_status": "clean",
                "transmission": "automatic",
                "drive": "awd",
                "type": "suv",
                "paint_color": "black",
                "state": "ca",
                "owners": 0,
                "location_cluster": 1
            }
        },
        {
            "name": "Cheap Old Car",
            "data": {
                "price": 2000,
                "year": 2005,
                "manufacturer": "ford",
                "model": "focus",
                "cylinders": 4,
                "fuel": "gas",
                "odometer": 200000,
                "title_status": "salvage",
                "transmission": "manual",
                "drive": "fwd",
                "type": "sedan",
                "paint_color": "red",
                "state": "tx",
                "owners": 5,
                "location_cluster": 50
            }
        },
        {
            "name": "Mid-range Car",
            "data": {
                "price": 15000,
                "year": 2018,
                "manufacturer": "honda",
                "model": "accord",
                "cylinders": 4,
                "fuel": "gas",
                "odometer": 80000,
                "title_status": "clean",
                "transmission": "automatic",
                "drive": "fwd",
                "type": "sedan",
                "paint_color": "white",
                "state": "ny",
                "owners": 2,
                "location_cluster": 25
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        # Show input data
        print("Input data:")
        for key, value in test_case['data'].items():
            print(f"  {key}: {value}")
        
        # Test preprocessing
        print("\nTesting preprocessing...")
        try:
            input_df = pd.DataFrame([test_case['data']])
            processed_data = predictor.preprocess_features(input_df)
            print(f"  Processed shape: {processed_data.shape}")
            print(f"  Processed columns: {list(processed_data.columns)}")
            print(f"  Data types: {processed_data.dtypes.to_dict()}")
            print(f"  Sample values: {processed_data.iloc[0].to_dict()}")
        except Exception as e:
            print(f"  ERROR in preprocessing: {e}")
            continue
        
        # Test prediction
        print("\nTesting prediction...")
        try:
            prediction, probabilities = predictor.predict_condition(test_case['data'])
            print(f"  Prediction: {prediction}")
            print(f"  Probabilities: {probabilities}")
            
            # Test market analysis
            price_analysis = predictor.analyze_market_price(test_case['data'], prediction)
            print(f"  Market Value: ${price_analysis['estimated_market_value']:,.0f}")
            print(f"  Price vs Market: {price_analysis['price_vs_market']:.1f}%")
            
        except Exception as e:
            print(f"  ERROR in prediction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_different_inputs()
