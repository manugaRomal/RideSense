#!/usr/bin/env python3
"""
Test script for Random Forest model from Google Drive
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor

def test_random_forest():
    """Test Random Forest model functionality"""
    print("Testing Random Forest Model from Google Drive")
    print("=" * 50)
    
    try:
        # Create predictor instance
        predictor = VehicleConditionPredictor()
        
        # Check model info
        model_info = predictor.get_model_info()
        print(f"Model info: {model_info}")
        
        if "error" in model_info:
            print(f"ERROR: {model_info['error']}")
            print("\nMake sure you've set up the Google Drive file ID in src/logic.py")
            return
        
        # Test with sample data
        sample_data = {
            "price": 25000,
            "year": 2020,
            "manufacturer": "honda",
            "model": "accord",
            "cylinders": 4,
            "fuel": "gas",
            "odometer": 30000,
            "title_status": "clean",
            "transmission": "automatic",
            "drive": "fwd",
            "type": "sedan",
            "paint_color": "black",
            "state": "ca",
            "owners": 1,
            "location_cluster": 10
        }
        
        print("\nTesting prediction...")
        prediction, probabilities = predictor.predict_condition(sample_data)
        
        if prediction and not prediction.startswith("Error"):
            print(f"SUCCESS: Prediction = {prediction}")
            print(f"SUCCESS: Probabilities = {probabilities}")
            
            # Test market analysis
            price_analysis = predictor.analyze_market_price(sample_data, prediction)
            print(f"SUCCESS: Market Value = ${price_analysis['estimated_market_value']:,.0f}")
            
        else:
            print(f"ERROR: Prediction failed - {prediction}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_random_forest()
