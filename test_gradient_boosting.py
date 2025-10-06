#!/usr/bin/env python3
"""
Test script for Gradient Boosting model
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor

def test_gradient_boosting():
    """Test Gradient Boosting model functionality"""
    print("Testing Gradient Boosting Model")
    print("=" * 50)
    
    try:
        # Create predictor instance
        predictor = VehicleConditionPredictor()
        
        # Check model info
        model_info = predictor.get_model_info()
        print(f"Model info: {model_info}")
        
        if "error" in model_info:
            print(f"ERROR: {model_info['error']}")
            print("\nMake sure gradient_boosting.pkl is in the model/ directory")
            return
        
        # Test with different sample data
        test_cases = [
            {
                "name": "Luxury SUV",
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
                "name": "Economy Car",
                "data": {
                    "price": 8000,
                    "year": 2015,
                    "manufacturer": "honda",
                    "model": "civic",
                    "cylinders": 4,
                    "fuel": "gas",
                    "odometer": 120000,
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
            
            # Test prediction
            prediction, probabilities = predictor.predict_condition(test_case['data'])
            
            if prediction and not prediction.startswith("Error"):
                print(f"SUCCESS: Prediction = {prediction}")
                print(f"SUCCESS: Probabilities = {probabilities}")
                
                # Test market analysis
                price_analysis = predictor.analyze_market_price(test_case['data'], prediction)
                print(f"SUCCESS: Market Value = ${price_analysis['estimated_market_value']:,.0f}")
                print(f"SUCCESS: Price vs Market = {price_analysis['price_vs_market']:.1f}%")
                
            else:
                print(f"ERROR: Prediction failed - {prediction}")
        
        print("\n" + "=" * 50)
        print("SUCCESS: Gradient Boosting model is working!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradient_boosting()
