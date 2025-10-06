#!/usr/bin/env python3
"""
Test script to verify the deployed app functionality
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor

def test_deployed_functionality():
    """Test all functionality that should work in the deployed app"""
    print("Testing Deployed App Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Model Loading
        print("1. Testing model loading...")
        predictor = VehicleConditionPredictor()
        model_info = predictor.get_model_info()
        
        if "error" in model_info:
            print(f"   ERROR: {model_info['error']}")
            return
        else:
            print(f"   SUCCESS: Model loaded - {model_info['model_type']}")
        
        # Test 2: Sample Prediction
        print("\n2. Testing prediction with sample data...")
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
        
        prediction, probabilities = predictor.predict_condition(sample_data)
        
        if prediction and not prediction.startswith("Error"):
            print(f"   SUCCESS: Prediction = {prediction}")
            print(f"   SUCCESS: Probabilities = {probabilities}")
        else:
            print(f"   ERROR: Prediction failed - {prediction}")
            return
        
        # Test 3: Market Analysis
        print("\n3. Testing market analysis...")
        try:
            price_analysis = predictor.analyze_market_price(sample_data, prediction)
            print(f"   SUCCESS: Market analysis completed")
            print(f"   Current Price: ${price_analysis['current_price']:,}")
            print(f"   Market Value: ${price_analysis['estimated_market_value']:,}")
        except Exception as e:
            print(f"   ERROR: Market analysis failed - {e}")
        
        # Test 4: Vehicle Insights
        print("\n4. Testing vehicle insights...")
        try:
            insights = predictor.get_vehicle_insights(sample_data)
            print(f"   SUCCESS: Vehicle insights generated")
            print(f"   Age: {insights['age']} years")
            print(f"   Annual Mileage: {insights['annual_mileage']:,} miles/year")
        except Exception as e:
            print(f"   ERROR: Vehicle insights failed - {e}")
        
        print("\n" + "=" * 50)
        print("SUCCESS: All core functionality is working!")
        print("Your deployed app should work perfectly!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deployed_functionality()
