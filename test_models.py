#!/usr/bin/env python3
"""
RideSense Gradient Boosting Model Testing
Test the Gradient Boosting model with sample and custom data
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor

def test_gradient_boosting_model(model_dir: str = "model") -> None:
    """Test Gradient Boosting model with sample data"""
    print("Testing Gradient Boosting model with sample data:")
    
    # Sample vehicle data
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
        "paint_color": "white",
        "state": "ca",
        "owners": 1,
        "location_cluster": 10
    }
    
    print("Sample data:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    
    try:
        # Create predictor instance
        predictor = VehicleConditionPredictor()
        
        # Test model loading
        model_info = predictor.get_model_info()
        if "error" in model_info:
            print(f"ERROR: {model_info['error']}")
            return
        
        print(f"SUCCESS: Model loaded - {model_info['model_type']}")
        print(f"Features: {model_info.get('features_count', 'Unknown')}")
        print(f"Classes: {model_info.get('classes_count', 'Unknown')}")
        print(f"Estimators: {model_info.get('n_estimators', 'Unknown')}")
        
        # Test prediction
        print(f"\n>> Testing Gradient Boosting...")
        prediction, probabilities = predictor.predict_condition(sample_data)
        
        if prediction and not prediction.startswith("Error"):
            print(f"SUCCESS: Prediction = {prediction}")
            print(f"SUCCESS: Probabilities = {probabilities}")
            
            # Test market analysis
            price_analysis = predictor.analyze_market_price(sample_data, prediction)
            print(f"SUCCESS: Market Value = ${price_analysis['estimated_market_value']:,.0f}")
            print(f"SUCCESS: Price vs Market = {price_analysis['price_vs_market']:.1f}%")
            
        else:
            print(f"ERROR: Prediction failed - {prediction}")
            
    except Exception as e:
        print(f"ERROR: Error testing Gradient Boosting: {str(e)}")
        import traceback
        traceback.print_exc()

def test_custom_data(custom_vehicle: dict) -> None:
    """Test with custom vehicle data"""
    print(f"\nTesting with custom vehicle data:")
    print("-" * 40)
    
    try:
        predictor = VehicleConditionPredictor()
        
        # Test prediction
        prediction, probabilities = predictor.predict_condition(custom_vehicle)
        
        if prediction and not prediction.startswith("Error"):
            print(f"SUCCESS: Prediction = {prediction}")
            print(f"SUCCESS: Probabilities = {probabilities}")
            
            # Test market analysis
            price_analysis = predictor.analyze_market_price(custom_vehicle, prediction)
            print(f"SUCCESS: Market Value = ${price_analysis['estimated_market_value']:,.0f}")
            print(f"SUCCESS: Price vs Market = {price_analysis['price_vs_market']:.1f}%")
            
        else:
            print(f"ERROR: Prediction failed - {prediction}")
            
    except Exception as e:
        print(f"ERROR: Error testing custom data: {str(e)}")

def main():
    """Main function to run Gradient Boosting model tests"""
    print("RideSense Gradient Boosting Model Testing")
    print("=" * 50)

    # Test Gradient Boosting model with sample data
    test_gradient_boosting_model()

    # Example: Test with custom data
    # custom_vehicle = {
    #     "price": 35000,
    #     "year": 2021,
    #     "manufacturer": "toyota",
    #     "model": "camry",
    #     "cylinders": 4,
    #     "fuel": "gas",
    #     "odometer": 25000,
    #     "title_status": "clean",
    #     "transmission": "automatic",
    #     "drive": "fwd",
    #     "type": "sedan",
    #     "paint_color": "blue",
    #     "state": "ny",
    #     "owners": 1,
    #     "location_cluster": 15
    # }
    # test_custom_data(custom_vehicle)

if __name__ == "__main__":
    main()