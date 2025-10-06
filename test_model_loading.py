#!/usr/bin/env python3
"""
Test script to check if the Decision Tree model is loading correctly
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor

def test_model_loading():
    """Test if the Decision Tree model loads correctly"""
    print("Testing Decision Tree Model Loading")
    print("=" * 50)
    
    try:
        # Create predictor instance
        predictor = VehicleConditionPredictor()
        
        # Check if model directory exists
        print(f"Model directory exists: {os.path.exists('model')}")
        print(f"Model directory contents: {os.listdir('model') if os.path.exists('model') else 'Directory not found'}")
        
        # Check if decision_tree.pkl exists
        model_path = os.path.join('model', 'decision_tree.pkl')
        print(f"Decision tree model exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"Model file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Check model info
        model_info = predictor.get_model_info()
        print(f"Model info: {model_info}")
        
        # Test prediction with sample data
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
        
        if prediction:
            print(f"SUCCESS: Prediction successful: {prediction}")
            print(f"SUCCESS: Probabilities: {probabilities}")
        else:
            print("ERROR: Prediction failed")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
