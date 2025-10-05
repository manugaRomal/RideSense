#!/usr/bin/env python3
"""
Test script to verify your trained models are working correctly
Contains all testing functionality
"""
import sys
import os
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor

def test_models_with_sample_data(model_dir: str = "model") -> None:
    """Test all models with sample data"""
    predictor = VehicleConditionPredictor(model_dir)
    models = predictor.load_models()
    
    if not models:
        print("ERROR: No models found!")
        return
    
    # Sample test data
    test_data = pd.DataFrame([{
        "price": 15000,
        "year": 2018,
        "manufacturer": "toyota",
        "model": "camry",
        "cylinders": 4,
        "fuel": "gas",
        "odometer": 75000,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "white",
        "state": "ca",
        "owners": 1,
        "location_cluster": 5
    }])
    
    print("Testing models with sample data:")
    print(test_data.to_string())
    print("\n" + "="*50)
    
    for model_name, model in models.items():
        try:
            print(f"\n>> Testing {model_name}...")
            print(f"SUCCESS: Model loaded successfully")
            
            # Preprocess the data
            processed_data = predictor.preprocess_features(test_data)
            print(f"INFO: Preprocessed data shape: {processed_data.shape}")
            
            # Test prediction
            prediction = model.predict(processed_data)[0]
            predicted_condition = predictor.condition_mapping.get(prediction, f"Class_{prediction}")
            print(f"PREDICTION: {prediction} -> {predicted_condition}")
            
            # Test probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_data)[0]
                proba_dict = {}
                for i, prob in enumerate(probabilities):
                    condition_name = predictor.condition_mapping.get(i, f"Class_{i}")
                    proba_dict[condition_name] = float(prob)
                print(f"PROBABILITIES: {proba_dict}")
            else:
                print("PROBABILITIES: Not available")
            
            # Model info
            info = predictor.get_model_info(model)
            for key, value in info.items():
                print(f"INFO: {key.title()}: {value}")
                
        except Exception as e:
            print(f"ERROR: Error testing {model_name}: {str(e)}")
    
    print("\n" + "="*50)
    print("SUCCESS: Model testing complete!")

def test_specific_model(model_name: str, model_dir: str = "model") -> None:
    """Test a specific model by name"""
    predictor = VehicleConditionPredictor(model_dir)
    models = predictor.load_models()
    
    if model_name not in models:
        print(f"ERROR: Model '{model_name}' not found!")
        print(f"Available models: {list(models.keys())}")
        return
    
    model = models[model_name]
    
    # Sample test data
    test_data = pd.DataFrame([{
        "price": 15000,
        "year": 2018,
        "manufacturer": "toyota",
        "model": "camry",
        "cylinders": 4,
        "fuel": "gas",
        "odometer": 75000,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "white",
        "state": "ca",
        "owners": 1,
        "location_cluster": 5
    }])
    
    print(f"Testing {model_name} with sample data:")
    print(test_data.to_string())
    print("\n" + "="*50)
    
    try:
        # Preprocess the data
        processed_data = predictor.preprocess_features(test_data)
        print(f"INFO: Preprocessed data shape: {processed_data.shape}")
        
        # Test prediction
        prediction = model.predict(processed_data)[0]
        predicted_condition = predictor.condition_mapping.get(prediction, f"Class_{prediction}")
        print(f"PREDICTION: {prediction} -> {predicted_condition}")
        
        # Test probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)[0]
            proba_dict = {}
            for i, prob in enumerate(probabilities):
                condition_name = predictor.condition_mapping.get(i, f"Class_{i}")
                proba_dict[condition_name] = float(prob)
            print(f"PROBABILITIES: {proba_dict}")
        else:
            print("PROBABILITIES: Not available")
        
        # Model info
        info = predictor.get_model_info(model)
        for key, value in info.items():
            print(f"INFO: {key.title()}: {value}")
            
    except Exception as e:
        print(f"ERROR: Error testing {model_name}: {str(e)}")

def test_custom_data(custom_data: dict, model_dir: str = "model") -> None:
    """Test models with custom data"""
    predictor = VehicleConditionPredictor(model_dir)
    models = predictor.load_models()
    
    if not models:
        print("ERROR: No models found!")
        return
    
    # Create DataFrame from custom data
    test_data = pd.DataFrame([custom_data])
    
    print("Testing models with custom data:")
    print(test_data.to_string())
    print("\n" + "="*50)
    
    for model_name, model in models.items():
        try:
            print(f"\n>> Testing {model_name}...")
            
            # Preprocess the data
            processed_data = predictor.preprocess_features(test_data)
            
            # Test prediction
            prediction = model.predict(processed_data)[0]
            predicted_condition = predictor.condition_mapping.get(prediction, f"Class_{prediction}")
            print(f"PREDICTION: {prediction} -> {predicted_condition}")
            
            # Test probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_data)[0]
                proba_dict = {}
                for i, prob in enumerate(probabilities):
                    condition_name = predictor.condition_mapping.get(i, f"Class_{i}")
                    proba_dict[condition_name] = float(prob)
                print(f"PROBABILITIES: {proba_dict}")
            else:
                print("PROBABILITIES: Not available")
                
        except Exception as e:
            print(f"ERROR: Error testing {model_name}: {str(e)}")
    
    print("\n" + "="*50)
    print("SUCCESS: Custom data testing complete!")

def main():
    """Main function to run model tests"""
    print("RideSense Model Testing")
    print("=" * 50)
    
    # Test all models with sample data
    test_models_with_sample_data()
    
    # Example: Test specific model
    # test_specific_model("Random Forest")
    
    # Example: Test with custom data
    # custom_vehicle = {
    #     "price": 25000,
    #     "year": 2020,
    #     "manufacturer": "honda",
    #     "model": "accord",
    #     "cylinders": 4,
    #     "fuel": "gas",
    #     "odometer": 30000,
    #     "title_status": "clean",
    #     "transmission": "automatic",
    #     "drive": "fwd",
    #     "type": "sedan",
    #     "paint_color": "black",
    #     "state": "ny",
    #     "owners": 1,
    #     "location_cluster": 10
    # }
    # test_custom_data(custom_vehicle)

if __name__ == "__main__":
    main()
