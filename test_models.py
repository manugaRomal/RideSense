#!/usr/bin/env python3
"""
RideSense Model Testing
Test all available models with sample and custom data
"""
import sys
import os
import joblib

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor

def test_single_model(model_name: str, model_path: str) -> bool:
    """Test a single model with sample data"""
    print(f"\n>> Testing {model_name}...")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        print(f"SUCCESS: {model_name} loaded successfully")
        
        # Create predictor instance for preprocessing
        predictor = VehicleConditionPredictor()
        
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
        
        # Preprocess data
        import pandas as pd
        input_df = pd.DataFrame([sample_data])
        processed_data = predictor.preprocess_features(input_df)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)[0]
            proba_dict = {}
            for i, prob in enumerate(probabilities):
                condition_name = predictor.condition_mapping.get(i, f"Class_{i}")
                proba_dict[condition_name] = float(prob)
        else:
            proba_dict = {predictor.condition_mapping.get(prediction, f"Class_{prediction}"): 1.0}
        
        predicted_condition = predictor.condition_mapping.get(prediction, f"Class_{prediction}")
        
        print(f"SUCCESS: Prediction = {predicted_condition}")
        print(f"SUCCESS: Probabilities = {proba_dict}")
        
        # Show model info
        if hasattr(model, 'n_estimators'):
            print(f"SUCCESS: Estimators = {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"SUCCESS: Max Depth = {model.max_depth}")
        if hasattr(model, 'learning_rate'):
            print(f"SUCCESS: Learning Rate = {model.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error testing {model_name}: {str(e)}")
        return False

def test_all_models(model_dir: str = "model") -> None:
    """Test all available models"""
    print("Testing All Available Models")
    print("=" * 50)
    
    # Define models to test
    models_to_test = [
        ("Random Forest", "random_forest.pkl"),
        ("Gradient Boosting", "gradient_boosting.pkl"),
        ("Decision Tree", "decision_tree.pkl"),
        ("Logistic Regression", "logistic_regression.pkl"),
        ("XGBoost Classifier", "xgboost_classifier.pkl")
    ]
    
    successful_models = []
    failed_models = []
    
    for model_name, model_file in models_to_test:
        model_path = os.path.join(model_dir, model_file)
        
        if os.path.exists(model_path):
            if test_single_model(model_name, model_path):
                successful_models.append(model_name)
            else:
                failed_models.append(model_name)
        else:
            print(f"\n>> Testing {model_name}...")
            print(f"SKIP: {model_file} not found")
    
    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)
    
    if successful_models:
        print(f"SUCCESS: {len(successful_models)} models working:")
        for model in successful_models:
            print(f"  [OK] {model}")
    
    if failed_models:
        print(f"FAILED: {len(failed_models)} models failed:")
        for model in failed_models:
            print(f"  [FAIL] {model}")
    
    print(f"\nTotal: {len(successful_models)}/{len(models_to_test)} models working")

def test_app_model():
    """Test the model that the app actually uses (with fallback)"""
    print("\n" + "=" * 50)
    print("TESTING APP MODEL (with fallback)")
    print("=" * 50)
    
    try:
        # Create predictor instance (this will use the fallback logic)
        predictor = VehicleConditionPredictor()
        
        # Check model info
        model_info = predictor.get_model_info()
        print(f"Model info: {model_info}")
        
        if "error" in model_info:
            print(f"ERROR: {model_info['error']}")
            return
        
        model_type = model_info.get('model_type', 'Unknown')
        print(f"SUCCESS: App is using {model_type} model")
        
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
            "paint_color": "white",
            "state": "ca",
            "owners": 1,
            "location_cluster": 10
        }
        
        print(f"\nTesting prediction with {model_type}...")
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
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run all model tests"""
    print("RideSense Model Testing Suite")
    print("=" * 50)
    
    # Test all available models
    test_all_models()
    
    # Test the app's actual model (with fallback)
    test_app_model()
    
    print("\n" + "=" * 50)
    print("TESTING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()