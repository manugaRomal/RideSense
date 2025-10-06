#!/usr/bin/env python3
"""
Test script to verify the separated architecture works correctly
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported correctly"""
    try:
        from src.logic import VehicleConditionPredictor
        from src.ui import RideSenseUI
        print("SUCCESS: All modules imported successfully!")
        return True
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False

def test_logic_module():
    """Test the logic module functionality"""
    try:
        from src.logic import VehicleConditionPredictor
        
        # Create predictor instance
        predictor = VehicleConditionPredictor()
        print("SUCCESS: VehicleConditionPredictor created successfully!")
        
        # Test model loading
        models = predictor.load_models()
        print(f"SUCCESS: Models loaded: {len(models)} models found")
        
        # Test condition mapping
        condition = predictor.condition_mapping[3]
        print(f"SUCCESS: Condition mapping works: 3 -> {condition}")
        
        return True
    except Exception as e:
        print(f"ERROR: Logic module error: {e}")
        return False

def test_ui_module():
    """Test the UI module functionality"""
    try:
        from src.ui import RideSenseUI
        
        # Create UI instance
        ui = RideSenseUI()
        print("SUCCESS: RideSenseUI created successfully!")
        
        # Test predictor access
        predictor = ui.predictor
        print("SUCCESS: UI has access to predictor!")
        
        return True
    except Exception as e:
        print(f"ERROR: UI module error: {e}")
        return False

def main():
    """Main test function"""
    print("Testing Separated Architecture")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Logic Module Test", test_logic_module),
        ("UI Module Test", test_ui_module)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n>> Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name} failed!")
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Architecture is working correctly.")
        print("INFO: You can now run: streamlit run app.py")
    else:
        print("WARNING: Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
