#!/usr/bin/env python3
"""
Script to install XGBoost and test all models
"""
import subprocess
import sys
import os

def install_xgboost():
    """Install XGBoost package"""
    print("INFO: Installing XGBoost...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==2.0.2"])
        print("SUCCESS: XGBoost installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install XGBoost: {e}")
        return False

def test_models():
    """Run the model test script"""
    print("\nINFO: Testing models...")
    try:
        subprocess.check_call([sys.executable, "test_models.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Model testing failed: {e}")
        return False

def main():
    print("RideSense Model Setup and Testing")
    print("=" * 50)
    
    # Check if XGBoost is already installed
    try:
        import xgboost
        print("SUCCESS: XGBoost is already installed")
    except ImportError:
        print("WARNING: XGBoost not found, installing...")
        if not install_xgboost():
            print("WARNING: Continuing without XGBoost...")
    
    # Test models
    if test_models():
        print("\nSUCCESS: All tests completed successfully!")
        print("INFO: You can now run: streamlit run app.py")
    else:
        print("\nWARNING: Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
