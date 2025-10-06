#!/usr/bin/env python3
"""
Debug script to check deployment environment
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_environment():
    """Debug the deployment environment"""
    print("=== DEPLOYMENT DEBUG INFO ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    print("\n=== DIRECTORY STRUCTURE ===")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    print("\n=== MODEL DIRECTORY CHECK ===")
    model_dir = "model"
    print(f"Model directory exists: {os.path.exists(model_dir)}")
    if os.path.exists(model_dir):
        print(f"Model directory contents: {os.listdir(model_dir)}")
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(model_dir, file)
                size = os.path.getsize(file_path)
                print(f"  {file}: {size:,} bytes ({size/1024/1024:.2f} MB)")
    
    print("\n=== TRYING TO LOAD MODEL ===")
    try:
        from src.logic import VehicleConditionPredictor
        predictor = VehicleConditionPredictor()
        
        model_info = predictor.get_model_info()
        print(f"Model info: {model_info}")
        
        if "error" in model_info:
            print(f"ERROR: {model_info['error']}")
        else:
            print("SUCCESS: Model loaded successfully")
            
    except Exception as e:
        print(f"ERROR loading predictor: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_environment()
