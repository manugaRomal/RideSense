#!/usr/bin/env python3
"""
Test script to verify visualizations are working correctly
"""
import sys
import os
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logic import VehicleConditionPredictor
from src.ui import RideSenseUI

def test_visualizations():
    """Test that all visualization methods work without errors"""
    print("Testing RideSense Visualizations")
    print("=" * 50)
    
    try:
        # Create predictor and UI instances
        predictor = VehicleConditionPredictor()
        ui = RideSenseUI()
        
        # Sample data for testing
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
        
        # Test prediction
        prediction, probabilities = predictor.predict_condition(sample_data)
        
        if prediction:
            print(f"✅ Prediction successful: {prediction}")
            print(f"✅ Probabilities: {probabilities}")
            
            # Test price analysis
            price_analysis = predictor.analyze_market_price(sample_data, prediction)
            print(f"✅ Price analysis: {price_analysis}")
            
            # Test visualization methods (without actually rendering)
            print("✅ Testing visualization methods...")
            
            # Test gauge
            ui.render_condition_gauge(prediction, probabilities)
            print("✅ Condition gauge method works")
            
            # Test probability chart
            ui.render_probability_chart(probabilities)
            print("✅ Probability chart method works")
            
            # Test price comparison
            ui.render_price_comparison_chart(sample_data, price_analysis)
            print("✅ Price comparison chart method works")
            
            # Test age mileage chart
            ui.render_age_mileage_chart(sample_data)
            print("✅ Age mileage chart method works")
            
            # Test market trend
            ui.render_market_trend_chart(sample_data)
            print("✅ Market trend chart method works")
            
            # Test feature importance
            ui.render_feature_importance_chart()
            print("✅ Feature importance chart method works")
            
            # Test condition distribution
            ui.render_condition_distribution_chart()
            print("✅ Condition distribution chart method works")
            
            # Test maintenance timeline
            ui.render_maintenance_timeline(sample_data)
            print("✅ Maintenance timeline method works")
            
            print("\n🎉 All visualization tests passed!")
            
        else:
            print("❌ Prediction failed")
            
    except Exception as e:
        print(f"❌ Error testing visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualizations()
