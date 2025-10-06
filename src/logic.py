"""
RideSense Logic Module
Contains all the business logic for vehicle condition prediction
"""
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import requests
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple, Optional

# Suppress XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

class VehicleConditionPredictor:
    """Main class for vehicle condition prediction logic"""
    
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.model = None
        self.condition_mapping = {
            0: "New",
            1: "Like New", 
            2: "Excellent",
            3: "Good",
            4: "Fair",
            5: "Salvage"
        }
        self.load_model()
    
    def load_model(self) -> bool:
        """Load Random Forest model from Google Drive or fallback to Decision Tree"""
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Try Random Forest first
        rf_model_path = os.path.join(self.model_dir, 'random_forest.pkl')
        
        # Try to load Random Forest from local file first
        if os.path.exists(rf_model_path):
            try:
                self.model = joblib.load(rf_model_path)
                print("Random Forest model loaded from local file")
                return True
            except Exception as e:
                print(f"Error loading local Random Forest model: {e}")
        
        # Try to download Random Forest from Google Drive
        print("Local Random Forest model not found, attempting to download from Google Drive...")
        if self.download_model_from_drive():
            return True
        
        # Fallback to Decision Tree if Random Forest download fails
        print("Random Forest download failed, falling back to Decision Tree model...")
        dt_model_path = os.path.join(self.model_dir, 'decision_tree.pkl')
        
        if os.path.exists(dt_model_path):
            try:
                self.model = joblib.load(dt_model_path)
                print("Decision Tree model loaded as fallback")
                return True
            except Exception as e:
                print(f"Error loading Decision Tree fallback model: {e}")
        
        print("No models available - neither Random Forest nor Decision Tree could be loaded")
        return False
    
    def download_model_from_drive(self) -> bool:
        """Download Random Forest model from Google Drive"""
        # Google Drive file ID - you'll need to replace this with your actual file ID
        # To get the file ID: 1. Upload your random_forest.pkl to Google Drive
        # 2. Right-click and "Get link" 
        # 3. Extract the file ID from the URL (long string between /d/ and /view)
        file_id = "1Zenpa8iO8fWWv2zNBrlUDIpY0kc3yDRj"  # Random Forest model from Google Drive - Updated
        
        # Google Drive direct download URL (try multiple formats)
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        model_path = os.path.join(self.model_dir, 'random_forest.pkl')
        
        try:
            print(f"Downloading Random Forest model from Google Drive...")
            print(f"File ID: {file_id}")
            print(f"Download URL: {url}")
            
            # First request to get the virus scan warning page
            response = requests.get(url, timeout=30)
            print(f"Response status: {response.status_code}")
            
            # Check if we got the virus scan warning page
            if "virus scan warning" in response.text.lower():
                print("Detected virus scan warning, trying alternative download method...")
                
                # Try alternative URL format
                alt_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
                print(f"Trying alternative URL: {alt_url}")
                
                # Try to get the direct download link from the sharing page
                try:
                    alt_response = requests.get(alt_url, timeout=30)
                    if alt_response.status_code == 200:
                        # Look for direct download link in the page
                        import re
                        download_pattern = r'href="([^"]*uc[^"]*export=download[^"]*)"'
                        download_match = re.search(download_pattern, alt_response.text)
                        
                        if download_match:
                            direct_url = download_match.group(1)
                            if not direct_url.startswith('http'):
                                direct_url = 'https://drive.google.com' + direct_url
                            print(f"Found direct download URL: {direct_url}")
                            response = requests.get(direct_url, stream=True, timeout=30)
                            print(f"Direct download response status: {response.status_code}")
                        else:
                            print("Could not find direct download link")
                            return False
                    else:
                        print(f"Alternative URL failed with status: {alt_response.status_code}")
                        return False
                except Exception as e:
                    print(f"Error with alternative download method: {e}")
                    return False
            else:
                print("No virus scan warning detected, proceeding with download")
            
            response.raise_for_status()
            
            # Download the file
            total_size = 0
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            print(f"Random Forest model downloaded successfully ({total_size:,} bytes)")
            
            # Load the downloaded model
            self.model = joblib.load(model_path)
            print("Random Forest model loaded successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Network error downloading Random Forest model: {e}")
            print("Please check your internet connection and Google Drive file sharing settings")
            return False
        except Exception as e:
            print(f"Error downloading Random Forest model from Google Drive: {e}")
            print("Please check your Google Drive file ID and internet connection")
            return False
    
    def preprocess_features(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input features for model prediction"""
        processed_data = input_data.copy()
        
        # Define categorical columns based on your dataset
        categorical_columns = [
            'manufacturer', 'model', 'fuel', 'title_status', 
            'transmission', 'drive', 'type', 'paint_color', 'state'
        ]
        
        # Handle categorical variables with consistent encoding
        for col in categorical_columns:
            if col in processed_data.columns:
                # Convert to string and handle missing values
                processed_data[col] = processed_data[col].astype(str).fillna('unknown').str.lower()
                
                # Use consistent encoding based on common values
                if col == 'manufacturer':
                    manufacturer_map = {
                        'ford': 0, 'chevrolet': 1, 'toyota': 2, 'honda': 3, 'bmw': 4, 
                        'mercedes': 5, 'audi': 6, 'nissan': 7, 'hyundai': 8, 'other': 9
                    }
                    processed_data[col] = processed_data[col].map(manufacturer_map).fillna(9)
                elif col == 'fuel':
                    fuel_map = {'gas': 0, 'diesel': 1, 'hybrid': 2, 'electric': 3, 'other': 4}
                    processed_data[col] = processed_data[col].map(fuel_map).fillna(4)
                elif col == 'title_status':
                    title_map = {'clean': 0, 'lien': 1, 'rebuilt': 2, 'salvage': 3, 'missing': 4, 'other': 5}
                    processed_data[col] = processed_data[col].map(title_map).fillna(5)
                elif col == 'transmission':
                    trans_map = {'automatic': 0, 'manual': 1, 'other': 2}
                    processed_data[col] = processed_data[col].map(trans_map).fillna(2)
                elif col == 'drive':
                    drive_map = {'fwd': 0, 'rwd': 1, '4wd': 2, 'awd': 3, 'other': 4}
                    processed_data[col] = processed_data[col].map(drive_map).fillna(4)
                elif col == 'type':
                    type_map = {'sedan': 0, 'suv': 1, 'truck': 2, 'coupe': 3, 'hatchback': 4, 'convertible': 5, 'wagon': 6, 'other': 7}
                    processed_data[col] = processed_data[col].map(type_map).fillna(7)
                elif col == 'paint_color':
                    color_map = {'black': 0, 'white': 1, 'silver': 2, 'red': 3, 'blue': 4, 'green': 5, 'other': 6}
                    processed_data[col] = processed_data[col].map(color_map).fillna(6)
                elif col == 'state':
                    # Simple hash-based encoding for states
                    processed_data[col] = processed_data[col].apply(lambda x: hash(x) % 50)
                elif col == 'model':
                    # Simple hash-based encoding for models
                    processed_data[col] = processed_data[col].apply(lambda x: hash(x) % 100)
        
        # Ensure all columns are numeric
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
        
        return processed_data
    
    def predict_condition(self, input_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """Make prediction using Decision Tree model"""
        if self.model is None:
            return "Error: Decision Tree model not loaded", {}
        
        try:
            # Create DataFrame from input data
            input_df = pd.DataFrame([input_data])
            
            # Preprocess the data
            processed_data = self.preprocess_features(input_df)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)[0]
                
                # Create probability dictionary with condition names
                proba_dict = {}
                for i, prob in enumerate(probabilities):
                    condition_name = self.condition_mapping.get(i, f"Class_{i}")
                    proba_dict[condition_name] = float(prob)
                
                # Map the prediction to condition name
                predicted_condition = self.condition_mapping.get(prediction, f"Class_{prediction}")
                
            else:
                predicted_condition = self.condition_mapping.get(prediction, f"Class_{prediction}")
                proba_dict = {predicted_condition: 1.0}
            
            return predicted_condition, proba_dict
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None, None
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get information about a model"""
        info = {}
        
        if hasattr(model, 'n_estimators'):
            info['estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth
        if hasattr(model, 'classes_'):
            info['classes'] = list(model.classes_)
        
        return info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Determine model type based on the model object
            model_type = "Random Forest" if hasattr(self.model, 'n_estimators') else "Decision Tree"
            
            info = {
                "model_type": model_type,
                "is_loaded": True,
                "has_probabilities": hasattr(self.model, 'predict_proba')
            }
            
            # Add model-specific information if available
            if hasattr(self.model, 'n_features_in_'):
                info["features_count"] = self.model.n_features_in_
            if hasattr(self.model, 'n_classes_'):
                info["classes_count"] = self.model.n_classes_
            if hasattr(self.model, 'n_estimators'):
                info["n_estimators"] = self.model.n_estimators
            if hasattr(self.model, 'max_depth'):
                info["max_depth"] = self.model.max_depth
                
            return info
            
        except Exception as e:
            return {"error": f"Error getting model info: {str(e)}"}
    
    def get_condition_interpretation(self, condition: str) -> Tuple[str, str]:
        """Get interpretation and styling for a condition"""
        condition_lower = condition.lower()
        
        interpretations = {
            'new': ("This vehicle is brand new with no previous use or wear.", "success"),
            'like new': ("This vehicle is in like-new condition with minimal wear and excellent maintenance history.", "success"),
            'excellent': ("This vehicle is in excellent condition with very minor wear expected for its age.", "info"),
            'good': ("This vehicle is in good condition with minor wear expected for its age and mileage.", "info"),
            'fair': ("This vehicle shows signs of wear but may still be a reasonable purchase with proper inspection.", "warning"),
            'salvage': ("This vehicle has significant damage or issues. Consider a thorough inspection and be cautious about purchase.", "error")
        }
        
        return interpretations.get(condition_lower, ("This vehicle's condition requires further evaluation.", "info"))
    
    def analyze_market_price(self, input_data: Dict[str, Any], predicted_condition: str) -> Dict[str, Any]:
        """Analyze market price and provide insights"""
        price = input_data.get('price', 0)
        year = input_data.get('year', 2020)
        manufacturer = input_data.get('manufacturer', '').lower()
        model = input_data.get('model', '').lower()
        odometer = input_data.get('odometer', 0)
        
        # Market analysis based on condition
        condition_multipliers = {
            'new': 1.0,
            'like new': 0.95,
            'excellent': 0.85,
            'good': 0.75,
            'fair': 0.60,
            'salvage': 0.40
        }
        
        # Age depreciation (rough estimate)
        current_year = 2024
        age = current_year - year
        age_depreciation = max(0.3, 1.0 - (age * 0.08))  # 8% per year, minimum 30%
        
        # Mileage impact
        mileage_impact = 1.0
        if odometer > 200000:
            mileage_impact = 0.5
        elif odometer > 150000:
            mileage_impact = 0.7
        elif odometer > 100000:
            mileage_impact = 0.85
        elif odometer > 50000:
            mileage_impact = 0.95
        
        # Calculate estimated market value
        base_value = price / (condition_multipliers.get(predicted_condition.lower(), 0.75))
        estimated_value = base_value * age_depreciation * mileage_impact
        
        # Price analysis
        price_analysis = {
            'current_price': price,
            'estimated_market_value': round(estimated_value, 2),
            'price_vs_market': round((price / estimated_value - 1) * 100, 1) if estimated_value > 0 else 0,
            'condition_multiplier': condition_multipliers.get(predicted_condition.lower(), 0.75),
            'age_depreciation': round(age_depreciation * 100, 1),
            'mileage_impact': round(mileage_impact * 100, 1)
        }
        
        return price_analysis
    
    def get_vehicle_insights(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get important vehicle insights and facts"""
        year = input_data.get('year', 2020)
        manufacturer = input_data.get('manufacturer', '').lower()
        model = input_data.get('model', '').lower()
        odometer = input_data.get('odometer', 0)
        owners = input_data.get('owners', 1)
        fuel = input_data.get('fuel', '').lower()
        transmission = input_data.get('transmission', '').lower()
        
        insights = {
            'age': 2024 - year,
            'annual_mileage': round(odometer / max(1, 2024 - year), 0),
            'ownership_stability': 'Single Owner' if owners == 1 else f'{owners} Previous Owners',
            'fuel_efficiency': self._get_fuel_efficiency_rating(fuel, manufacturer, model),
            'reliability_rating': self._get_reliability_rating(manufacturer, model),
            'maintenance_tips': self._get_maintenance_tips(manufacturer, model, year),
            'market_trends': self._get_market_trends(manufacturer, model),
            'red_flags': self._get_red_flags(input_data)
        }
        
        return insights
    
    def _get_fuel_efficiency_rating(self, fuel: str, manufacturer: str, model: str) -> str:
        """Get fuel efficiency rating"""
        if fuel == 'electric':
            return 'Excellent (Electric)'
        elif fuel == 'hybrid':
            return 'Very Good (Hybrid)'
        elif fuel == 'gas':
            if manufacturer in ['toyota', 'honda', 'hyundai']:
                return 'Good (Gas)'
            else:
                return 'Average (Gas)'
        elif fuel == 'diesel':
            return 'Good (Diesel)'
        else:
            return 'Unknown'
    
    def _get_reliability_rating(self, manufacturer: str, model: str) -> str:
        """Get reliability rating based on manufacturer and model"""
        reliable_brands = ['toyota', 'honda', 'lexus', 'mazda', 'subaru']
        average_brands = ['ford', 'chevrolet', 'nissan', 'hyundai', 'kia']
        
        if manufacturer in reliable_brands:
            return 'High Reliability'
        elif manufacturer in average_brands:
            return 'Average Reliability'
        else:
            return 'Variable Reliability'
    
    def _get_maintenance_tips(self, manufacturer: str, model: str, year: int) -> list:
        """Get maintenance tips based on vehicle"""
        tips = []
        
        if year < 2015:
            tips.append("Check for rust and corrosion")
            tips.append("Inspect suspension components")
        
        if manufacturer in ['bmw', 'mercedes', 'audi']:
            tips.append("Regular premium maintenance recommended")
            tips.append("Check for electronic system issues")
        
        if manufacturer in ['toyota', 'honda']:
            tips.append("Generally low maintenance costs")
            tips.append("Regular oil changes sufficient")
        
        tips.extend([
            "Check tire condition and alignment",
            "Inspect brake system",
            "Verify all lights and electronics work"
        ])
        
        return tips[:5]  # Return top 5 tips
    
    def _get_market_trends(self, manufacturer: str, model: str) -> str:
        """Get market trends for the vehicle"""
        popular_models = ['camry', 'accord', 'civic', 'corolla', 'f-150', 'silverado']
        
        if model in popular_models:
            return f"High demand for {model.title()} - good resale value"
        elif manufacturer in ['toyota', 'honda']:
            return "Strong brand reputation - stable market value"
        else:
            return "Standard market conditions"
    
    def _get_red_flags(self, input_data: Dict[str, Any]) -> list:
        """Identify potential red flags"""
        red_flags = []
        
        year = input_data.get('year', 2020)
        odometer = input_data.get('odometer', 0)
        owners = input_data.get('owners', 1)
        title_status = input_data.get('title_status', '').lower()
        
        if year < 2010:
            red_flags.append("Vehicle is over 14 years old")
        
        if odometer > 200000:
            red_flags.append("Very high mileage - potential mechanical issues")
        
        if owners > 3:
            red_flags.append("Multiple previous owners - check maintenance history")
        
        if title_status in ['salvage', 'rebuilt']:
            red_flags.append("Salvage/Rebuilt title - significant damage history")
        
        if title_status == 'lien':
            red_flags.append("Lien on title - verify ownership")
        
        return red_flags
    
    def get_condition_css_class(self, condition: str) -> str:
        """Get CSS class for condition styling"""
        condition_lower = condition.lower().replace(' ', '-')
        return f"condition-{condition_lower}" if condition_lower in ['new', 'like-new', 'excellent', 'good', 'fair', 'salvage'] else "condition-good"
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate input data"""
        required_fields = ['price', 'year', 'manufacturer', 'model', 'fuel', 'transmission', 'drive', 'type', 'state']
        
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return False, f"Missing required field: {field}"
        
        # Validate numeric fields
        if input_data['price'] <= 0:
            return False, "Price must be greater than 0"
        
        if input_data['year'] < 1900 or input_data['year'] > 2025:
            return False, "Year must be between 1900 and 2025"
        
        if input_data['odometer'] < 0:
            return False, "Odometer must be non-negative"
        
        if input_data['owners'] < 0:
            return False, "Number of owners must be non-negative"
        
        return True, "Valid"
    
    def create_input_dataframe(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Create a DataFrame from input data"""
        return pd.DataFrame([{
            "price": input_data['price'],
            "year": input_data['year'],
            "manufacturer": input_data['manufacturer'],
            "model": input_data['model'],
            "cylinders": input_data.get('cylinders', 4),
            "fuel": input_data['fuel'],
            "odometer": input_data.get('odometer', 0),
            "title_status": input_data.get('title_status', 'clean'),
            "transmission": input_data['transmission'],
            "drive": input_data['drive'],
            "type": input_data['type'],
            "paint_color": input_data.get('paint_color', 'white'),
            "state": input_data['state'],
            "owners": input_data.get('owners', 1),
            "location_cluster": input_data.get('location_cluster', 0)
        }])


if __name__ == "__main__":
    # Example usage
    predictor = VehicleConditionPredictor()
    models = predictor.load_models()
    print(f"Loaded {len(models)} models: {list(models.keys())}")
