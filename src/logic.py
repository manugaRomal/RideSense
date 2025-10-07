"""
RideSense Logic Module
Contains all the business logic for vehicle condition prediction
"""
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple, Optional, List

# Suppress XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

class VehicleConditionPredictor:
    """Main class for vehicle condition prediction logic"""
    
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.model = None
        self.label_encoders = None
        self.target_encoder = None
        self.categorical_cols = None
        self.numeric_cols = None
        self.condition_mapping = {
            0: "New",
            1: "Like New", 
            2: "Excellent",
            3: "Good",
            4: "Fair",
            5: "Salvage"
        }
        self.manufacturer_models = {
            "acura": ["other", "mdx sh-", "tl", "mdx"],
            "alfa-romeo": ["romeo stelvio ti", "other"],
            "audi": ["other", "a4", "s5 plus 4d"],
            "bmw": ["other", "3 series", "328i", "x5", "3 series 330i xdrive", "x5 xdrive35i"],
            "buick": ["other", "enclave"],
            "cadillac": ["other", "escape"],
            "chevrolet": ["silverado 1500", "colorado", "corvette", "camry", "trailblazer", "other", "tahoe", "traverse", "impala", "trax", "malibu", "suburban", "equinox", "cruze"],
            "chrysler": ["town & country", "3500", "other", "1500"],
            "dodge": ["other", "1500", "charger", "f150", "durango", "1500 big horn", "caravan", "journey", "challenger r/t 2d", None],
            "fiat": ["other"],
            "ford": ["f-150", "other", "f150", "f150 supercrew", "f-250 duty", "mustang", "expedition", "wrangler", "focus", "escape", "taurus", "edge", "explorer", "santa fe", "transit", "fusion", "econoline", "flex"],
            "gmc": ["silverado 1500", "sierra 2500hd", "acadia", "other", "yukon", "terrain", "sierra"],
            "honda": ["odyssey", "civic", "cr-v", "other", "accord", "pilot", "fit"],
            "hyundai": ["sonata", "elantra", "other"],
            "infiniti": ["other"],
            "jaguar": ["other"],
            "jeep": ["cherokee", "wrangler unlimited", "wrangler", "other", "cherokee laredo", "liberty", "patriot", "compass"],
            "kia": ["other", "optima", "soul", "sorento", "forester"],
            "lexus": ["other", "es 350", "rx 350"],
            "lincoln": ["other"],
            "mazda": ["mx-5 miata", "other", "mazda3"],
            "mercedes-benz": ["other", "c-class", "benz e350", "gla-class gla 45"],
            "mercury": ["other"],
            "mini": ["other"],
            "mitsubishi": ["outlander", "other"],
            "nissan": ["other", "elantra", "altima", "frontier", "pathfinder", "durango", "rogue", "altima 2.5 s", "versa", "maxima"],
            "other": ["other"],
            "pontiac": ["other"],
            "porsche": ["other"],
            "rover": ["other"],
            "saturn": ["other"],
            "subaru": ["impreza", "forester", "legacy", "other", "outback"],
            "tesla": ["other"],
            "toyota": ["tundra", "tacoma", "other", "rav4", "camry", "corolla", "4runner", "prius", None, "sienna"],
            "unknown": ["scion im 4d", "other", "genesis g80 3.8 4d", "scion xb"],
            "volkswagen": ["jetta", "passat", "other", "tiguan"],
            "volvo": ["other"]
        }
        self.load_model()
    
    def load_model(self) -> bool:
        """Load Random Forest model with label encoders, fallback to Gradient Boosting"""
        print("🔍 Starting model loading process...")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            print(f"📁 Creating model directory: {self.model_dir}")
            os.makedirs(self.model_dir)
        else:
            print(f"📁 Model directory exists: {self.model_dir}")
        
        # Try Random Forest first
        rf_model_path = os.path.join(self.model_dir, 'random_forest.pkl')
        print(f"🔍 Checking for Random Forest model at: {rf_model_path}")
        
        if os.path.exists(rf_model_path):
            try:
                print("📦 Loading Random Forest model bundle...")
                bundle = joblib.load(rf_model_path)
                
                # Extract components
                self.model = bundle["model"]
                self.label_encoders = bundle["label_encoders"]
                self.target_encoder = bundle["target_encoder"]
                self.categorical_cols = bundle["categorical_cols"]
                self.numeric_cols = bundle["numeric_cols"]
                
                # Log model details
                print("✅ Random Forest model loaded successfully!")
                print(f"   📊 Model type: {type(self.model).__name__}")
                print(f"   🔢 Categorical columns: {len(self.categorical_cols)} - {self.categorical_cols}")
                print(f"   🔢 Numeric columns: {len(self.numeric_cols)} - {self.numeric_cols}")
                print(f"   🎯 Target encoder classes: {len(self.target_encoder.classes_)} - {list(self.target_encoder.classes_)}")
                
                # Log model parameters if available
                if hasattr(self.model, 'n_estimators'):
                    print(f"   🌳 Number of estimators: {self.model.n_estimators}")
                if hasattr(self.model, 'max_depth'):
                    print(f"   📏 Max depth: {self.model.max_depth}")
                if hasattr(self.model, 'random_state'):
                    print(f"   🎲 Random state: {self.model.random_state}")
                
                return True
            except Exception as e:
                print(f"❌ Error loading Random Forest model: {e}")
                print(f"   📋 Full error: {str(e)}")
        
        # Fallback to Gradient Boosting
        print("⚠️  Random Forest not available, trying Gradient Boosting...")
        gb_model_path = os.path.join(self.model_dir, 'gradient_boosting.pkl')
        print(f"🔍 Checking for Gradient Boosting model at: {gb_model_path}")
        
        if os.path.exists(gb_model_path):
            try:
                print("📦 Loading Gradient Boosting model...")
                self.model = joblib.load(gb_model_path)
                
                # Log model details
                print("✅ Gradient Boosting model loaded successfully!")
                print(f"   📊 Model type: {type(self.model).__name__}")
                
                # Log model parameters if available
                if hasattr(self.model, 'n_estimators'):
                    print(f"   🌳 Number of estimators: {self.model.n_estimators}")
                if hasattr(self.model, 'learning_rate'):
                    print(f"   📈 Learning rate: {self.model.learning_rate}")
                if hasattr(self.model, 'max_depth'):
                    print(f"   📏 Max depth: {self.model.max_depth}")
                
                print("⚠️  Note: Using fallback model without label encoders")
                return True
            except Exception as e:
                print(f"❌ Error loading Gradient Boosting model: {e}")
                print(f"   📋 Full error: {str(e)}")
                return False
        else:
            print("❌ Neither Random Forest nor Gradient Boosting model found")
            print("   📁 Available files in model directory:")
            if os.path.exists(self.model_dir):
                for file in os.listdir(self.model_dir):
                    file_path = os.path.join(self.model_dir, file)
                    file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                    print(f"      - {file} ({file_size:,} bytes)")
            print("   💡 Please ensure random_forest.pkl or gradient_boosting.pkl is in the model/ directory")
            return False
    
    
    def preprocess_features(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input features for model prediction using label encoders"""
        processed_data = input_data.copy()
        
        # Use label encoders from the trained model (Random Forest)
        if self.label_encoders is not None and self.categorical_cols is not None:
            print("   🔧 Using Random Forest label encoders...")
            for col in self.categorical_cols:
                if col in processed_data.columns:
                    le = self.label_encoders[col]
                    # Transform new data to the same encoding as training
                    processed_data[col] = le.transform(processed_data[col].astype(str))
        else:
            # Fallback to manual encoding for Gradient Boosting
            print("   🔧 Using manual encoding for Gradient Boosting fallback...")
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
                        fuel_map = {'gasoline': 0, 'other': 1, 'diesel': 2, 'hybrid': 3, 'unknown': 4, 'electric': 5}
                        processed_data[col] = processed_data[col].map(fuel_map).fillna(1)
                    elif col == 'title_status':
                        title_map = {'clean': 0, 'rebuilt': 1, 'lien': 2, 'other': 3, 'salvage': 4, 'missing': 5, 'parts only': 6}
                        processed_data[col] = processed_data[col].map(title_map).fillna(3)
                    elif col == 'transmission':
                        trans_map = {'other': 0, 'automatic': 1, 'manual': 2, 'unknown': 3}
                        processed_data[col] = processed_data[col].map(trans_map).fillna(0)
                    elif col == 'drive':
                        drive_map = {'unknown': 0, 'rwd': 1, '4wd': 2, 'fwd': 3}
                        processed_data[col] = processed_data[col].map(drive_map).fillna(0)
                    elif col == 'type':
                        type_map = {'pickup': 0, 'other': 1, 'unknown': 2, 'coupe': 3, 'suv': 4, 'hatchback': 5, 'van': 6, 'sedan': 7, 'offroad': 8, 'bus': 9, 'convertible': 10, 'wagon': 11}
                        processed_data[col] = processed_data[col].map(type_map).fillna(1)
                    elif col == 'paint_color':
                        color_map = {'white': 0, 'blue': 1, 'red': 2, 'black': 3, 'silver': 4, 'grey': 5, 'unknown': 6, 'brown': 7, 'other': 8, 'green': 9, 'custom': 10}
                        processed_data[col] = processed_data[col].map(color_map).fillna(8)
                    elif col == 'state':
                        # Simple hash-based encoding for states
                        processed_data[col] = processed_data[col].apply(lambda x: hash(x) % 50)
                    elif col == 'model':
                        # Simple hash-based encoding for models
                        processed_data[col] = processed_data[col].apply(lambda x: hash(x) % 100)
        
        return processed_data
    
    def get_manufacturers(self) -> List[str]:
        """Get list of available manufacturers"""
        return list(self.manufacturer_models.keys())
    
    def get_models_for_manufacturer(self, manufacturer: str) -> List[str]:
        """Get list of available models for a given manufacturer"""
        manufacturer = manufacturer.lower()
        if manufacturer in self.manufacturer_models:
            # Filter out None values and return the list
            return [model for model in self.manufacturer_models[manufacturer] if model is not None]
        return ["other"]
    
    def predict_condition(self, input_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """Make prediction using loaded model with label encoders"""
        if self.model is None:
            print("❌ Error: No model loaded")
            return "Error: No model loaded", {}
        
        try:
            print("🔮 Starting prediction process...")
            print(f"   📊 Input data keys: {list(input_data.keys())}")
            
            # Create DataFrame from input data
            new_data = pd.DataFrame([input_data])
            print(f"   📋 DataFrame shape: {new_data.shape}")
            
            # Preprocess the data using label encoders
            print("🔧 Preprocessing data with label encoders...")
            processed_data = self.preprocess_features(new_data)
            print(f"   📊 Processed data shape: {processed_data.shape}")
            print(f"   🔢 Processed data columns: {list(processed_data.columns)}")
            
            # Get prediction probabilities
            print("🎯 Making prediction...")
            probs = self.model.predict_proba(processed_data)[0]  # get probabilities for first row
            
            # Handle different model types
            if self.target_encoder is not None:
                # Random Forest with target encoder
                print("   🎯 Using Random Forest target encoder...")
                classes = self.target_encoder.classes_
            else:
                # Gradient Boosting fallback - use condition mapping
                print("   🎯 Using Gradient Boosting condition mapping...")
                # Use the same classes as Random Forest to match probability array length
                classes = ['excellent', 'fair', 'good', 'like new', 'new', 'salvage', 'salvaged']
            
            print(f"   📈 Raw probabilities: {probs}")
            print(f"   🎯 Available classes: {list(classes)}")
            
            # Combine into a DataFrame
            prob_df = pd.DataFrame({
                "condition": classes,
                "probability": probs
            }).sort_values(by="probability", ascending=False)
            
            # Get predicted condition and confidence
            predicted_condition = prob_df.iloc[0]["condition"]
            confidence = prob_df.iloc[0]['probability']
            
            print(f"✅ Prediction completed!")
            print(f"   🎯 Predicted condition: {predicted_condition}")
            print(f"   📊 Confidence: {confidence:.2%}")
            print(f"   📈 Top 3 probabilities:")
            for i, (_, row) in enumerate(prob_df.head(3).iterrows()):
                print(f"      {i+1}. {row['condition']}: {row['probability']:.2%}")
            
            # Create probability dictionary
            proba_dict = {}
            for _, row in prob_df.iterrows():
                proba_dict[row["condition"]] = float(row["probability"])
            
            return predicted_condition, proba_dict
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            print(f"   📋 Full error details: {str(e)}")
            import traceback
            print(f"   🔍 Traceback: {traceback.format_exc()}")
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
        """Get information about the loaded model (Random Forest or Gradient Boosting)"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Determine model type based on the model object
            if hasattr(self.model, 'n_estimators') and hasattr(self.model, 'learning_rate'):
                model_type = "Gradient Boosting"
            elif hasattr(self.model, 'n_estimators'):
                model_type = "Random Forest"
            else:
                model_type = "Unknown"
            
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
            if hasattr(self.model, 'learning_rate'):
                info["learning_rate"] = self.model.learning_rate
                
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
        
        # Validate categorical values
        valid_cylinders = [8, 6, 4, 5, 3, 10, 12]
        if input_data.get('cylinders') not in valid_cylinders:
            return False, f"Cylinders must be one of: {valid_cylinders}"
        
        valid_fuel = ["gasoline", "other", "diesel", "hybrid", "unknown", "electric"]
        if input_data.get('fuel') not in valid_fuel:
            return False, f"Fuel must be one of: {valid_fuel}"
        
        valid_title_status = ["clean", "rebuilt", "lien", "other", "salvage", "missing", "parts only"]
        if input_data.get('title_status') not in valid_title_status:
            return False, f"Title status must be one of: {valid_title_status}"
        
        valid_transmission = ["other", "automatic", "manual", "unknown"]
        if input_data.get('transmission') not in valid_transmission:
            return False, f"Transmission must be one of: {valid_transmission}"
        
        valid_drive = ["unknown", "rwd", "4wd", "fwd"]
        if input_data.get('drive') not in valid_drive:
            return False, f"Drive must be one of: {valid_drive}"
        
        valid_type = ["pickup", "other", "unknown", "coupe", "suv", "hatchback", "van", "sedan", "offroad", "bus", "convertible", "wagon"]
        if input_data.get('type') not in valid_type:
            return False, f"Vehicle type must be one of: {valid_type}"
        
        valid_paint_color = ["white", "blue", "red", "black", "silver", "grey", "unknown", "brown", "other", "green", "custom"]
        if input_data.get('paint_color') not in valid_paint_color:
            return False, f"Paint color must be one of: {valid_paint_color}"
        
        valid_states = ["al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "nc", "ne", "nv", "nj", "nm", "ny", "nh", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]
        if input_data.get('state') not in valid_states:
            return False, f"State must be one of: {valid_states}"
        
        valid_location_cluster = [9, 3, -1, 8, 2, 0, 7, 5, 4, 6, 1]
        if input_data.get('location_cluster') not in valid_location_cluster:
            return False, f"Location cluster must be one of: {valid_location_cluster}"
        
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
