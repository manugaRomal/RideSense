"""
RideSense UI Module
Contains all the Streamlit user interface components
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
from typing import Dict, Any, List
from .logic import VehicleConditionPredictor

# Suppress Streamlit deprecation warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")

class RideSenseUI:
    """Main UI class for the RideSense application"""
    
    def __init__(self):
        self.predictor = VehicleConditionPredictor()
        self.setup_page_config()
        self.setup_css()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="RideSense - Vehicle Condition Predictor", 
            page_icon="🚗",
            layout="wide"
        )
    
    def setup_css(self):
        """Setup custom CSS styling"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                font-weight: bold;
                text-align: center;
                color: #1f77b4;
                margin-bottom: 2rem;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #666;
                text-align: center;
                margin-bottom: 3rem;
            }
            .prediction-box {
                background-color: #f0f2f6;
                padding: 2rem;
                border-radius: 10px;
                border-left: 5px solid #1f77b4;
                margin: 2rem 0;
            }
            .condition-new {
                color: #28a745;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .condition-like-new {
                color: #20c997;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .condition-excellent {
                color: #17a2b8;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .condition-good {
                color: #6f42c1;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .condition-fair {
                color: #ffc107;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .condition-salvage {
                color: #dc3545;
                font-weight: bold;
                font-size: 1.5rem;
            }
            .model-info {
                background-color: #e8f4fd;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🚗 RideSense</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Vehicle Condition Prediction System</p>', unsafe_allow_html=True)
    
    def render_sidebar(self, models: Dict[str, Any]) -> tuple:
        """Render the sidebar with model selection and info"""
        with st.sidebar:
            st.header("🤖 Model Selection")
            
            # Model selection
            selected_model_name = st.selectbox(
                "Choose Model",
                list(models.keys()),
                help="Select which trained model to use for prediction"
            )
            
            selected_model = models[selected_model_name]
            
            # Model information
            st.markdown('<div class="model-info">', unsafe_allow_html=True)
            st.subheader("📊 Model Info")
            st.write(f"**Selected:** {selected_model_name}")
            
            model_info = self.predictor.get_model_info(selected_model)
            for key, value in model_info.items():
                st.write(f"**{key.title()}:** {value}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Available models list
            st.subheader("📋 Available Models")
            for model_name in models.keys():
                status = "🟢" if model_name == selected_model_name else "⚪"
                st.write(f"{status} {model_name}")
        
        return selected_model_name, selected_model
    
    def render_input_form(self) -> Dict[str, Any]:
        """Render the input form and return input data"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📝 Vehicle Specifications")
            
            # Basic Information
            st.subheader("Basic Information")
            price = st.number_input("Price ($)", min_value=100, max_value=100000, value=15000, help="Vehicle asking price")
            year = st.number_input("Year", min_value=1900, max_value=2025, value=2015, help="Manufacturing year")
            manufacturer = st.selectbox("Manufacturer", ["ford", "chevrolet", "toyota", "honda", "bmw", "mercedes", "audi", "nissan", "hyundai", "other"])
            model_input = st.text_input("Model", value="camry", help="Vehicle model (e.g., camry, accord, f150)")
            
            # Vehicle Specifications
            st.subheader("Vehicle Specifications")
            cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4, help="Number of engine cylinders")
            fuel = st.selectbox("Fuel Type", ["gas", "diesel", "hybrid", "electric", "other"])
            odometer = st.number_input("Odometer (miles)", min_value=0, max_value=500000, value=60000, help="Total mileage")
            title_status = st.selectbox("Title Status", ["clean", "lien", "rebuilt", "salvage", "missing", "other"])
            transmission = st.selectbox("Transmission", ["automatic", "manual", "other"])
            drive = st.selectbox("Drive Type", ["fwd", "rwd", "4wd", "awd", "other"])
            vehicle_type = st.selectbox("Vehicle Type", ["sedan", "suv", "truck", "coupe", "hatchback", "convertible", "wagon", "other"])
            paint_color = st.selectbox("Paint Color", ["black", "white", "silver", "red", "blue", "green", "other"])
            
            # Ownership & Location
            st.subheader("Ownership & Location")
            state = st.text_input("State (e.g., CA)", value="ca")
            owners = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=1, help="Number of previous owners (fewer is generally better)")
            location_cluster = st.number_input("Location Cluster", min_value=0, max_value=100, value=0, help="Geographic location cluster identifier")
        
        return {
            "price": price,
            "year": year,
            "manufacturer": manufacturer,
            "model": model_input,
            "cylinders": cylinders,
            "fuel": fuel,
            "odometer": odometer,
            "title_status": title_status,
            "transmission": transmission,
            "drive": drive,
            "type": vehicle_type,
            "paint_color": paint_color,
            "state": state,
            "owners": owners,
            "location_cluster": location_cluster
        }
    
    def render_input_summary(self, input_data: Dict[str, Any]):
        """Render the input summary section"""
        st.header("📊 Input Summary")
        
        # Create input data DataFrame
        input_df = self.predictor.create_input_dataframe(input_data)
        st.dataframe(input_df, width='stretch')
    
    def render_prediction_button(self) -> bool:
        """Render the prediction button and return if clicked"""
        st.markdown("---")
        return st.button("🔮 Predict Vehicle Condition", type="primary", use_container_width=True)
    
    def render_prediction_results(self, prediction: str, probabilities: Dict[str, float], model_name: str):
        """Render the prediction results"""
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Main prediction box
        condition_class = self.predictor.get_condition_css_class(prediction)
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Predicted Condition: <span class="{condition_class}">{prediction}</span></h2>
            <h3>Model Used: {model_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction probabilities
        if probabilities:
            st.subheader("📈 Prediction Confidence")
            
            # Create probability chart
            proba_df = pd.DataFrame(list(probabilities.items()), columns=["Condition", "Probability"])
            proba_df = proba_df.sort_values("Probability", ascending=False)
            
            # Bar chart
            fig = px.bar(
                proba_df, 
                x="Condition", 
                y="Probability", 
                color="Probability",
                color_continuous_scale="RdYlGn",
                text_auto='.2%',
                title="Condition Probability Distribution"
            )
            fig.update_traces(textfont_size=12, textangle=0, textposition="outside")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence score
            max_prob = max(probabilities.values())
            confidence_color = "green" if max_prob > 0.7 else "orange" if max_prob > 0.5 else "red"
            st.markdown(f"**Confidence Score:** <span style='color: {confidence_color}'>{max_prob:.1%}</span>", unsafe_allow_html=True)
    
    def render_interpretation(self, prediction: str):
        """Render the interpretation section"""
        st.subheader("Interpretation")
        interpretation, message_type = self.predictor.get_condition_interpretation(prediction)
        
        if message_type == "success":
            st.success(interpretation)
        elif message_type == "info":
            st.info(interpretation)
        elif message_type == "warning":
            st.warning(interpretation)
        elif message_type == "error":
            st.error(interpretation)
        else:
            st.info(interpretation)
    
    def render_market_analysis(self, input_data: Dict[str, Any], prediction: str):
        """Render market price analysis"""
        st.subheader("Market Price Analysis")
        
        price_analysis = self.predictor.analyze_market_price(input_data, prediction)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${price_analysis['current_price']:,.0f}"
            )
        
        with col2:
            st.metric(
                label="Estimated Market Value",
                value=f"${price_analysis['estimated_market_value']:,.0f}"
            )
        
        with col3:
            price_diff = price_analysis['price_vs_market']
            if price_diff > 0:
                st.metric(
                    label="Price vs Market",
                    value=f"+{price_diff:.1f}%",
                    delta=f"Above market"
                )
            else:
                st.metric(
                    label="Price vs Market",
                    value=f"{price_diff:.1f}%",
                    delta=f"Below market"
                )
        
        # Price breakdown
        with st.expander("Price Breakdown Details"):
            st.write(f"**Condition Impact:** {price_analysis['condition_multiplier']*100:.0f}% of base value")
            st.write(f"**Age Depreciation:** {price_analysis['age_depreciation']:.1f}% remaining value")
            st.write(f"**Mileage Impact:** {price_analysis['mileage_impact']:.1f}% of value retained")
    
    def render_vehicle_insights(self, input_data: Dict[str, Any]):
        """Render vehicle insights and facts"""
        st.subheader("Vehicle Insights & Facts")
        
        insights = self.predictor.get_vehicle_insights(input_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"• Vehicle Age: {insights['age']} years")
            st.write(f"• Annual Mileage: {insights['annual_mileage']:,} miles/year")
            st.write(f"• Ownership: {insights['ownership_stability']}")
            st.write(f"• Fuel Efficiency: {insights['fuel_efficiency']}")
            st.write(f"• Reliability: {insights['reliability_rating']}")
        
        with col2:
            st.write("**Market Information:**")
            st.write(f"• Market Trend: {insights['market_trends']}")
            
            if insights['red_flags']:
                st.write("**Potential Concerns:**")
                for flag in insights['red_flags']:
                    st.write(f"⚠️ {flag}")
            else:
                st.write("**No major concerns identified**")
        
        # Maintenance tips
        with st.expander("Maintenance Tips"):
            st.write("**Recommended Checks:**")
            for tip in insights['maintenance_tips']:
                st.write(f"• {tip}")
    
    def render_similar_vehicles(self, input_data: Dict[str, Any]):
        """Render similar vehicles analysis"""
        st.subheader("Similar Vehicles Analysis")
        
        # Generate sample similar vehicles based on input
        year = input_data.get('year', 2020)
        manufacturer = input_data.get('manufacturer', 'toyota')
        model = input_data.get('model', 'camry')
        price = input_data.get('price', 15000)
        
        # Create sample similar vehicles data
        similar_vehicles = []
        base_price = price
        
        for i in range(3):
            similar_vehicles.append({
                "Year": year + (i-1),
                "Mileage": f"{input_data.get('odometer', 75000) + (i-1)*10000:,}",
                "Price": f"${base_price + (i-1)*2000:,}",
                "Condition": ["Good", "Excellent", "Fair"][i],
                "Location": ["CA", "NY", "TX"][i]
            })
        
        # Display as table
        import pandas as pd
        similar_df = pd.DataFrame(similar_vehicles)
        st.dataframe(similar_df, width='stretch')
        
        st.info("These are estimated similar vehicles based on your criteria. Actual market prices may vary.")
    
    def render_model_comparison(self, input_data: Dict[str, Any], models: Dict[str, Any]):
        """Render the model comparison section"""
        if len(models) > 1:
            with st.expander("Compare All Models"):
                st.subheader("Model Comparison")
                
                comparison_data = self.predictor.compare_models(
                    self.predictor.create_input_dataframe(input_data)
                )
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width='stretch')
                    
                    # Agreement analysis
                    predictions = [row["Prediction"] for row in comparison_data]
                    most_common = max(set(predictions), key=predictions.count)
                    agreement = predictions.count(most_common) / len(predictions)
                    
                    st.write(f"**Model Agreement:** {agreement:.1%} ({predictions.count(most_common)}/{len(predictions)} models predict '{most_common}')")
    
    def render_footer(self):
        """Render the footer"""
        st.markdown("---")
        st.markdown("🚗 **RideSense** - ML-Powered Vehicle Condition Predictor")
        st.markdown("Built with Streamlit and Scikit-learn")
    
    def render_error_message(self, message: str):
        """Render error message"""
        st.error(message)
    
    def render_info_message(self, message: str):
        """Render info message"""
        st.info(message)
    
    def render_spinner(self, message: str):
        """Render loading spinner"""
        return st.spinner(message)
    
    def run(self):
        """Main method to run the UI"""
        # Render header
        self.render_header()
        
        # Load models
        with self.render_spinner("Loading trained models..."):
            models = self.predictor.load_models()
        
        if not models:
            self.render_error_message("❌ No trained models found! Please ensure your model files are in the 'model' directory.")
            self.render_info_message("Expected files: random_forest.pkl, decision_tree.pkl, gradient_boosting.pkl, etc.")
            return
        
        # Render sidebar and get selected model
        selected_model_name, selected_model = self.render_sidebar(models)
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Render input form
            input_data = self.render_input_form()
        
        with col2:
            # Render input summary
            self.render_input_summary(input_data)
            
            # Render prediction button
            predict_button = self.render_prediction_button()
            
            if predict_button:
                # Validate input data
                is_valid, error_message = self.predictor.validate_input_data(input_data)
                
                if not is_valid:
                    self.render_error_message(f"❌ {error_message}")
                    return
                
                # Make prediction
                with self.render_spinner("Analyzing vehicle specifications..."):
                    prediction, probabilities = self.predictor.predict_condition(
                        selected_model, 
                        self.predictor.create_input_dataframe(input_data)
                    )
                
                if prediction is not None:
                    # Render prediction results
                    self.render_prediction_results(prediction, probabilities, selected_model_name)
                    
                    # Render interpretation
                    self.render_interpretation(prediction)
                    
                    # Render market analysis
                    self.render_market_analysis(input_data, prediction)
                    
                    # Render vehicle insights
                    self.render_vehicle_insights(input_data)
                    
                    # Render similar vehicles
                    self.render_similar_vehicles(input_data)
                    
                    # Render model comparison
                    self.render_model_comparison(input_data, models)
        
        # Render footer
        self.render_footer()


def main():
    """Main function to run the UI"""
    ui = RideSenseUI()
    ui.run()


if __name__ == "__main__":
    main()
