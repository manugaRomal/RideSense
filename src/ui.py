"""
RideSense UI Module
Contains all the Streamlit user interface components
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
from typing import Dict, Any, List

# Reset Plotly theme after streamlit import
import plotly.io as pio
pio.templates.default = 'plotly'
from .logic import VehicleConditionPredictor

# Suppress Streamlit deprecation warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")

class RideSenseUI:
    """Main UI class for the RideSense application"""
    
    def __init__(self):
        self.predictor = None
        self.setup_css()
    
    def _ensure_predictor_loaded(self):
        """Lazy load the predictor to avoid blocking Streamlit startup"""
        if self.predictor is None:
            print("🔄 Lazy loading VehicleConditionPredictor...")
            self.predictor = VehicleConditionPredictor()
            print("✅ VehicleConditionPredictor loaded successfully!")
    
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
    
    def render_sidebar(self):
        """Render the sidebar with model info"""
        with st.sidebar:
            st.header("🤖 AI Analysis")
            
            # Model information
            st.markdown('<div class="model-info">', unsafe_allow_html=True)
            st.subheader("📊 System Status")
            
            self._ensure_predictor_loaded()
            model_info = self.predictor.get_model_info()
            if "error" in model_info:
                st.error(f"❌ {model_info['error']}")
            else:
                st.success("✅ AI model ready")
                model_type = model_info.get('model_type', 'AI')
                st.info(f"Using {model_type} algorithm for accurate predictions")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # System info
            st.subheader("ℹ️ About This System")
            st.write("• Analyzes vehicle condition using AI")
            st.write("• Provides market price insights")
            st.write("• Offers maintenance recommendations")
            st.write("• Shows similar vehicle comparisons")
    
    def render_input_form(self) -> Dict[str, Any]:
        """Render the input form and return input data"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📝 Vehicle Specifications")
            
            # Basic Information
            st.subheader("Basic Information")
            price = st.number_input("Price ($)", min_value=100, max_value=100000, value=15000, help="Vehicle asking price")
            year = st.number_input("Year", min_value=1900, max_value=2025, value=2015, help="Manufacturing year")
            
            # Dynamic manufacturer dropdown
            self._ensure_predictor_loaded()
            manufacturers = self.predictor.get_manufacturers()
            manufacturer = st.selectbox("Manufacturer", manufacturers, index=manufacturers.index("toyota") if "toyota" in manufacturers else 0)
            
            # Dynamic model dropdown based on selected manufacturer
            available_models = self.predictor.get_models_for_manufacturer(manufacturer)
            model_input = st.selectbox("Model", available_models, index=0 if available_models else 0)
            
            # Vehicle Specifications
            st.subheader("Vehicle Specifications")
            cylinders = st.selectbox("Cylinders", [8, 6, 4, 5, 3, 10, 12], index=2, help="Number of engine cylinders")
            fuel = st.selectbox("Fuel Type", ["gasoline", "other", "diesel", "hybrid", "unknown", "electric"], index=0)
            odometer = st.number_input("Odometer (miles)", min_value=0, max_value=500000, value=60000, help="Total mileage")
            title_status = st.selectbox("Title Status", ["clean", "rebuilt", "lien", "other", "salvage", "missing", "parts only"], index=0)
            transmission = st.selectbox("Transmission", ["other", "automatic", "manual", "unknown"], index=1)
            drive = st.selectbox("Drive Type", ["unknown", "rwd", "4wd", "fwd"], index=3)
            vehicle_type = st.selectbox("Vehicle Type", ["pickup", "other", "unknown", "coupe", "suv", "hatchback", "van", "sedan", "offroad", "bus", "convertible", "wagon"], index=7)
            paint_color = st.selectbox("Paint Color", ["white", "blue", "red", "black", "silver", "grey", "unknown", "brown", "other", "green", "custom"], index=0)
            
            # Ownership & Location
            st.subheader("Ownership & Location")
            state = st.selectbox("State", ["al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "nc", "ne", "nv", "nj", "nm", "ny", "nh", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"], index=4)
            owners = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=1, help="Number of previous owners (fewer is generally better)")
            location_cluster = st.selectbox("Location Cluster", [9, 3, -1, 8, 2, 0, 7, 5, 4, 6, 1], index=5, help="Geographic location cluster identifier")
        
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
        self._ensure_predictor_loaded()
        input_df = self.predictor.create_input_dataframe(input_data)
        st.dataframe(input_df, use_container_width=True)
    
    def render_prediction_button(self) -> bool:
        """Render the prediction button and return if clicked"""
        st.markdown("---")
        return st.button("🔮 Predict Vehicle Condition", type="primary", use_container_width=True)
    
    def render_prediction_results(self, prediction: str, probabilities: Dict[str, float], model_name: str):
        """Render the prediction results"""
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Main prediction box
        self._ensure_predictor_loaded()
        condition_class = self.predictor.get_condition_css_class(prediction)
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Predicted Condition: <span class="{condition_class}">{prediction}</span></h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction probabilities
        if probabilities:
            st.subheader("📈 Prediction Confidence")
            
            # Debug: Print probabilities to console
            print(f"🔍 DEBUG - Raw probabilities: {probabilities}")
            print(f"🔍 DEBUG - Probability values: {list(probabilities.values())}")
            print(f"🔍 DEBUG - Sum of probabilities: {sum(probabilities.values())}")
            
            # Create probability chart
            proba_df = pd.DataFrame(list(probabilities.items()), columns=["Condition", "Probability"])
            proba_df = proba_df.sort_values("Probability", ascending=False)
            
            # Debug: Print DataFrame
            print(f"🔍 DEBUG - DataFrame:\n{proba_df}")
            print(f"🔍 DEBUG - DataFrame Probability column: {proba_df['Probability'].tolist()}")
            
            # Bar chart using go.Figure for more control
            fig = go.Figure(data=[
                go.Bar(
                    x=proba_df["Condition"],
                    y=proba_df["Probability"],
                    marker_color=['#28a745', '#20c997', '#17a2b8', '#6f42c1', '#ffc107', '#dc3545', '#6c757d'],
                    text=[f"{prob:.1%}" for prob in proba_df["Probability"]],
                    textfont_size=12,
                    textangle=0,
                    textposition="outside",
                    # Ensure proper data binding
                    customdata=proba_df["Probability"],
                    hovertemplate="<b>%{x}</b><br>Probability: %{customdata:.1%}<extra></extra>"
                )
            ])
            
            fig.update_layout(
                title="Condition Probability Distribution",
                xaxis_title="Vehicle Condition",
                yaxis_title="Probability",
                yaxis=dict(
                    tickformat='.1%',
                    range=[0, 1],  # Force Y-axis to be 0-100%
                    dtick=0.2,  # Set tick intervals to 20%
                    showgrid=True,
                    gridcolor='lightgray',
                    zeroline=True
                ),
                showlegend=False,
                height=400,
                # Force plotly template explicitly
                template='plotly',
                margin=dict(l=50, r=50, t=50, b=50),
                # Ensure consistent rendering
                autosize=True
            )
            # Force plotly template before rendering
            fig.update_layout(template='plotly')
            
            # Use unique key to force chart refresh and prevent caching
            import time
            chart_key = f"prob_chart_{int(time.time())}"
            # Force plotly theme in st.plotly_chart
            st.plotly_chart(fig, use_container_width=True, key=chart_key, theme="streamlit")
            
            # Confidence score
            max_prob = max(probabilities.values())
            confidence_color = "green" if max_prob > 0.7 else "orange" if max_prob > 0.5 else "red"
            st.markdown(f"**Confidence Score:** <span style='color: {confidence_color}'>{max_prob:.1%}</span>", unsafe_allow_html=True)
    
    def render_interpretation(self, prediction: str):
        """Render the interpretation section"""
        st.subheader("Interpretation")
        self._ensure_predictor_loaded()
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
        
        self._ensure_predictor_loaded()
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
        
        self._ensure_predictor_loaded()
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
        st.dataframe(similar_df, use_container_width=True)
        
        st.info("These are estimated similar vehicles based on your criteria. Actual market prices may vary.")
    
    def render_condition_gauge(self, prediction: str, probabilities: Dict[str, float]):
        """Render a gauge showing prediction confidence"""
        confidence = max(probabilities.values()) if probabilities else 0.0
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Confidence: {prediction}"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_probability_chart(self, probabilities: Dict[str, float]):
        """Render probability distribution as bar chart"""
        if not probabilities:
            return
        
        conditions = list(probabilities.keys())
        probs = list(probabilities.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=conditions,
                y=probs,
                marker_color=['#28a745', '#20c997', '#17a2b8', '#6f42c1', '#ffc107', '#dc3545', '#6c757d'],
                text=[f"{prob:.1%}" for prob in probs],
                textfont_size=12,
                textangle=0,
                textposition="outside",
                # Ensure proper data binding
                customdata=probs,
                hovertemplate="<b>%{x}</b><br>Probability: %{customdata:.1%}<extra></extra>"
            )
        ])
        
        fig.update_layout(
            title="Condition Probability Distribution",
            xaxis_title="Vehicle Condition",
            yaxis_title="Probability",
            yaxis=dict(
                tickformat='.1%',
                range=[0, 1],  # Force Y-axis to be 0-100%
                dtick=0.2,  # Set tick intervals to 20%
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True
            ),
            showlegend=False,
            # Force plotly template explicitly
            template='plotly',
            margin=dict(l=50, r=50, t=50, b=50),
            # Ensure consistent rendering
            autosize=True
        )
        
        # Force plotly template before rendering
        fig.update_layout(template='plotly')
        
        # Force plotly theme in st.plotly_chart
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
    def render_price_comparison_chart(self, input_data: Dict[str, Any], price_analysis: Dict[str, Any]):
        """Render price comparison visualization"""
        current_price = price_analysis['current_price']
        market_value = price_analysis['estimated_market_value']
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Current Price', 'Market Value'],
                y=[current_price, market_value],
                marker_color=['#ff6b6b', '#4ecdc4'],
                text=[f'${current_price:,.0f}', f'${market_value:,.0f}'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Price vs Market Value Comparison",
            yaxis_title="Price ($)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_age_mileage_chart(self, input_data: Dict[str, Any]):
        """Render age vs mileage scatter plot with condition zones"""
        year = input_data.get('year', 2020)
        odometer = input_data.get('odometer', 0)
        age = 2024 - year
        
        # Create condition zones
        fig = go.Figure()
        
        # Add condition zones
        fig.add_trace(go.Scatter(
            x=[0, 20], y=[0, 200000],
            fill='tonexty',
            fillcolor='rgba(40, 167, 69, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Excellent Zone',
            showlegend=True
        ))
        
        # Add your vehicle point
        fig.add_trace(go.Scatter(
            x=[age], y=[odometer],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Your Vehicle',
            showlegend=True
        ))
        
        fig.update_layout(
            title="Vehicle Age vs Mileage Analysis",
            xaxis_title="Age (years)",
            yaxis_title="Mileage",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_market_trend_chart(self, input_data: Dict[str, Any]):
        """Render market trend over time"""
        year = input_data.get('year', 2020)
        current_year = 2024
        
        # Generate sample trend data
        years = list(range(year, current_year + 1))
        values = [50000 * (0.9 ** (current_year - y)) for y in years]  # Depreciation trend
        
        fig = go.Figure(data=go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name='Market Value Trend',
            line=dict(color='#667eea', width=3)
        ))
        
        fig.update_layout(
            title="Market Value Trend Over Time",
            xaxis_title="Year",
            yaxis_title="Estimated Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_importance_chart(self):
        """Render feature importance from Gradient Boosting model"""
        self._ensure_predictor_loaded()
        if self.predictor.model is None:
            return
        
        # Get feature importance
        if hasattr(self.predictor.model, 'feature_importances_'):
            importance = self.predictor.model.feature_importances_
            features = ['price', 'year', 'odometer', 'manufacturer', 'model', 'cylinders', 
                       'fuel', 'transmission', 'drive', 'type', 'paint_color', 'state', 
                       'owners', 'location_cluster', 'title_status']
            
            # Create importance chart
            fig = go.Figure(data=[
                go.Bar(
                    x=features,
                    y=importance,
                    marker_color='#4ecdc4'
                )
            ])
            
            fig.update_layout(
                title="Feature Importance in Decision Making",
                xaxis_title="Features",
                yaxis_title="Importance Score",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_condition_distribution_chart(self):
        """Render overall condition distribution"""
        # Sample data - replace with your actual distribution
        conditions = ['New', 'Like New', 'Excellent', 'Good', 'Fair', 'Salvage']
        counts = [5, 15, 25, 35, 15, 5]  # Sample percentages
        
        fig = go.Figure(data=[go.Pie(
            labels=conditions,
            values=counts,
            hole=0.3,
            marker_colors=['#28a745', '#20c997', '#17a2b8', '#6f42c1', '#ffc107', '#dc3545']
        )])
        
        fig.update_layout(
            title="Overall Vehicle Condition Distribution",
            annotations=[dict(text='Market<br>Distribution', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_maintenance_timeline(self, input_data: Dict[str, Any]):
        """Render maintenance timeline based on vehicle age and mileage"""
        year = input_data.get('year', 2020)
        odometer = input_data.get('odometer', 0)
        age = 2024 - year
        
        # Generate maintenance milestones
        milestones = [
            {'service': 'Oil Change', 'interval': 5000, 'next': odometer + 5000},
            {'service': 'Tire Rotation', 'interval': 10000, 'next': odometer + 10000},
            {'service': 'Brake Check', 'interval': 20000, 'next': odometer + 20000},
            {'service': 'Transmission', 'interval': 60000, 'next': odometer + 60000}
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[m['service'] for m in milestones],
                y=[m['next'] for m in milestones],
                marker_color='#ff6b6b',
                text=[f"{m['next']:,} miles" for m in milestones],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Upcoming Maintenance Schedule",
            xaxis_title="Service Type",
            yaxis_title="Mileage",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        # Check if model is loaded
        self._ensure_predictor_loaded()
        model_info = self.predictor.get_model_info()
        if "error" in model_info:
            self.render_error_message(f"❌ Model Error: {model_info['error']}")
            self.render_info_message("Please ensure random_forest.pkl or gradient_boosting.pkl is in the 'model' directory.")
            return
        
        # Render sidebar
        self.render_sidebar()
        
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
                self._ensure_predictor_loaded()
                is_valid, error_message = self.predictor.validate_input_data(input_data)
                
                if not is_valid:
                    self.render_error_message(f"❌ {error_message}")
                    return
                
                # Make prediction
                with self.render_spinner("Analyzing vehicle specifications..."):
                    prediction, probabilities = self.predictor.predict_condition(input_data)
                
                if prediction is not None:
                    # Render prediction results
                    model_type = self.predictor.get_model_info().get('model_type', 'AI Model')
                    self.render_prediction_results(prediction, probabilities, model_type)
                    
                    # Render interpretation
                    self.render_interpretation(prediction)
                    
                    # Market Price Analysis (moved to top)
                    self._ensure_predictor_loaded()
                    price_analysis = self.predictor.analyze_market_price(input_data, prediction)
                    self.render_market_analysis(input_data, prediction)
                    
                    # Render visualizations
                    st.markdown("---")
                    st.subheader("📊 Interactive Analysis")
                    
                    # Price comparison chart
                    self.render_price_comparison_chart(input_data, price_analysis)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Condition confidence gauge
                        self.render_condition_gauge(prediction, probabilities)
                        
                        # Probability distribution chart
                        self.render_probability_chart(probabilities)
                    
                    with col2:
                        # Age vs mileage analysis
                        self.render_age_mileage_chart(input_data)
                        
                        # Market trend chart
                        self.render_market_trend_chart(input_data)
                    
                    # Feature importance chart
                    with st.expander("🔍 Feature Importance Analysis"):
                        self.render_feature_importance_chart()
                    
                    # Market distribution chart
                    with st.expander("📈 Market Distribution"):
                        self.render_condition_distribution_chart()
                    
                    # Maintenance timeline
                    with st.expander("🔧 Maintenance Schedule"):
                        self.render_maintenance_timeline(input_data)
                    
                    # Render vehicle insights
                    self._ensure_predictor_loaded()
                    self.render_vehicle_insights(input_data)
                    
                    # Render similar vehicles
                    self.render_similar_vehicles(input_data)
        
        # Render footer
        self.render_footer()


def main():
    """Main function to run the UI"""
    ui = RideSenseUI()
    ui.run()


if __name__ == "__main__":
    main()
