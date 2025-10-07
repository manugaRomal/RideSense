"""
RideSense - Vehicle Condition Prediction System
Fixed version with lazy model loading
"""
import sys
import os
import warnings
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*use_container_width.*")

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="RideSense - Vehicle Condition Predictor", 
    page_icon="🚗",
    layout="wide"
)

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main function to run the RideSense application with lazy loading"""
    try:
        # Import and initialize UI with lazy model loading
        from src.ui import RideSenseUI
        
        with st.spinner("Initializing AI model..."):
            ui = RideSenseUI()
        
        ui.run()
        
    except Exception as e:
        st.error(f"❌ Error loading application: {str(e)}")
        st.exception(e)
        st.info("Please check that the random_forest.pkl file is in the model/ directory.")

if __name__ == "__main__":
    main()
