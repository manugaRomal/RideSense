# app.py
import streamlit as st
import pandas as pd
import numpy as np
# import joblib  # Uncomment when model is trained
import plotly.express as px

st.set_page_config(page_title="Vehicle Condition Predictor", layout="wide")
st.title("Ridesense - Vehicle Condition Predictor")

# ---------- STEP 1: User Inputs ----------
st.sidebar.header("Vehicle Details")

year = st.sidebar.number_input("Year", min_value=1900, max_value=2025, value=2015)
price = st.sidebar.number_input("Price ($)", min_value=100, max_value=100000, value=15000)
odometer = st.sidebar.number_input("Odometer (miles)", min_value=0, max_value=500000, value=60000)

manufacturer = st.sidebar.selectbox("Manufacturer", ["ford", "chevrolet", "toyota", "honda", "other"])
fuel = st.sidebar.selectbox("Fuel Type", ["gas", "diesel", "hybrid", "electric", "other"])
transmission = st.sidebar.selectbox("Transmission", ["automatic", "manual", "other"])
drive = st.sidebar.selectbox("Drive", ["fwd", "rwd", "4wd", "other"])
paint_color = st.sidebar.selectbox("Paint Color", ["black", "white", "silver", "red", "blue", "other"])
model_input = st.sidebar.text_input("Model", "")

# Optional location inputs
state = st.sidebar.text_input("State (e.g., CA)", "")
region = st.sidebar.text_input("Region", "")
lat = st.sidebar.number_input("Latitude", value=0.0)
long = st.sidebar.number_input("Longitude", value=0.0)

# Collect inputs into DataFrame (to match model features)
input_data = pd.DataFrame([{
    "year": year,
    "price": price,
    "odometer": odometer,
    "manufacturer": manufacturer,
    "fuel": fuel,
    "transmission": transmission,
    "drive": drive,
    "paint_color": paint_color,
    "model": model_input,
    "state": state,
    "region": region,
    "lat": lat,
    "long": long
}])

st.subheader("Vehicle Input Summary")
st.table(input_data)

# ---------- STEP 2: Prediction ----------
if st.button("Predict Condition"):
    st.info("Model not trained yet! Replace this section with your trained model prediction.")

    # ------------------------
    # Example placeholder logic
    # ------------------------
    # Uncomment and replace with your actual model
    # model = joblib.load("vehicle_condition_model.pkl")
    # prediction = model.predict(input_data)[0]
    # prediction_proba = model.predict_proba(input_data)[0]  # If classifier supports probabilities

    prediction = "Good"
    prediction_proba = {"excellent": 0.2, "good": 0.6, "like new": 0.15, "fair": 0.05}

    st.success(f"Predicted Condition: **{prediction}**")
    
    # Display probabilities as a bar chart
    st.subheader("Prediction Confidence")
    proba_df = pd.DataFrame(list(prediction_proba.items()), columns=["Condition", "Probability"])
    fig = px.bar(proba_df, x="Condition", y="Probability", color="Condition", text="Probability")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- STEP 3: Similar Vehicles Placeholder ----------
    st.subheader("Similar Vehicles Statistics (Placeholder)")
    # Normally you would query your dataset for similar vehicles
    sim_df = pd.DataFrame({
        "Price ($)": [16000, 15500, 16200],
        "Odometer (miles)": [59000, 61000, 60000],
        "Year": [2015, 2014, 2015],
        "Condition": ["good", "good", "good"]
    })
    st.table(sim_df)

    # ---------- STEP 4: Location Cluster / Map ----------
    st.subheader("Location Cluster (Placeholder)")
    map_df = pd.DataFrame({
        "lat": [lat, lat + 0.01, lat - 0.01],
        "long": [long, long + 0.01, long - 0.01],
        "Condition": ["good", "good", "excellent"]
    })
    st.map(map_df)

    st.info("Once the model is trained, this section will show the actual prediction and analytics.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("Developed by Your Name | ML Vehicle Condition Predictor")
