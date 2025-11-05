import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import os

# Set Streamlit Page Config
st.set_page_config(
    page_title="🚔 Crime Prediction App",
    page_icon="⚠️",
    layout="wide"
)

# Load the trained ML model
@st.cache_resource
def load_model():
    with open("crime_model_compressed.pkl", "rb") as f:
        return joblib.load(f)

# Load label encoders
@st.cache_resource
def load_encoders():
    with open("label_encoders.pkl", "rb") as f:
        return pickle.load(f)

# Load crime dataset to populate dropdowns
@st.cache_data
def load_data():
    return pd.read_csv("crime.csv")

# Load model, encoders, and data
model = load_model()
encoders = load_encoders()
crime_data = load_data()

# Extract unique states and state-district mapping
locations = crime_data["STATE/UT"].unique()
state_district_map = crime_data.groupby("STATE/UT")["DISTRICT"].unique().to_dict()

# UI CSS Styling
st.markdown("""
    <style>
    .main { background-color: white; padding: 20px; border-radius: 12px; }
    h1 { color: #e63946; font-weight: bold; text-align: center; }
    .stButton>button { border-radius: 8px; padding: 10px 20px; background-color: #007BFF; color: white; }
    .stButton>button:hover { background-color: #0056b3; }
    .alert { padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; }
    .alert-danger { background-color: #ff4b4b; color: white; }
    .alert-success { background-color: #28a745; color: white; }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1>🚨 Crime Prediction App</h1>", unsafe_allow_html=True)
st.markdown("## 🔍 Check the crime risk in your area!")

# User Input
st.markdown("### 📌 Select Details Below")
col1, col2, col3 = st.columns(3)

with col1:
    location = st.selectbox("📍 **Choose State/UT**", locations)

available_districts = state_district_map.get(location, [])

with col2:
    district = st.selectbox("🏙️ **Select District**", available_districts)

with col3:
    year = st.number_input("📅 **Enter Year**", min_value=2000, max_value=2025, value=2023, step=1)

# Prediction Section
if st.button("🔮 **Predict Crime Risk**"):
    try:
        with st.spinner("🔎 Analyzing crime data... Please wait."):

            # Encode input values
            location_encoded = encoders["STATE/UT"].transform([location])[0]
            district_encoded = encoders["DISTRICT"].transform([district])[0]

            # Create input DataFrame
            input_df = pd.DataFrame([[location_encoded, district_encoded, year]],
                                    columns=["STATE/UT", "DISTRICT", "YEAR"])

            # Predict
            prediction = model.predict(input_df)[0]

        # Output
        if prediction == 1:
            st.markdown('<div class="alert alert-danger">⚠️ <b>High Crime Risk! Be Cautious.</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-success">✅ <b>Low Crime Risk! Area is relatively safe.</b></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align:center;">
        🔹 <b>Developed by Gokila R</b> | AI-Powered Crime Analysis 🔹<br>
        📧 Contact: <a href="mailto:gokilag812@gmail.com">gokilag812@gmail.com</a>
    </div>
""", unsafe_allow_html=True)
