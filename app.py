
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib

# ============================================
# LOAD MODEL
# ============================================

with open("housing_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================================
# LOAD PREPROCESSOR / SCALER
# ============================================

with open("scaler.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# ============================================
# STREAMLIT UI
# ============================================

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

# ============================================
# USER INPUTS
# ============================================

area = st.number_input("Area (sq ft)", min_value=0)

bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10)

stories = st.number_input("Stories", min_value=1, max_value=10)

parking = st.number_input("Parking Spaces", min_value=0, max_value=10)

mainroad = st.selectbox("Main Road", ["yes", "no"])

guestroom = st.selectbox("Guest Room", ["yes", "no"])

basement = st.selectbox("Basement", ["yes", "no"])

hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])

airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])

prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# ============================================
# CREATE INPUT DATAFRAME
# ============================================

input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedroom],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'parking': [parking],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})

# ============================================
# PREDICTION
# ============================================

if st.button("Predict Price"):

    # Preprocess input
    processed_data = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(processed_data)

    st.success(f"🏡 Predicted House Price: {prediction[0]:,.2f}")
