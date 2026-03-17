import pandas as pd
import streamlit as st
import joblib

# Load trained model
model = joblib.load("house_model.pkl")

st.title("House Price Prediction")
st.write("Enter the house details below:")

# Input fields
median_income = st.number_input("Median Income", min_value=0.0, value=3.0)
housing_median_age = st.number_input("House Age", min_value=0.0, value=30.0)
total_rooms = st.number_input("Total Rooms", min_value=0, value=1000)
total_bedrooms = st.number_input("Total Bedrooms", min_value=0, value=300)
population = st.number_input("Population", min_value=0, value=1500)
households = st.number_input("Households", min_value=0, value=500)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

# Predict button
if st.button("Predict Price"):
    # Make sure input column names match training data exactly
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income]
    })

    prediction = model.predict(input_data)
    st.success("Estimated House Price: ${prediction[0]:,.2f}")