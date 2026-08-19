import streamlit as st
import joblib
import pandas as pd

st.title("🏠 Property Price Predictor")
st.write("Enter property details to get price prediction")

# Load model
model = joblib.load('models/property_price_model.joblib')

# Input fields
location = st.selectbox("Location", ["Downtown", "Suburb", "Rural"])
area_sqft = st.number_input("Area in sqft", 500, 10000, 1850)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
stories = st.number_input("Stories", 1, 5, 2)
age_years = st.number_input("Age in years", 0, 100, 12)
garage = st.selectbox("Garage", [0, 1])
balcony = st.selectbox("Balcony", [0, 1])
basement = st.selectbox("Basement", [0, 1])
nearby_school_score = st.slider("School Score", 0.0, 10.0, 8.5)
crime_rate = st.number_input("Crime Rate", 0.0, 1.0, 0.2)
property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Independent House"])

sample = {
    "location": location,
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "age_years": age_years,
    "garage": garage,
    "balcony": balcony,
    "basement": basement,
    "nearby_school_score": nearby_school_score,
    "crime_rate": crime_rate,
    "property_type": property_type
}

if st.button("Predict Price"):
    from src.property_price_model.predict import predict_property_price
    predicted_price = predict_property_price(sample)
    st.success(f"Predicted property price: ${predicted_price:,.2f}")