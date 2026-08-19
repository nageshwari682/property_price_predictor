import streamlit as st
import joblib
import pandas as pd
import traceback

st.title("🏠 Property Price Predictor")
st.write("Enter property details to get price prediction")

# EMERGENCY MODEL LOADER
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/property_price_model.joblib')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error("⚠️ Model file is corrupted. Need to retrain.")
        st.code(traceback.format_exc())
        st.stop()

model = load_model()
col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
with col2:
    sqft = st.number_input("Square Feet", 500, 10000, 1500)
    location = st.selectbox("Location", ["Hyderabad", "Mumbai", "Delhi"])

age_years = st.number_input("Age of Property (years)", 0, 100, 5)
property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Independent House"])
nearby_school_score = st.slider("Nearby School Score", 0, 10, 7)
balcony = st.selectbox("Balcony", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
garage = st.selectbox("Garage", ["Yes", "No"])
stories = st.number_input("Number of Stories", 1, 10, 1)
crime_rate = st.number_input("Crime Rate", 0.0, 100.0, 2.5)

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "area_sqft": sqft,
        "location": location,
        "age_years": age_years,
        "property_type": property_type,
        "nearby_school_score": nearby_school_score,
        "balcony": 1 if balcony == "Yes" else 0,
        "basement": 1 if basement == "Yes" else 0,
        "garage": 1 if garage == "Yes" else 0,
        "stories": stories,
        "crime_rate": crime_rate,
    }])

    pred = model.predict(input_df)[0]
    st.success(f"Predicted property price: ₹{pred:,.2f}")