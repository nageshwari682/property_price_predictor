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

# INPUTS - CHANGE THESE TO MATCH YOUR DATA
col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
with col2:
    sqft = st.number_input("Square Feet", 500, 10000, 1500)
    location = st.selectbox("Location", ["Hyderabad", "Mumbai", "Delhi"])

if st.button("Predict Price"):
    # MAKE DATAFRAME MATCH TRAINING COLUMNS
    input_df = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms, 
        'sqft': sqft,
        'location': location
    }])
    input_df = pd.get_dummies(input_df) # important if you used get_dummies
    
    pred = model.predict(input_df)[0]
    st.success(f"Predicted property price: ₹{pred:,.2f}")