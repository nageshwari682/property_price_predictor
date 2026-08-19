import joblib
import pandas as pd

# 1. Load your original model
print("Loading model...")
model = joblib.load('models/property_price_model.joblib')

# 2. Re-save it with compression that works on Streamlit
print("Re-saving model...")
joblib.dump(model, 'models/property_price_model.joblib', compress=3, protocol=4)
print("Done! Model re-saved")