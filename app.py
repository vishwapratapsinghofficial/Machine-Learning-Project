import streamlit as st
import pandas as pd
import joblib


# for run the app type => python -m streamlit run app.py

# Load model
model = joblib.load('model.pkl')

# App UI
st.title("Car Price Predictor")
st.write("Enter car details to predict its selling price.")

brand = st.selectbox("Brand", ['Hyundai', 'Honda', 'Maruti'])
model_name = st.text_input("Model", "i20")
year = st.slider("Year", 2000, 2025, 2018)
mileage = st.number_input("Mileage (km)", value=30000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner_type = st.selectbox("Owner Type", ['First', 'Second', 'Third'])
engine = st.number_input("Engine Capacity (cc)", value=1200)

# Predict
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        'Brand': brand,
        'Model': model_name,
        'Year': year,
        'Mileage': mileage,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Owner_Type': owner_type,
        'Engine': engine
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹{int(prediction):,}")
