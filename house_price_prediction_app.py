# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import joblib
import pickle

# Load and preprocess data
def load_data():
    df = pd.read_csv('hdb-resale-price.csv')  # Replace with your dataset path
    return df

# Load the model
def load_model():
    model = joblib.load('model.pkl')
    return model

# Preprocess input features
def preprocess_input(data, scaler):
    data_scaled = scaler.transform(data)
    return data_scaled

# Main function to run the app
def main():
    st.title("Car Price Prediction App")
    st.write("This app predicts the price of a car based on its features.")
    
    # Load data
    df = load_data()
    
    # Display data
    if st.checkbox("Show raw data"):
        st.write(df.head())
    

    # User input for new prediction
    st.header("Input Features")
    # ['closest_mrt_dist', 'cbd_dist', 'floor_area_sqm']
    
    closest_mrt_dist = st.number_input("Closest MRT Distance (m)", min_value=31.759821, max_value=3496.402761, help="The distance to the nearest MRT station in meters.")
    cbd_dist = st.number_input("CBD Distance (m)", min_value=592.121638	, max_value=20225.103698, help="The distance to the Central Business District in meters.")
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=31.0, max_value=266.0, help="The floor area of the property in square meters.")
    
    input_features = pd.DataFrame({
        'closest_mrt_dist': [closest_mrt_dist],
        'cbd_dist': [cbd_dist],
        'floor_area_sqm': [floor_area_sqm]
    })
    
    st.write(input_features)

    input_features = pd.DataFrame({
        'closest_mrt_dist': [closest_mrt_dist/1000],
        'cbd_dist': [cbd_dist/1000],
        'floor_area_sqm': [floor_area_sqm/1000]
    })
    
    
    # Load model
    model = load_model()
    
    if st.button("Predict"):
        # Make prediction
        prediction = model.predict(input_features)
        
        # Display prediction
        st.subheader("Predicted Selling Price (Per sqm)")
        st.write(f"${prediction[0]:,.2f}")
    
main()
