from catboost import CatBoostRegressor, Pool
import joblib
import streamlit as st
import pandas as pd

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('hdb-resale-price.csv')
    data['lease_commence_year'] = pd.to_datetime(data['lease_commence_date'], format='%Y').dt.year
    return data

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('house_price_prediction_catboost.pkl', 'rb') as file:
            model = joblib.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

data = load_data()
model = load_model()

st.write("""
# House Price Prediction App
This app predicts the **HDB resale price**!
""")

st.sidebar.header('User Input Parameters')

def user_input_features(data):
    closest_mrt_dist = st.sidebar.slider('Closest MRT Distance', float(data['closest_mrt_dist'].min()), float(data['closest_mrt_dist'].max()), float(data['closest_mrt_dist'].mean()))
    cbd_dist = st.sidebar.slider('CBD Distance', float(data['cbd_dist'].min()), float(data['cbd_dist'].max()), float(data['cbd_dist'].mean()))
    floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', float(data['floor_area_sqm'].min()), float(data['floor_area_sqm'].max()), float(data['floor_area_sqm'].mean()))
    years_remaining = st.sidebar.slider('Years Remaining', float(data['years_remaining'].min()), float(data['years_remaining'].max()), float(data['years_remaining'].mean()))
    lease_commence_year = st.sidebar.slider('Lease Commence Year', int(data['lease_commence_year'].min()), int(data['lease_commence_year'].max()), int(data['lease_commence_year'].mean()))
    lease_commence_month = st.sidebar.slider('Lease Commence Month', 1, 12, 1)
    lease_commence_day = st.sidebar.slider('Lease Commence Day', 1, 31, 1)
    transaction_year = st.sidebar.slider('Transaction Year', 2012, 2014, 2013)
    transaction_month = st.sidebar.slider('Transaction Month', 1, 12, 1)
    transaction_day = st.sidebar.slider('Transaction Day', 1, 31, 1)
    
    features = {
        'closest_mrt_dist': closest_mrt_dist,
        'cbd_dist': cbd_dist,
        'floor_area_sqm': floor_area_sqm,
        'years_remaining': years_remaining,
        'lease_commence_year': lease_commence_year,
        'lease_commence_month': lease_commence_month,
        'lease_commence_day': lease_commence_day,
        'transaction_year': transaction_year,
        'transaction_month': transaction_month,
        'transaction_day': transaction_day
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features(data)

st.subheader('User Input Parameters')
st.write(input_df)

# Apply the model to make predictions
if model:
    try:
        # Ensure the input DataFrame is compatible with CatBoost
        input_data = Pool(data=input_df)
        prediction = model.predict(input_data)
        st.subheader('Prediction')
        st.write(f"Predicted Resale Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
else:
    st.error("Model could not be loaded. Please check the model file and try again.")
