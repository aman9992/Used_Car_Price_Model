# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import datetime

# --- Load the Final Pipeline ---
try:
    # This loads the entire preprocessing and model pipeline
    pipeline = pk.load(open('pipeline.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'pipeline.pkl' not found. Please run the training script first to generate it.")
    st.stop()

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
st.title('🚗 Used Car Price Predictor')
st.markdown("This app uses a robust **XGBoost model** with a proper preprocessing pipeline to predict car prices.")

# --- Custom Function for Indian Currency Formatting ---
def format_price(price):
    """Formats the price into Indian Rupee style (lakhs, thousands)."""
    price_str = str(int(price))
    if len(price_str) > 3:
        last_three = price_str[-3:]
        other_digits = price_str[:-3]
        formatted_digits = ''
        for i, digit in enumerate(reversed(other_digits)):
            if i % 2 == 0 and i > 0:
                formatted_digits = digit + ',' + formatted_digits
            else:
                formatted_digits = digit + formatted_digits
        return "₹ " + formatted_digits + ',' + last_three
    else:
        return "₹ " + price_str

# --- Sidebar for User Inputs ---
st.sidebar.header("Choose Car Specifications")
try:
    # Load data just for populating dropdowns
    cars_data = pd.read_csv('CarDetails_105k.csv').dropna()
    cars_data['brand'] = cars_data['name'].str.split().str[0]
except FileNotFoundError:
    st.error("Error: 'CarDetails_105k.csv' not found.")
    st.stop()

# --- Input fields are text-based to match the original data ---
# Year slider now stops at 2020, as the model was trained on data up to that year
year = st.sidebar.slider('Manufactured Year', 1994, 2020, 2018)
brand = st.sidebar.selectbox('Car Brand', sorted(cars_data['brand'].unique()))
km_driven = st.sidebar.slider('Kilometers Driven', 100, 200000, 50000)
fuel = st.sidebar.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.sidebar.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.sidebar.selectbox('Transmission Type', cars_data['transmission'].unique())
owner = st.sidebar.selectbox('Owner Type', cars_data['owner'].unique())
mileage = st.sidebar.slider('Mileage (kmpl)', 10.0, 40.0, 19.0)
engine = st.sidebar.slider('Engine (CC)', 700, 5000, 1400)
max_power = st.sidebar.slider('Max Power (bhp)', 30, 250, 100)
seats = st.sidebar.selectbox('Number of Seats', [2, 4, 5, 6, 7, 8, 9, 10], index=2)

# --- Prediction Logic ---
if st.sidebar.button('**Predict Price**'):
    # Get current year to calculate age
    current_year = datetime.datetime.now().year
    age = current_year - year
    
    # Create a DataFrame from user input
    input_data = pd.DataFrame(
        [[km_driven, mileage, engine, max_power, seats, age, fuel, seller_type, transmission, owner, brand]],
        columns=['km_driven', 'mileage', 'engine', 'max_power', 'seats', 'age', 'fuel', 'seller_type', 'transmission', 'owner', 'brand']
    )
    
    # Use the pipeline to preprocess and predict
    car_price = pipeline.predict(input_data)
    predicted_price = abs(round(car_price[0]))
    
    # Use the custom function to format the price
    formatted_price = format_price(predicted_price)
    
    st.success(f"### Predicted Price: {formatted_price}")
else:
    st.info("Please select features in the sidebar and click 'Predict Price'.")