import streamlit as st
import joblib
import pandas as pd
Model=joblib.load('LR_Model.pkl')
encoded_columns=joblib.load('Columns.pkl')
Scaler=joblib.load('Scaler.pkl')
#sets the title to frontend 
st.title("🏡 California House Price Prediction",text_alignment="center")

#Set the page configuration for the Streamlit app, including the title, icon, layout 
st.set_page_config(page_title="House Price Prediction", page_icon=":house:", layout="wide")

#sets the description of the app to frontend
st.write("This app predicts the price of a house based on various features."
          "Please enter the details below and click on the 'Predict' button to get the estimated price.")

# Create input fields for the user to enter the features of the house
col1, col2 = st.columns(2)

# Add features in each column
with col1:
    longitude =st.number_input(
        "Longitude", 
        min_value=-180.0,          
        max_value=180.0,
        value=0.0,
        )
    
    latitude = st.number_input(
        "Latitude", 
        min_value=-90.0,          
        max_value=90.0,
        value=0.0,
        )
    population = st.number_input(
        "Population", 
        min_value=0,          
        max_value=10000,
        value=0,
        )
    housing_median_age = st.number_input(
        "Housing Median Age", 
        min_value=0,          
        max_value=100,
        value=0,
        )
    households = st.number_input(
        "Households", 
        min_value=0,          
        max_value=1000,
        value=0,
        )
    
with col2:

    total_rooms = st.number_input(
        "Total Rooms", 
        min_value=0,          
        max_value=1000,
        value=0,
        )
    total_bedrooms = st.number_input(
        "Total Bedrooms", 
        min_value=0,          
        max_value=1000,
        value=0,
        )

    median_income = st.number_input(
        "Median Income", 
        min_value=0.0,          
        max_value=100.0,
        value=0.0,
        )
    #create input field for ocean_proximity with options
    ocean_proximity = st.selectbox(
        "Ocean Proximity", 
        options=["<1H OCEAN", "INLAND", "ISLAND", 
                 "NEAR BAY", "NEAR OCEAN"]
        )

    currency = st.selectbox(
        "Select currency for prediction", 
        options=["USD", "INR", "EUR", "GBP"],
        index=1
        )
exchange_rates = {"USD": 1.0, "INR": 82.5, "EUR": 0.92, "GBP": 0.78}
currency_symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}

if st.button("Predict"):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'median_income': [median_income],
        'households': [households],
        'ocean_proximity': [ocean_proximity]
    })
    numerical_columns = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                         "total_bedrooms", "population", "median_income", "households"]
    x_encoded = pd.get_dummies(input_data)
    x_encoded = x_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Apply scaling to numerical features
    x_encoded[numerical_columns] = Scaler.transform(x_encoded[numerical_columns])

    # Make prediction using the trained model
    prediction = Model.predict(x_encoded)[0]
    converted_price = prediction * exchange_rates[currency]
    symbol = currency_symbols[currency]

    # Show the processed input for debugging
    st.write("Processed Input DataFrame:", x_encoded)

    # Display the predicted price in the selected currency with custom background
    st.markdown(
        f"<div class='prediction-box'>Predicted Selling Price ({currency}): <strong>{symbol}{converted_price:,.2f}</strong></div>",
        unsafe_allow_html=True,
    )


import base64

def get_base64_of_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_path = "C:/Users/HP-PC/Desktop/housing model/background.jpg"
encoded_image = get_base64_of_image(image_path)

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{encoded_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
[data-testid="stMarkdownContainer"] {{
    color: darkwhite;
    font-size: 18px;
    font-weight: bold;
}}
.prediction-box {{
    background: rgba(0, 111, 255, 0.85);
    color: white;
    padding: 18px;
    border-radius: 14px;
    border: 2px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    font-size: 1.2rem;
    max-width: 100%;
    margin: 16px 0;
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


