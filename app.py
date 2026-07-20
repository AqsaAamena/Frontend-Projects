#Q1. (Setup and Libraries)

#Streamlit: used to build interactive web apps directly in Python
import streamlit as st
#Pandas: powerful library for data manipulation and analysis
import pandas as pd
#Joblib: helps in saving and loading machine learning models efficiently
import joblib
model = joblib.load("LR_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")
# Configure the Streamlit page
st.set_page_config(
    # Sets the title shown in the browser tab
    page_title="Ford Car Price Predictor",
    # Centers the app layout for better readability,keeping inputs and outputs aligned.
    layout="centered"                       
)

# Display the main title of the app
st.title("Ford Car Price Predictor")
# Display a short description for user guidance
st.write("🚗 **Welcome!**\nEnter the car details below and let the app predict its **selling price** with ease.")

# Numerical input fields for car details
# Manufacturing Year
year = st.number_input(
    "Manufacturing Year",
    min_value=1990,   # cars older than 1990 are rare in datasets
    max_value=2026,   # current year
    value=2015        # default year
)

# Mileage
mileage = st.number_input(
    "Mileage (in miles)",
    min_value=0,
    max_value=500000,  # upper bound for used cars
    value=30000        # default mileage
)

# Road Tax
tax = st.number_input(
    "Road Tax (£)",
    min_value=0,
    max_value=1000,
    value=150
)

# Miles per Gallon (MPG)
mpg = st.number_input(
    "Miles per Gallon (MPG)",
    min_value=10,
    max_value=100,
    value=40
)

# Engine Size
engine_size = st.number_input(
    "Engine Size (Litres)",
    min_value=1.0,
    max_value=5.0,
    value=2.0
)
#Transmission type
transmission = st.selectbox(
    "Transmission",
    options=["Automatic", "Manual", "Semi-Auto"]
)

#Fuel Type
fuel_type = st.selectbox(
    "Fuel Type",
    options=["Petrol", "Diesel", "Hybrid", "Electric","Other"]
)
#-------Advantage of using selectbox------ 
#- It restricts user input to valid predefined options, avoiding typos or invalid entries.
# Text input for Car Model
car_model = st.text_input(
    "Car Model",
    placeholder="e.g., Fiesta, Focus, Kuga"
)

# Predict Price button
if st.button("Click to Predict Price"):
    # When the Predict Price button is clicked
    #Q8. (Creating Input DataFrame & Encoding)
    #Creates a DataFrame from user inputs
    input_data = pd.DataFrame({
        "year": [year],
        "mileage": [mileage],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size],
        "transmission": [transmission],
        "fuelType": [fuel_type],
        "model": [car_model]
    })
    numerical_columns = ["year", "mileage", "tax", "mpg", "engineSize"]

    #Perform One-Hot Encoding on categorical features
    X_encoded = pd.get_dummies(input_data)

    # Align encoded columns with training columns
    X_encoded = X_encoded.reindex(columns=encoded_columns, fill_value=0)

     #Apply scaling to numerical features
    X_encoded[numerical_columns] = scaler.transform(X_encoded[numerical_columns])

    #Make prediction using the trained model
    prediction = model.predict(X_encoded)

    # Show the processed input for debugging
    st.write("Processed Input DataFrame:", X_encoded)

    # Display the predicted price
    st.success(f"Predicted Selling Price: £{prediction[0]:,.2f}")



    
