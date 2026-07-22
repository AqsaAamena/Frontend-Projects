import streamlit as st
import pandas as pd
import joblib

# Load saved model and columns
model = joblib.load("heart_disease_model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")

col1, col2 = st.columns(2)
# Input fields for all features
with col1:
    age = st.number_input(
        "Age", 
        min_value=1, 
        max_value=120, 
        value=50
        )
    sex = st.selectbox(
        "Sex", 
        ["M", "F"]
        )
    chest_pain = st.selectbox(
        "Chest Pain Type", 
        ["ATA", "NAP", "ASY", "TA"]
        )
    resting_bp = st.number_input(
        "RestingBP", 
        min_value=0, 
        max_value=250, 
        value=120
         )
    cholesterol = st.number_input(
        "Cholesterol", 
        min_value=0, 
        max_value=600, 
        value=200
        )

with col2:
    fasting_bs = st.selectbox(
        "FastingBS (0 = No, 1 = Yes)", 
        [0, 1]
        )
    resting_ecg = st.selectbox(
        "RestingECG", 
        ["Normal", "ST", "LVH"]
        )
    max_hr = st.number_input(
        "MaxHR", 
        min_value=60, 
        max_value=220, 
        value=150
        )
    exercise_angina = st.selectbox(
        "ExerciseAngina", 
        ["Y", "N"]
        )
    oldpeak = st.number_input(
        "Oldpeak", 
     min_value=-5.0, 
     max_value=10.0, 
     value=1.0, 
     step=0.1)
st_slope = st.selectbox(
     "ST_Slope", 
    ["Up", "Flat", "Down"]
    )

# Predict button
if st.button("Predict"):
    # Create dataframe from inputs
    input_data = {
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    }

    df_input = pd.DataFrame(input_data)

    # One-hot encode and align with training columns
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=columns, fill_value=0)

    # Make prediction
    prediction = model.predict(df_input)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease: Yes")
    else:
        st.success("✅ Heart Disease: No")
        

import base64

def get_base64_of_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

import streamlit as st
import base64

# Function to encode image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Encode your image
bin_file = "beating_heart.jpeg"   # make sure this file is in the same folder as app.py
bg_img = get_base64_of_bin_file(bin_file)

# Inject CSS with base64 image
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{bg_img}");
    background-size: cover;
    background-position: full;

    background-repeat: repeat;
    background-attachment: fixed;
}}
[data-testid="stMarkdownContainer"] {{
    color:purple;
    font-size: 18px;
    font-weight: bold;
}}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

