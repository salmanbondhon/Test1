# import streamlit as st
# import joblib
# import pandas as pd

# # Load the trained model
# model = joblib.load("KNN_model.pkl")  # Make sure the model file is in the same directory

# # Function to make predictions
# def predict(input_data):
#     prediction = model.predict([input_data])
#     return prediction

# # Streamlit app code
# st.title('Streamlit App with Model')

# # User input (for example, numeric input)
# input_data = st.text_input("Enter input data:")

# # Convert input data to the format required by the model
# # Assuming the model expects a list of features (e.g., a list of numbers)
# if input_data:
#     input_data = [float(i) for i in input_data.split(",")]  # Assuming comma-separated input
#     prediction = predict(input_data)
#     st.write(f"Prediction: {prediction}")

import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")  # Ensure the model file is in the same directory

# Function to make predictions
def predict(input_data):
    prediction = model.predict([input_data])
    return prediction[0]  # Return the single prediction value

# Streamlit app code
st.title("Prostate Cancer Recurrence Prediction")

st.markdown("""
This app predicts if cancer may occur again based on user-provided data. 
Fill in the fields below and click **Predict**.
""")

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=120, step=1, value=65)
baseline_psa = st.number_input("Baseline PSA", min_value=0.0, value=12.5, step=0.1)
post_treatment_psa = st.number_input("Post-treatment PSA", min_value=0.0, value=2.3, step=0.1)
gleason_score = st.number_input("Gleason Score", min_value=0, max_value=10, step=1, value=7)
t_stage = st.number_input("T Stage", min_value=0, max_value=4, step=1, value=2)
n_stage = st.number_input("N Stage", min_value=0, max_value=1, step=1, value=1)
m_stage = st.number_input("M Stage", min_value=0, max_value=1, step=1, value=0)
bcr_status = st.number_input("BCR Status (0/1)", min_value=0, max_value=1, step=1, value=0)
time_to_recurrence = st.number_input("Time to Recurrence (months)", min_value=0.0, step=0.1, value=8.3)
family_history = st.number_input("Family History (0/1)", min_value=0, max_value=1, step=1, value=0)
comorbidities = st.number_input("Comorbidities (0/1)", min_value=0, max_value=1, step=1, value=0)
psa_month_1 = st.number_input("PSA Month 1", min_value=0.0, step=0.1, value=2.5)
psa_month_3 = st.number_input("PSA Month 3", min_value=0.0, step=0.1, value=2.7)
psa_month_6 = st.number_input("PSA Month 6", min_value=0.0, step=0.1, value=3.0)
psa_month_12 = st.number_input("PSA Month 12", min_value=0.0, step=0.1, value=3.5)
race_black = st.number_input("Race: Black (0/1)", min_value=0, max_value=1, step=1, value=0)
race_other = st.number_input("Race: Other (0/1)", min_value=0, max_value=1, step=1, value=0)
race_white = st.number_input("Race: White (0/1)", min_value=0, max_value=1, step=1, value=1)
ethnicity_non_hispanic = st.number_input("Ethnicity: Non-Hispanic (0/1)", min_value=0, max_value=1, step=1, value=1)
region_north = st.number_input("Region: North (0/1)", min_value=0, max_value=1, step=1, value=0)
region_south = st.number_input("Region: South (0/1)", min_value=0, max_value=1, step=1, value=1)
region_west = st.number_input("Region: West (0/1)", min_value=0, max_value=1, step=1, value=0)
treatment_radiation = st.number_input("Treatment: Radiation Therapy (0/1)", min_value=0, max_value=1, step=1, value=1)
treatment_surgery = st.number_input("Treatment: Surgery (0/1)", min_value=0, max_value=1, step=1, value=0)

# Collect all inputs in a list
input_data = [
    age, baseline_psa, post_treatment_psa, gleason_score, t_stage, n_stage, m_stage, 
    bcr_status, time_to_recurrence, family_history, comorbidities, psa_month_1, psa_month_3, 
    psa_month_6, psa_month_12, race_black, race_other, race_white, ethnicity_non_hispanic, 
    region_north, region_south, region_west, treatment_radiation, treatment_surgery
]

# Prediction button
if st.button("Predict"):
    prediction = predict(input_data)
    
    # Display the prediction result
    if prediction == 1:
        st.error("Cancer may occur again.")
    else:
        st.success("Cancer may not occur again.")
