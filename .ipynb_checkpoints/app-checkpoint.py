import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🩺 Heart Disease Prediction App")
st.write("**Enter patient details below:**")

# Input fields
age = st.slider('Age', 20, 100)
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
cp = st.slider('Chest Pain Type (1-4)', 1, 4)
trestbps = st.slider('Resting Blood Pressure (mmHg)', 80, 200)
chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', [0, 1])
restecg = st.selectbox('Resting ECG (0-2)', [0, 1, 2])
thalach = st.slider('Max Heart Rate Achieved', 60, 220)
exang = st.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [0, 1])
oldpeak = st.slider('ST Depression (Oldpeak)', 0.0, 6.0, 0.0)
slope = st.selectbox('ST Slope (1=up, 2=flat, 3=down)', [1, 2, 3])

# Prediction
if st.button('Predict'):
    patient_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope]])
    patient_data_scaled = scaler.transform(patient_data)
    prediction = model.predict(patient_data_scaled)
    probability = model.predict_proba(patient_data_scaled)

    if prediction[0] == 1:
        st.error(f"⚠️ Heart Disease Detected.\nPrediction Probability: {probability[0][1]*100:.2f}%")
    else:
        st.success(f"✅ No Heart Disease Detected.\nPrediction Probability: {probability[0][0]*100:.2f}%")

st.caption("Made by Raj 🔥")
