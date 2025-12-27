import streamlit as st
import numpy as np
import pickle

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ===============================
# Load Model & Scaler
# ===============================
model = pickle.load(open("models/heart_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ===============================
# Title
# ===============================
st.markdown("<h1 style='text-align:center;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)

# ===============================
# User Inputs
# ===============================
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, 170)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Old Peak", 0.0, 6.0, 0.0)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible)", [1, 2, 3])

# ===============================
# Prediction
# ===============================
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ No Heart Disease Detected (Risk: {probability*100:.2f}%)")
