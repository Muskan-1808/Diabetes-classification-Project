import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction Web App")
st.markdown("Decision Tree Classification (0 = No Diabetes, 1 = Diabetes)")

# -------- LOAD MODEL ONLY --------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# -------- FEATURE LIST --------
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# -------- USER INPUT (EMPTY FIELDS) --------
st.subheader("📝 Enter Patient Details")

Pregnancies = st.text_input("Pregnancies")
Glucose = st.text_input("Glucose")
BloodPressure = st.text_input("Blood Pressure")
SkinThickness = st.text_input("Skin Thickness")
Insulin = st.text_input("Insulin")
BMI = st.text_input("BMI")
DPF = st.text_input("Diabetes Pedigree Function")
Age = st.text_input("Age")

# -------- HELPER FUNCTION --------
def is_number(value):
    return re.fullmatch(r"-?\d+(\.\d+)?", value) is not None

# -------- PREDICTION --------
if st.button("🔍 Predict"):

    raw_inputs = [
        Pregnancies.strip(),
        Glucose.strip(),
        BloodPressure.strip(),
        SkinThickness.strip(),
        Insulin.strip(),
        BMI.strip(),
        DPF.strip(),
        Age.strip()
    ]

    # 1️⃣ Empty check
    if not all(raw_inputs):
        st.warning("❗ Please fill all the fields before predicting")

    # 2️⃣ Numeric validation
    elif not all(is_number(val) for val in raw_inputs):
        st.warning("❗ Please enter valid numeric values only")

    else:
        values = [float(val) for val in raw_inputs]

        input_data = pd.DataFrame([values], columns=FEATURES)

        result = model.predict(input_data)[0]

        if result == 1:
            st.error("⚠️ Person is likely Diabetic")
        else:
            st.success("✅ Person is NOT Diabetic")
