import streamlit as st
import pandas as pd
import joblib

# Clear cache (important for your issue)
st.cache_data.clear()
st.cache_resource.clear()

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction Web App")
st.markdown("Decision Tree Classification (0 = No Diabetes, 1 = Diabetes)")

# ---------------- LOAD MODEL ONLY ----------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# ---------------- USER INPUT ----------------
st.subheader("📝 Enter Patient Details")

Pregnancies = st.number_input("Pregnancies", 0, 20)
Glucose = st.number_input("Glucose", 0, 300)
BloodPressure = st.number_input("Blood Pressure", 0, 200)
SkinThickness = st.number_input("Skin Thickness", 0, 100)
Insulin = st.number_input("Insulin", 0, 900)
BMI = st.number_input("BMI", 0.0, 70.0)
DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
Age = st.number_input("Age", 1, 120)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):
    input_data = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DPF],
        "Age": [Age]
    })

    result = model.predict(input_data)

    if result == 1:
        st.error("⚠️ Person is likely Diabetic")
    else:
        st.success("✅ Person is NOT Diabetic")
