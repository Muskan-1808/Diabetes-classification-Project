import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction Web App")
st.markdown("Decision Tree Classification (0 = No Diabetes, 1 = Diabetes)")

# -------- LOAD TRAINED MODEL ONLY --------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# -------- FEATURE LIST (MUST MATCH TRAINING) --------
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

# -------- USER INPUT (EMPTY BY DEFAULT) --------
st.subheader("📝 Enter Patient Details")

Pregnancies = st.text_input("Pregnancies")
Glucose = st.text_input("Glucose")
BloodPressure = st.text_input("Blood Pressure")
SkinThickness = st.text_input("Skin Thickness")
Insulin = st.text_input("Insulin")
BMI = st.text_input("BMI")
DPF = st.text_input("Diabetes Pedigree Function")
Age = st.text_input("Age")

# -------- PREDICTION WITH STRONG VALIDATION --------
if st.button("🔍 Predict"):

    # Strip spaces from inputs
    inputs = [
        Pregnancies.strip(),
        Glucose.strip(),
        BloodPressure.strip(),
        SkinThickness.strip(),
        Insulin.strip(),
        BMI.strip(),
        DPF.strip(),
        Age.strip()
    ]

    # Check empty fields
    if not all(inputs):
        st.warning("❗ Please fill all the fields before predicting")

    else:
        try:
            # Convert all inputs to float safely
            values = [float(val) for val in inputs]

            input_data = pd.DataFrame(
                [values],
                columns=FEATURES
            )

            result = model.predict(input_data)[0]

            if result == 1:
                st.error("⚠️ Person is likely Diabetic")
            else:
                st.success("✅ Person is NOT Diabetic")

        except ValueError:
            st.warning("❗ Please enter valid numeric values only")
