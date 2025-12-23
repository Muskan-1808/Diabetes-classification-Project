import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction Web App")
st.write("Decision Tree Classification (Outcome: 0 = No Diabetes, 1 = Diabetes)")
pickle.load(open("diabetes_model.pkl", "dt1"))

# -------- USER INPUT SECTION --------
st.subheader("📝 Enter Patient Details")

Pregnancies = st.text_input("Pregnancies")
Glucose = st.text_input("Glucose")
BloodPressure= st.text_input("Blood Pressure")
skinThickness = st.text_input("Skin Thickness")
Insulin = st.text_input("Insulin")
BMI = st.text_input("BMI")
Diabetes_Pedigree_Function= st.text_input("Diabetes Pedigree Function")
Age = st.text_input("Age")

# Prediction
if st.button("🔍 Predict"):
    try:
        input_data = pd.DataFrame({
            "Pregnancies": [int(Pregnancies)],
            "Glucose": [int(Glucose)],
            "BloodPressure": [int(BloodPressure)],
            "SkinThickness": [int(skinThickness)],
            "Insulin": [int(Insulin)],
            "BMI": [float(BMI)],
            "DiabetesPedigreeFunction": [float(Diabetes_Pedigree_Function)],
            "Age": [int(Age)]
        })

        result = dt1.predict(input_data)[0]

        if result == 1:
            st.error("⚠️ Outcome = 1 → Person has Diabetes")
        else:
            st.success("✅ Outcome = 0 → Person does NOT have Diabetes")

    except ValueError:
        st.warning("❗ Please enter valid numeric values for all inputs")

# Model accuracy
accuracy = dt1.score(X_test, y_test)
st.write(f"📊 Model Accuracy: **{accuracy:.2f}**")
