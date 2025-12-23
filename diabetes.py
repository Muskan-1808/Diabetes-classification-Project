import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction Web App")
st.write("Decision Tree Classification (Outcome: 0 = No Diabetes, 1 = Diabetes)")

# ---------------- DEBUG: SHOW FILES ----------------
st.write("📂 Files available in app directory:")
st.write(os.listdir())

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("glucose.csv")

try:
    data = load_data()
    st.success("✅ Dataset loaded successfully")
except Exception as e:
    st.error("❌ Dataset loading failed")
    st.write(e)
    st.stop()

# ---------------- DATA CHECK ----------------
st.write("🔍 Dataset Preview:")
st.write(data.head())

# ---------------- SPLIT FEATURES ----------------
try:
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
except KeyError:
    st.error("❌ Column 'Outcome' not found in CSV")
    st.write("Available columns:", data.columns)
    st.stop()

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=400
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(X_train, y_train)

# ---------------- USER INPUT ----------------
st.subheader("📝 Enter Patient Details")

Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0, format="%.2f")
DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, format="%.3f"
)
Age = st.number_input("Age", min_value=1, step=1)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):
    input_data = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age]
    })

    result = model.predict(input_data)[0]

    if result == 1:
        st.error("⚠️ Outcome = 1 → Person has Diabetes")
    else:
        st.success("✅ Outcome = 0 → Person does NOT have Diabetes")

# ---------------- ACCURACY ----------------
accuracy = model.score(X_test, y_test)
st.write(f"📊 Model Accuracy: **{accuracy:.2f}**")
