import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Clear Streamlit caches (uncomment to force cache clear during development)
st.cache_data.clear()
st.cache_resource.clear()

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction Web App")
st.markdown("Decision Tree Classification (0 = No Diabetes, 1 = Diabetes)")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")  # your dataset file here

data = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model():
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=400
    )
    model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    model.fit(X_train, y_train)
    # Calculate accuracy internally but do NOT display
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model  # return only the model

model = train_model()

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

    result = model.predict(input_data)[0]

    if result == 1:
        st.error("⚠️ Person is likely Diabetic")
    else:
        st.success("✅ Person is NOT Diabetic")
