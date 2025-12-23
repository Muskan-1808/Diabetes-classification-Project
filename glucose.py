import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction Web App")
st.write("Decision Tree Classification (Outcome: 0 = No Diabetes, 1 = Diabetes)")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Diabetes - Copy.csv")

data = load_data()

# Separate features and target
X = data.drop("Outcome", axis=1)   # INPUT FEATURES
y = data["Outcome"]                # TARGET (0 or 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=400
)

# Train Decision Tree model
model = DecisionTreeClassifier(criterion="entropy",max_depth=3)
model.fit(X_train, y_train)

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

        result = model.predict(input_data)[0]

        if result == 1:
            st.error("⚠️ Outcome = 1 → Person has Diabetes")
        else:
            st.success("✅ Outcome = 0 → Person does NOT have Diabetes")

    except ValueError:
        st.warning("❗ Please enter valid numeric values for all inputs")

# Model accuracy
accuracy = model.score(X_test, y_test)
st.write(f"📊 Model Accuracy: **{accuracy:.2f}**")
