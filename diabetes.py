import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction Web App")
st.markdown("Decision Tree Classification (0 = No Diabetes, 1 = Diabetes)")

# -------- LOAD TRAINED MODEL ONLY --------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# 🔥 GET FEATURE NAMES DIRECTLY FROM MODEL
FEATURES = list(model.feature_names_in_)

# -------- USER INPUT (EMPTY FIELDS) --------
st.subheader("📝 Enter Patient Details")

user_inputs = {}
for feature in FEATURES:
    user_inputs[feature] = st.text_input(feature)

# -------- HELPER FUNCTION --------
def is_number(value):
    return re.fullmatch(r"-?\d+(\.\d+)?", value) is not None

# -------- PREDICTION --------
if st.button("🔍 Predict"):

    values = [v.strip() for v in user_inputs.values()]

    # 1️⃣ Empty check
    if not all(values):
        st.warning("❗ Please fill all the fields before predicting")

    # 2️⃣ Numeric validation
    elif not all(is_number(v) for v in values):
        st.warning("❗ Please enter valid numeric values only")

    else:
        # 3️⃣ Convert to float
        numeric_values = [float(v) for v in values]

        # 4️⃣ Create DataFrame with EXACT feature schema
        input_data = pd.DataFrame(
            [numeric_values],
            columns=FEATURES
        )

        # 5️⃣ Predict
        result = model.predict(input_data)[0]

        if result == 1:
            st.error("⚠️ Person is likely Diabetic")
        else:
            st.success("✅ Person is NOT Diabetic")
