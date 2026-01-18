import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -----------------------------
# Load model from Hugging Face Model Hub
# -----------------------------
MODEL_REPO_ID = "avatar2102/tourism-package-model"
MODEL_FILENAME = "best_tourism_model_v1.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = joblib.load(model_path)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Tourism Package Purchase Prediction App")
st.write(
    "This app predicts whether a customer is likely to purchase the newly introduced "
    "Wellness Tourism Package based on customer details and interaction attributes."
)
st.write("Enter the customer details below and click **Predict**.")

# -----------------------------
# Collect user inputs (based on dataset dictionary)
# -----------------------------
Age = st.number_input("Age", min_value=18, max_value=100, value=35)

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])

CityTier = st.selectbox("City Tier", [1, 2, 3], index=1)

Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])

NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)

PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5], index=3)

MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

NumberOfTrips = st.number_input("Number of Trips (Annual)", min_value=0, max_value=50, value=2)

Passport = st.selectbox("Passport", ["Yes", "No"])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])

NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (<5 years)", min_value=0, max_value=10, value=0)

Designation = st.text_input("Designation", value="Executive")

MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=30000)

PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=3)

ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)

DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=500, value=30)

# -----------------------------
# Create input dataframe (match training features)
# -----------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# Classification threshold (same style as your training logic)
classification_threshold = 0.45

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    pred_proba = model.predict_proba(input_data)[0, 1]
    pred = int(pred_proba >= classification_threshold)

    st.subheader("Prediction Result")
    st.write(f"Purchase probability: **{pred_proba:.3f}**")

    if pred == 1:
        st.success("✅ Likely to purchase the Wellness Tourism Package (ProdTaken = 1)")
    else:
        st.warning("❌ Not likely to purchase the Wellness Tourism Package (ProdTaken = 0)")
