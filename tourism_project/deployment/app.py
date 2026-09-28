
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("tourism_model.joblib")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Wellness Tourism Predictor", layout="wide")
st.title("Wellness Tourism Package Purchase Predictor")
st.write("Estimate customer purchase propensity and prioritize campaign outreach.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        Age = st.number_input("Age", 18, 90, 35)
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        DurationOfPitch = st.number_input("Duration of Pitch", 0.0, 120.0, 15.0)
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Gender = st.selectbox("Gender", ["Male", "Female"])

    with c2:
        NumberOfPersonVisiting = st.number_input("Persons Visiting", 1, 10, 2)
        NumberOfFollowups = st.number_input("Follow-ups", 0.0, 10.0, 3.0)
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Divorced", "Unmarried", "Single"])
        NumberOfTrips = st.number_input("Annual Trips", 0.0, 30.0, 3.0)

    with c3:
        Passport = st.selectbox("Has Passport", [0, 1])
        PitchSatisfactionScore = st.selectbox("Pitch Satisfaction", [1, 2, 3, 4, 5])
        OwnCar = st.selectbox("Owns Car", [0, 1])
        NumberOfChildrenVisiting = st.number_input("Children Visiting", 0.0, 10.0, 0.0)
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Income", 0.0, 500000.0, 25000.0)

    threshold = st.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05)
    submit = st.form_submit_button("Predict")

if submit:
    row = pd.DataFrame([{
        "Age": Age, "TypeofContact": TypeofContact, "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch, "Occupation": Occupation,
        "Gender": Gender, "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups, "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus, "NumberOfTrips": NumberOfTrips,
        "Passport": Passport, "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar, "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation, "MonthlyIncome": MonthlyIncome
    }])

    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= threshold)

    st.metric("Purchase Probability", f"{probability:.1%}")
    if prediction:
        st.success("Priority lead at the selected threshold.")
    else:
        st.info("Lower-priority lead at the selected threshold.")
