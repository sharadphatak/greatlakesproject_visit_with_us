
from pathlib import Path
import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = Path(__file__).with_name("tourism_model.joblib")
model = joblib.load(MODEL_PATH)

def predict(
    Age, TypeofContact, CityTier, DurationOfPitch, Occupation, Gender,
    NumberOfPersonVisiting, NumberOfFollowups, ProductPitched,
    PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport,
    PitchSatisfactionScore, OwnCar, NumberOfChildrenVisiting,
    Designation, MonthlyIncome, threshold
):
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
    label = "Priority lead" if probability >= threshold else "Lower-priority lead"
    return label, probability

inputs = [
    gr.Number(label="Age", value=35),
    gr.Dropdown(["Self Enquiry", "Company Invited"], label="Type of Contact", value="Self Enquiry"),
    gr.Dropdown([1,2,3], label="City Tier", value=1),
    gr.Number(label="Duration of Pitch", value=15),
    gr.Dropdown(["Salaried","Small Business","Large Business","Free Lancer"], label="Occupation", value="Salaried"),
    gr.Dropdown(["Male","Female"], label="Gender", value="Male"),
    gr.Number(label="Persons Visiting", value=2),
    gr.Number(label="Follow-ups", value=3),
    gr.Dropdown(["Basic","Deluxe","Standard","Super Deluxe","King"], label="Product Pitched", value="Basic"),
    gr.Dropdown([3,4,5], label="Preferred Property Star", value=3),
    gr.Dropdown(["Married","Divorced","Unmarried","Single"], label="Marital Status", value="Married"),
    gr.Number(label="Annual Trips", value=3),
    gr.Dropdown([0,1], label="Has Passport", value=0),
    gr.Dropdown([1,2,3,4,5], label="Pitch Satisfaction", value=3),
    gr.Dropdown([0,1], label="Owns Car", value=0),
    gr.Number(label="Children Visiting", value=0),
    gr.Dropdown(["Executive","Manager","Senior Manager","AVP","VP"], label="Designation", value="Executive"),
    gr.Number(label="Monthly Income", value=25000),
    gr.Slider(0.10,0.90,value=0.50,step=0.05,label="Decision Threshold")
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=[gr.Textbox(label="Lead Classification"), gr.Number(label="Purchase Probability")],
    title="Wellness Tourism Package Purchase Predictor",
    description="Customer purchase propensity model for the Visit with Us MLOps project."
)

if __name__ == "__main__":
    demo.launch()
