
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"

EXPECTED_COLUMNS = {
    "CustomerID","ProdTaken","Age","TypeofContact","CityTier","DurationOfPitch",
    "Occupation","Gender","NumberOfPersonVisiting","NumberOfFollowups",
    "ProductPitched","PreferredPropertyStar","MaritalStatus","NumberOfTrips",
    "Passport","PitchSatisfactionScore","OwnCar","NumberOfChildrenVisiting",
    "Designation","MonthlyIncome"
}

df = pd.read_csv(DATA_PATH)
missing = EXPECTED_COLUMNS - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
if df["CustomerID"].duplicated().any():
    raise ValueError("CustomerID must be unique.")
if not set(df["ProdTaken"].dropna().unique()).issubset({0, 1}):
    raise ValueError("ProdTaken must contain only 0/1.")

print(f"Registered {len(df)} rows and {df.shape[1]} columns.")
print(df["ProdTaken"].value_counts().sort_index())
