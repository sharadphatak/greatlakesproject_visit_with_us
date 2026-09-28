
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"

df = pd.read_csv(DATA_PATH)
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

X_train.to_csv(ROOT / "Xtrain.csv", index=False)
X_test.to_csv(ROOT / "Xtest.csv", index=False)
y_train.to_frame("ProdTaken").to_csv(ROOT / "ytrain.csv", index=False)
y_test.to_frame("ProdTaken").to_csv(ROOT / "ytest.csv", index=False)

print("Prepared train/test artifacts:", X_train.shape, X_test.shape)
