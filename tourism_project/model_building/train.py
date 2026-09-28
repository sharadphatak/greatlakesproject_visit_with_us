
from pathlib import Path
import json
import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = ROOT / "tourism_project" / "deployment"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(ROOT / "Xtrain.csv")
X_test = pd.read_csv(ROOT / "Xtest.csv")
y_train = pd.read_csv(ROOT / "ytrain.csv")["ProdTaken"].astype(int)
y_test = pd.read_csv(ROOT / "ytest.csv")["ProdTaken"].astype(int)

cat = X_train.select_dtypes(include="object").columns.tolist()
num = [c for c in X_train.columns if c not in cat]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])

param_grid = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [None, 12],
    "classifier__min_samples_leaf": [1, 2]
}

search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=3, n_jobs=-1)

mlflow.set_experiment("tourism-package-prediction")

with mlflow.start_run():
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "cv_roc_auc": search.best_score_
    }

    mlflow.log_params(search.best_params_)
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    model_path = DEPLOY_DIR / "tourism_model.joblib"
    metrics_path = DEPLOY_DIR / "metrics.json"
    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump({
            "best_params": search.best_params_,
            "metrics": {k: float(v) for k, v in metrics.items()}
        }, f, indent=2)

    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(metrics_path))

    MIN_ROC_AUC = 0.80
    if metrics["roc_auc"] < MIN_ROC_AUC:
        raise RuntimeError(
            f"Model quality gate failed: ROC-AUC={metrics['roc_auc']:.4f} < {MIN_ROC_AUC}"
        )

    print("Best parameters:", search.best_params_)
    print("Metrics:", metrics)
    print("QUALITY GATE PASSED")
