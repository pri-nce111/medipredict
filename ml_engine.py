"""Machine-learning engine: data loading, model comparison, prediction, explanation.

Dataset: Pima Indians Diabetes (public, UCI / Kaggle). Put the CSV at data/diabetes.csv
with columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age, Outcome.
If the file is missing, a clearly-labelled SYNTHETIC demo dataset is generated so the
app still runs. Replace it with the real CSV for your final project/research.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "diabetes.csv")
BUNDLE_PATH = os.path.join(BASE, "models", "bundle.joblib")

# column -> (label, min, max, default)  — used for input validation in the UI
FEATURES = {
    "Pregnancies": ("Pregnancies", 0, 20, 1),
    "Glucose": ("Glucose (mg/dL)", 40, 400, 120),
    "BloodPressure": ("Blood pressure, diastolic (mm Hg)", 30, 200, 72),
    "SkinThickness": ("Skin thickness (mm; 0 = unknown)", 0, 100, 20),
    "Insulin": ("Insulin (µU/mL; 0 = unknown)", 0, 900, 80),
    "BMI": ("BMI", 10.0, 70.0, 28.0),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function", 0.05, 2.5, 0.4),
    "Age": ("Age (years)", 1, 120, 35),
}
FEATURE_COLS = list(FEATURES)
# In this dataset a 0 in these columns really means "not measured"
ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# DB column name  <->  dataset column name
DB_TO_DS = {
    "pregnancies": "Pregnancies", "glucose": "Glucose", "blood_pressure": "BloodPressure",
    "skin_thickness": "SkinThickness", "insulin": "Insulin", "bmi": "BMI",
    "dpf": "DiabetesPedigreeFunction", "age": "Age",
}


def _zeros_to_nan(X):
    X = pd.DataFrame(X, columns=FEATURE_COLS).copy()
    for col in ZERO_MISSING:
        X[col] = X[col].replace(0, np.nan)
    return X.values


def _synthetic_data(n=768, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 80, n)
    preg = np.clip(rng.poisson(3.8, n), 0, 17)
    bmi = np.clip(rng.normal(32, 7, n), 18, 67)
    glucose = np.clip(rng.normal(120, 32, n), 44, 199)
    bp = np.clip(rng.normal(69, 19, n), 24, 122)
    skin = np.clip(rng.normal(21, 16, n), 0, 99)
    insulin = np.clip(rng.gamma(2, 40, n), 0, 846)
    dpf = np.clip(rng.gamma(2, 0.25, n), 0.08, 2.4)
    z = -8.2 + 0.035 * glucose + 0.07 * bmi + 0.02 * age + 0.9 * dpf + 0.04 * preg
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"Pregnancies": preg, "Glucose": glucose.round(), "BloodPressure": bp.round(),
                         "SkinThickness": skin.round(), "Insulin": insulin.round(), "BMI": bmi.round(1),
                         "DiabetesPedigreeFunction": dpf.round(3), "Age": age, "Outcome": y})


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        missing = [c for c in FEATURE_COLS + ["Outcome"] if c not in df.columns]
        if missing:
            raise ValueError(f"data/diabetes.csv is missing columns: {missing}")
        return df, "Pima Indians Diabetes (data/diabetes.csv)"
    return _synthetic_data(), "SYNTHETIC DEMO DATA — add data/diabetes.csv for the real dataset"


def _make_pipeline(model):
    return Pipeline([
        ("zero_to_nan", FunctionTransformer(_zeros_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def _candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }


def train_and_compare():
    df, source = load_data()
    X, y = df[FEATURE_COLS], df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipelines, rows, test_preds = {}, [], {}
    for name, model in _candidates().items():
        pipe = _make_pipeline(model).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        pipelines[name] = pipe
        test_preds[name] = pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred, zero_division=0),
            "F1-score": f1_score(y_te, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te, proba),
        })
    metrics = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    best = metrics.loc[0, "Model"]

    # Explanation: permutation importance of the best model on held-out data
    imp = permutation_importance(pipelines[best], X_te, y_te, n_repeats=10, random_state=42, scoring="f1")
    importance = pd.Series(imp.importances_mean, index=FEATURE_COLS).clip(lower=0).sort_values()

    bundle = {
        "pipelines": pipelines, "metrics": metrics, "best": best, "importance": importance,
        "source": source, "n_rows": len(df),
        "class_balance": y.value_counts().to_dict(),
        "y_test": y_te.values, "y_pred": test_preds[best],
        "confusion": confusion_matrix(y_te, test_preds[best]),
        "dataset": df,
    }
    os.makedirs(os.path.dirname(BUNDLE_PATH), exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    return bundle


def load_or_train(force=False):
    if not force and os.path.exists(BUNDLE_PATH):
        try:
            return joblib.load(BUNDLE_PATH)
        except Exception:
            pass
    return train_and_compare()


def risk_category(p: float) -> str:
    return "Low" if p < 0.33 else ("Medium" if p < 0.66 else "High")


def predict(bundle, values_ds: dict, model_name=None):
    """values_ds uses dataset column names. Returns (probability, category, model_name)."""
    model_name = model_name or bundle["best"]
    X = pd.DataFrame([[values_ds[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    p = float(bundle["pipelines"][model_name].predict_proba(X)[0, 1])
    return p, risk_category(p), model_name
