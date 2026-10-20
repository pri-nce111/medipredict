# MediPredict — AI-Based Disease Risk Prediction & Healthcare Analytics

Python · Streamlit · Scikit-learn · Pandas · SQLite · Matplotlib

## Run
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Download the Pima Indians Diabetes CSV (Kaggle: "Pima Indians Diabetes Database", or UCI) and save it as
`data/diabetes.csv` with columns:
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`.
Without it the app uses a labelled synthetic demo dataset. After adding the CSV, delete `models/bundle.joblib`
(or click "Retrain models") to retrain.

## Default admin
`admin@medipredict.local` / `Admin@1234` — change/delete after first login.

## Structure
- `database.py` — SQLite tables, password hashing (PBKDF2), validation, CRUD
- `ml_engine.py` — preprocessing, 4-model comparison, metrics, explanation, prediction
- `report.py` — downloadable summary
- `app.py` — Streamlit UI (login, dashboard, prediction, history, model comparison, admin)

*Educational project, not a medical device.*
