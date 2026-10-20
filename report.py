"""Builds the downloadable prediction summary (plain-text/Markdown)."""
from datetime import datetime

PREVENTIVE_INFO = [
    "Maintain a healthy body weight; even modest weight loss lowers diabetes risk.",
    "Aim for at least 150 minutes of moderate physical activity per week.",
    "Favour whole grains, vegetables, legumes and fibre; limit sugary drinks and ultra-processed food.",
    "Do not smoke, and keep alcohol intake moderate.",
    "Have blood glucose, blood pressure and cholesterol checked regularly, especially with a family history.",
]
SOURCES = [
    "World Health Organization — Diabetes fact sheet: https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "CDC — Diabetes prevention: https://www.cdc.gov/diabetes/prevention/",
    "NHS — Type 2 diabetes: https://www.nhs.uk/conditions/type-2-diabetes/",
]
DISCLAIMER = (
    "LIMITATIONS: This is an educational machine-learning tool, not a medical device. It was trained on a "
    "small public dataset (mostly adult women of Pima heritage) and may not generalise to other "
    "populations. Predictions are statistical estimates, not diagnoses. Always consult a qualified "
    "healthcare professional for diagnosis, treatment and medical decisions."
)

LABELS = {
    "Pregnancies": "Pregnancies", "Glucose": "Glucose (mg/dL)", "BloodPressure": "Blood pressure (mm Hg)",
    "SkinThickness": "Skin thickness (mm)", "Insulin": "Insulin (µU/mL)", "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes pedigree function", "Age": "Age",
}


def build_report(patient_name, values_ds, model_name, probability, risk, importance, metrics_row=None):
    lines = [
        "# MediPredict — Disease Risk Prediction Summary",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"**Patient:** {patient_name}",
        f"**Condition assessed:** Diabetes risk",
        f"**Model used:** {model_name}",
        f"**Predicted probability:** {probability:.1%}",
        f"**Risk category:** {risk}",
        "",
        "## Input values",
    ]
    lines += [f"- {LABELS[k]}: {v}" for k, v in values_ds.items()]
    lines += ["", "## Most influential factors (model-wide)"]
    top = importance.sort_values(ascending=False).head(4)
    lines += [f"- {LABELS[k]} (importance {v:.3f})" for k, v in top.items()]
    if metrics_row is not None:
        lines += ["", "## Model performance on held-out test data",
                  f"Accuracy {metrics_row['Accuracy']:.2f} · Precision {metrics_row['Precision']:.2f} · "
                  f"Recall {metrics_row['Recall']:.2f} · F1 {metrics_row['F1-score']:.2f}"]
    lines += ["", "## General preventive-health information"] + [f"- {t}" for t in PREVENTIVE_INFO]
    lines += ["", "## Reliable sources"] + [f"- {s}" for s in SOURCES]
    lines += ["", "## Limitations & advice", DISCLAIMER]
    if risk in ("Medium", "High"):
        lines += ["", "**Recommendation:** Please arrange a check-up with a doctor to discuss these results."]
    return "\n".join(lines)
