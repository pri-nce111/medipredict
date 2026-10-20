"""MediPredict — Streamlit web app.   Run:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import database as db
import ml_engine as ml
import report as rpt

st.set_page_config(page_title="MediPredict", page_icon="🩺", layout="wide")
db.init_db()


@st.cache_resource(show_spinner="Training models (first run only)...")
def get_bundle():
    return ml.load_or_train()


bundle = get_bundle()
RISK_COLOR = {"Low": "#2e9e5b", "Medium": "#e0a800", "High": "#d64545"}


# ------------------------------------------------------------------ auth UI
def auth_page():
    st.title("🩺 MediPredict")
    st.caption("AI-Based Disease Risk Prediction and Healthcare Analytics System")
    tab_login, tab_reg = st.tabs(["Log in", "Create account"])
    with tab_login:
        with st.form("login"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                user = db.authenticate(email, pw)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Incorrect email or password.")
    with tab_reg:
        with st.form("register"):
            email = st.text_input("Email", key="reg_email")
            pw = st.text_input("Password (8+ chars, letters and digits)", type="password", key="reg_pw")
            pw2 = st.text_input("Confirm password", type="password")
            if st.form_submit_button("Create account"):
                if pw != pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db.register_user(email, pw)
                    (st.success if ok else st.error)(msg)


# ------------------------------------------------------------------ helpers
def scope_user_id(user):
    return None if user["role"] == "admin" else user["id"]


def importance_chart(imp):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh([rpt.LABELS[c] for c in imp.index], imp.values, color="#2a7de1")
    ax.set_xlabel("Permutation importance (drop in F1)")
    ax.set_title("What drives the model's predictions")
    fig.tight_layout()
    return fig


def input_form(prefix, defaults=None):
    """Renders validated number inputs, returns dict keyed by dataset column names."""
    values = {}
    cols = st.columns(2)
    for i, (col, (label, lo, hi, dflt)) in enumerate(ml.FEATURES.items()):
        start = (defaults or {}).get(col, dflt)
        is_float = isinstance(lo, float) or isinstance(hi, float)
        with cols[i % 2]:
            if is_float:
                values[col] = st.number_input(label, float(lo), float(hi), float(start), step=0.1,
                                              key=f"{prefix}_{col}", format="%.2f")
            else:
                values[col] = st.number_input(label, int(lo), int(hi), int(start), step=1, key=f"{prefix}_{col}")
    return values


def ds_to_db(values_ds):
    inv = {v: k for k, v in ml.DB_TO_DS.items()}
    return {inv[k]: v for k, v in values_ds.items()}


# ------------------------------------------------------------------ pages
def page_predict(user):
    st.header("New Prediction")
    st.write("Enter the patient's details. All fields are range-checked.")
    model_choice = st.selectbox("Model", list(bundle["pipelines"]),
                                index=list(bundle["pipelines"]).index(bundle["best"]),
                                help="The best model by F1-score is preselected.")
    name = st.text_input("Patient name / ID", placeholder="e.g. John Doe")
    values = input_form("new")
    if st.button("Run prediction", type="primary"):
        if not name.strip():
            st.error("Please enter a patient name or ID.")
            return
        p, risk, used = ml.predict(bundle, values, model_choice)
        db.save_record(user["id"], name.strip(), ds_to_db(values), used, p, risk)
        st.session_state["last"] = dict(name=name.strip(), values=values, p=p, risk=risk, model=used)

    last = st.session_state.get("last")
    if last:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Diabetes risk probability", f"{last['p']:.1%}")
        c2.markdown(f"**Risk category**<br><span style='font-size:2rem;color:{RISK_COLOR[last['risk']]}'>"
                    f"{last['risk']}</span>", unsafe_allow_html=True)
        c3.metric("Model", last["model"])
        st.progress(min(max(last["p"], 0.0), 1.0))
        st.pyplot(importance_chart(bundle["importance"]))
        row = bundle["metrics"][bundle["metrics"]["Model"] == last["model"]].iloc[0]
        text = rpt.build_report(last["name"], last["values"], last["model"], last["p"], last["risk"],
                                bundle["importance"], row)
        st.download_button("⬇ Download prediction summary", text,
                           file_name=f"medipredict_{last['name'].replace(' ', '_')}.md", mime="text/markdown")
        with st.expander("Preventive-health information & limitations"):
            for t in rpt.PREVENTIVE_INFO:
                st.write("• " + t)
            st.caption("Sources: WHO, CDC, NHS (links are in the downloadable report).")
            st.warning(rpt.DISCLAIMER)
        st.success("Saved to prediction history.")


def page_dashboard(user):
    st.header("Healthcare Analytics Dashboard")
    hist = pd.DataFrame(db.get_history(scope_user_id(user)))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total predictions", len(hist))
    c2.metric("High-risk cases", int((hist["risk_category"] == "High").sum()) if len(hist) else 0)
    c3.metric("Avg. risk", f"{hist['probability'].mean():.1%}" if len(hist) else "—")
    c4.metric("Best model F1", f"{bundle['metrics'].loc[0, 'F1-score']:.2f}")
    st.caption(f"Dataset: {bundle['source']} · {bundle['n_rows']} rows")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk categories (saved predictions)")
        if len(hist):
            counts = hist["risk_category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.bar(counts.index, counts.values, color=[RISK_COLOR[k] for k in counts.index])
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No predictions yet — run one first.")
    with right:
        st.subheader("Glucose distribution by outcome")
        df = bundle["dataset"]
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for label, color, nm in [(0, "#2e9e5b", "No diabetes"), (1, "#d64545", "Diabetes")]:
            vals = df.loc[(df["Outcome"] == label) & (df["Glucose"] > 0), "Glucose"]
            ax.hist(vals, bins=20, alpha=0.6, color=color, label=nm)
        ax.set_xlabel("Glucose")
        ax.legend()
        st.pyplot(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("BMI vs Age (dataset)")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        d = df[df["BMI"] > 0]
        ax.scatter(d["Age"], d["BMI"], c=d["Outcome"].map({0: "#2e9e5b", 1: "#d64545"}), s=10, alpha=0.6)
        ax.set_xlabel("Age")
        ax.set_ylabel("BMI")
        st.pyplot(fig)
    with right:
        st.subheader(f"Actual vs predicted — {bundle['best']}")
        cm = bundle["confusion"]
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in pd.DataFrame(cm).stack().items():
            ax.text(j, i, int(v), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["Pred: No", "Pred: Yes"])
        ax.set_yticks([0, 1], ["Actual: No", "Actual: Yes"])
        st.pyplot(fig)

    st.subheader("Model evaluation (20% held-out test set)")
    st.dataframe(bundle["metrics"].style.format({c: "{:.3f}" for c in bundle["metrics"].columns[1:]}),
                 width="stretch")


def page_models():
    st.header("Model Comparison (Research)")
    m = bundle["metrics"]
    st.dataframe(m.style.format({c: "{:.3f}" for c in m.columns[1:]}), width="stretch")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = range(len(m))
    w = 0.2
    for k, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-score"]):
        ax.bar([i + k * w for i in x], m[metric], width=w, label=metric)
    ax.set_xticks([i + 1.5 * w for i in x], m["Model"], rotation=10)
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, loc="lower center")
    st.pyplot(fig)
    st.write(f"**Best model by F1-score:** {bundle['best']}")
    st.caption(f"Class balance (0/1): {bundle['class_balance']}. Recall matters most in screening because a "
               "missed positive case is costlier than a false alarm.")
    if st.button("Retrain models"):
        st.cache_resource.clear()
        ml.load_or_train(force=True)
        st.rerun()
    st.pyplot(importance_chart(bundle["importance"]))


def page_history(user):
    st.header("Prediction History")
    search = st.text_input("Search by patient name")
    rows = db.get_history(scope_user_id(user), search)
    if not rows:
        st.info("No records found.")
        return
    df = pd.DataFrame(rows)
    show = df[["patient_id", "created_at", "name", "age", "glucose", "bmi", "model_name",
               "probability", "risk_category"] + (["owner"] if user["role"] == "admin" else [])]
    st.dataframe(show.style.format({"probability": "{:.1%}"}), width="stretch", hide_index=True)

    st.subheader("Edit or delete a record")
    pid = st.selectbox("Select patient record", df["patient_id"],
                       format_func=lambda i: f"#{i} — {df.loc[df.patient_id == i, 'name'].iloc[0]}")
    rec = df[df["patient_id"] == pid].iloc[0]
    if user["role"] != "admin" and rec["user_id"] != user["id"]:
        st.error("Not authorised.")
        return
    defaults = {ds: rec[dbc] for dbc, ds in ml.DB_TO_DS.items()}
    with st.expander("Edit (re-runs the prediction)"):
        new_name = st.text_input("Name", rec["name"], key=f"en_{pid}")
        new_vals = input_form(f"edit{pid}", defaults)
        if st.button("Save changes"):
            p, risk, used = ml.predict(bundle, new_vals, bundle["best"])
            db.update_record(int(pid), new_name, ds_to_db(new_vals), used, p, risk)
            st.success("Record updated.")
            st.rerun()
    if st.button("🗑 Delete this record"):
        db.delete_record(int(pid))
        st.rerun()
    st.download_button("⬇ Export history (CSV)", show.to_csv(index=False), "history.csv", "text/csv")


def page_admin():
    st.header("Administration")
    users = pd.DataFrame(db.list_users())
    st.dataframe(users, width="stretch", hide_index=True)
    uid = st.selectbox("User to remove", users["id"],
                       format_func=lambda i: users.loc[users.id == i, "email"].iloc[0])
    if st.button("Delete user and their records"):
        if uid == st.session_state["user"]["id"]:
            st.error("You cannot delete your own account.")
        else:
            db.delete_user(int(uid))
            st.rerun()


# ------------------------------------------------------------------ main
user = st.session_state.get("user")
if not user:
    auth_page()
    st.stop()

with st.sidebar:
    st.title("🩺 MediPredict")
    st.caption(f"{user['email']} · {user['role']}")
    pages = ["Dashboard", "New Prediction", "History", "Model Comparison"]
    if user["role"] == "admin":
        pages.append("Admin")
    page = st.radio("Navigate", pages)
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.caption("Educational tool — not a medical diagnosis.")

{"Dashboard": lambda: page_dashboard(user), "New Prediction": lambda: page_predict(user),
 "History": lambda: page_history(user), "Model Comparison": page_models,
 "Admin": page_admin}[page]()
