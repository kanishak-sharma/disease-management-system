import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Management System", page_icon="🩺", layout="centered")

TITLES = {
    "diabetes": "Diabetes",
    "heart": "Heart Disease",
    "parkinsons": "Parkinson's Disease",
}

@st.cache_resource
def load_model(name):
    return joblib.load(f"models/{name}.joblib")

st.title("🩺 Multi-Disease Prediction System")
st.caption("Decision support only. This is not a medical diagnosis.")

disease = st.sidebar.radio("Select disease", list(TITLES), format_func=TITLES.get)
bundle = load_model(disease)

st.subheader(f"{TITLES[disease]}: patient details")

with st.form("patient_form"):
    cols = st.columns(2)
    values = {}
    for i, feature in enumerate(bundle["features"]):
        default = float(bundle["defaults"][feature])
        values[feature] = cols[i % 2].number_input(feature, value=default)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([values], columns=bundle["features"])
    pipe = bundle["pipeline"]
    pred = int(pipe.predict(row)[0])
    prob = float(pipe.predict_proba(row)[0][1])

    st.progress(prob, text=f"Model risk score: {prob:.0%}")
    if pred == 1:
        st.error(f"Higher risk indicated for {TITLES[disease]}. Please consult a doctor.")
    else:
        st.success(f"Lower risk indicated for {TITLES[disease]}.")
    st.caption("This tool supports decisions and does not replace a medical professional.")