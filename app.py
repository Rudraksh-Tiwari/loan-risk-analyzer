import streamlit as st
import pandas as pd
import numpy as np
import shap
from groq import Groq
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Loan Risk Analyzer",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    df = pd.read_csv("loan_data.csv")
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos,
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    xgb.fit(X_train, y_train)
    explainer = shap.TreeExplainer(xgb)

    return xgb, explainer, X_train.columns.tolist()

xgb, explainer, feature_names = load_model()

import os
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY", "your_key_here"))

def get_decision(risk):
    if risk < 0.3:
        return "APPROVED", "✅", "success"
    elif risk < 0.7:
        return "FLAGGED FOR REVIEW", "⚠️", "warning"
    else:
        return "REJECTED", "❌", "error"

def get_llm_explanation(prob, risk, decision, reasons):
    prompt = f"""You are a loan officer AI assistant at a bank.

A loan application has been analyzed by our risk system:

- Decision: {decision}
- Approval Probability: {prob:.0%}
- Risk Score: {risk:.0%}
- Top 3 factors:
  1. {reasons[0]}
  2. {reasons[1]}
  3. {reasons[2]}

Write a clear, professional explanation for the applicant."""

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

def build_input(age, income, emp_exp, loan_amt, int_rate,
                cred_hist, credit_score, gender, education,
                home, intent, prev_default):

    # FIX: Cap at 0.66 (actual dataset max), not 1.0 — prevents out-of-distribution values
    loan_pct = round(min(loan_amt / income, 0.66), 2)

    input_dict = {
        'person_age': age,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'loan_amnt': loan_amt,
        'loan_int_rate': int_rate,
        'loan_percent_income': loan_pct,
        'cb_person_cred_hist_length': cred_hist,
        'credit_score': credit_score,
        'person_gender_male': 1 if gender == "male" else 0,
        'person_education_Bachelor': 1 if education == "Bachelor" else 0,
        'person_education_Doctorate': 1 if education == "Doctorate" else 0,
        'person_education_High School': 1 if education == "High School" else 0,
        'person_education_Master': 1 if education == "Master" else 0,
        'person_home_ownership_OTHER': 1 if home == "OTHER" else 0,
        'person_home_ownership_OWN': 1 if home == "OWN" else 0,
        'person_home_ownership_RENT': 1 if home == "RENT" else 0,
        'loan_intent_DEBTCONSOLIDATION': 1 if intent == "DEBTCONSOLIDATION" else 0,  # FIX: was missing!
        'loan_intent_EDUCATION': 1 if intent == "EDUCATION" else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if intent == "HOMEIMPROVEMENT" else 0,
        'loan_intent_MEDICAL': 1 if intent == "MEDICAL" else 0,
        'loan_intent_PERSONAL': 1 if intent == "PERSONAL" else 0,
        'loan_intent_VENTURE': 1 if intent == "VENTURE" else 0,
        'previous_loan_defaults_on_file_Yes': 1 if prev_default == "Yes" else 0,
    }

    df = pd.DataFrame([input_dict])

    for col in feature_names:
        if col not in df:
            df[col] = 0

    df = df[feature_names]

    return df, loan_pct


# ─── UI ─────────────────────────────────────────────────────
st.title("🏦 Loan Risk Analyzer")
st.markdown("AI-powered loan decision system with explainable risk assessment")
st.divider()

st.subheader("Quick Load Sample Applicant")
p1, p2, p3 = st.columns(3)

if p1.button("✅ Good Applicant", use_container_width=True):
    st.session_state.preset = "good"
if p2.button("❌ Bad Applicant", use_container_width=True):
    st.session_state.preset = "bad"
if p3.button("⚠️ Borderline Case", use_container_width=True):
    st.session_state.preset = "borderline"

st.divider()

presets = {
    "good": {
        "age": 38, "gender": "male", "education": "Master",
        "income": 120000, "emp_exp": 10, "home": "OWN",
        "loan_amt": 4000, "intent": "EDUCATION", "int_rate": 6.0,
        "cred_hist": 15, "credit_score": 820, "prev_default": "No"
    },
    "bad": {
        "age": 21, "gender": "female", "education": "High School",
        "income": 340000, "emp_exp": 0, "home": "RENT",
        "loan_amt": 35000, "intent": "EDUCATION", "int_rate": 21.2,
        "cred_hist": 4, "credit_score": 481, "prev_default": "No"
    },
    "borderline": {
        "age": 28, "gender": "male", "education": "High School",
        "income": 50000, "emp_exp": 3, "home": "RENT",
        "loan_amt": 10000, "intent": "PERSONAL", "int_rate": 11.0,
        "cred_hist": 5, "credit_score": 620, "prev_default": "No"
    }
}

preset = st.session_state.get("preset", None)
vals = presets[preset] if preset else None

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Applicant Details")
    age = st.slider("Age", 18, 70, vals["age"] if vals else 28)
    gender = st.selectbox("Gender", ["male", "female"],
                          index=0 if not vals else ["male", "female"].index(vals["gender"]))
    education = st.selectbox("Education",
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
        index=0 if not vals else ["High School", "Associate", "Bachelor", "Master", "Doctorate"].index(vals["education"]))
    income = st.number_input("Annual Income (₹)",
        min_value=10000, max_value=500000, step=5000,
        value=vals["income"] if vals else 50000)
    emp_exp = st.slider("Employment Experience (years)", 0, 20,
        vals["emp_exp"] if vals else 3)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"],
        index=0 if not vals else ["RENT", "OWN", "MORTGAGE", "OTHER"].index(vals["home"]))

with col2:
    st.subheader("Loan Details")
    loan_amt = st.number_input("Loan Amount (₹)",
        min_value=500, max_value=35000, step=500,
        value=vals["loan_amt"] if vals else 10000)
    intent = st.selectbox("Loan Purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        index=0 if not vals else ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"].index(vals["intent"]))
    int_rate = st.slider("Interest Rate (%)", 5.0, 24.0,
        vals["int_rate"] if vals else 11.0, 0.1)

    loan_pct_display = round(min(loan_amt / income, 0.66), 2)  # FIX: consistent cap
    st.metric("Loan % of Income", f"{loan_pct_display:.0%}")

    cred_hist = st.slider("Credit History Length (years)", 1, 30,
        vals["cred_hist"] if vals else 5)
    credit_score = st.slider("Credit Score", 390, 850,  # FIX: min changed from 300 → 390 (dataset min)
        vals["credit_score"] if vals else 650)
    prev_default = st.selectbox("Previous Loan Defaults", ["No", "Yes"],
        index=0 if not vals else ["No", "Yes"].index(vals["prev_default"]))

st.divider()

# ─── PREDICT ────────────────────────────────────────────────
if st.button("🔍 Analyze Loan Application", use_container_width=True):

    input_df, loan_pct = build_input(
        age, income, emp_exp, loan_amt, int_rate,
        cred_hist, credit_score, gender, education,
        home, intent, prev_default
    )

    default_prob = xgb.predict_proba(input_df)[0][1]
    prob = 1 - default_prob
    risk = default_prob

    decision, icon, dtype = get_decision(risk)

    shap_vals = explainer.shap_values(input_df)[0]
    shap_vals_risk = -shap_vals

    top_idx = abs(shap_vals_risk).argsort()[-3:][::-1]

    reasons = []
    for i in top_idx:
        direction = "increases risk" if shap_vals_risk[i] > 0 else "decreases risk"
        reasons.append(
            f"{feature_names[i]} = {input_df.iloc[0][feature_names[i]]:.2f} ({direction})"
        )

    st.subheader("Analysis Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Decision", f"{icon} {decision}")
    r2.metric("Approval Probability", f"{prob:.0%}")
    r3.metric("Risk Score", f"{risk:.0%}")

    if dtype == "success":
        st.success("✅ Loan Approved — Low risk applicant")
    elif dtype == "warning":
        st.warning("⚠️ Flagged for Manual Review")
    else:
        st.error("❌ Loan Rejected — High risk applicant")

    st.subheader("Top Risk Factors")
    for i, reason in enumerate(reasons):
        st.write(f"**{i+1}.** {reason}")

    st.subheader("AI Explanation")
    with st.spinner("Generating explanation..."):
        explanation = get_llm_explanation(prob, risk, decision, reasons)
    st.info(explanation)
