
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

# ─── MODEL ────────────────────────────────────────────────
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

    model = XGBClassifier(
        scale_pos_weight=scale_pos,
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    return model, explainer, X_train.columns.tolist()

model, explainer, feature_names = load_model()

# ─── GROQ ────────────────────────────────────────────────
import os
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY", "your_key_here"))

# ─── DECISION LOGIC ───────────────────────────────────────
def get_decision(risk):
    if risk < 0.3:
        return "APPROVED", "✅", "success"
    elif risk < 0.7:
        return "FLAGGED FOR REVIEW", "⚠️", "warning"
    else:
        return "REJECTED", "❌", "error"

# ─── LLM EXPLANATION ─────────────────────────────────────
def get_llm_explanation(prob, risk, decision, reasons):
    prompt = f"""You are a loan officer AI assistant at a bank.

A loan application has been analyzed:

- Decision: {decision}
- Approval Probability: {prob:.0%}
- Risk Score: {risk:.0%}
- Top 3 factors:
  1. {reasons[0]}
  2. {reasons[1]}
  3. {reasons[2]}

Write a clear, professional explanation.
Avoid technical ML terms.
Speak directly to the applicant."""

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    return response.choices[0].message.content

# ─── INPUT BUILDER ───────────────────────────────────────
def build_input(age, income, emp_exp, loan_amt, int_rate,
                cred_hist, credit_score, gender, education,
                home, intent, prev_default):

    loan_pct = round(min(loan_amt / income, 1.0), 2)

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
        'loan_intent_EDUCATION': 1 if intent == "EDUCATION" else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if intent == "HOMEIMPROVEMENT" else 0,
        'loan_intent_MEDICAL': 1 if intent == "MEDICAL" else 0,
        'loan_intent_PERSONAL': 1 if intent == "PERSONAL" else 0,
        'loan_intent_VENTURE': 1 if intent == "VENTURE" else 0,
        'previous_loan_defaults_on_file_Yes': 1 if prev_default == "Yes" else 0,
    }

    df = pd.DataFrame([input_dict])

    # ensure all columns match training
    for col in feature_names:
        if col not in df:
            df[col] = 0

    df = df[feature_names]

    return df, loan_pct

# ─── UI ──────────────────────────────────────────────────
st.title("🏦 Loan Risk Analyzer")
st.markdown("AI-powered loan decision system with explainable risk assessment")
st.divider()

# ─── PRESETS ─────────────────────────────────────────────
st.subheader("Quick Load Sample Applicant")

p1, p2, p3 = st.columns(3)

if p1.button("✅ Good Applicant"):
    st.session_state.preset = "good"
if p2.button("❌ Bad Applicant"):
    st.session_state.preset = "bad"
if p3.button("⚠️ Borderline"):
    st.session_state.preset = "borderline"

presets = {
    "good": {
        "age": 38, "gender": "male", "education": "Master",
        "income": 120000, "emp_exp": 10, "home": "OWN",
        "loan_amt": 4000, "intent": "EDUCATION", "int_rate": 6.0,
        "cred_hist": 15, "credit_score": 820, "prev_default": "No"
    },
    "bad": {
        "age": 21, "gender": "female", "education": "High School",
        "income": 15000, "emp_exp": 0, "home": "RENT",
        "loan_amt": 12000, "intent": "PERSONAL", "int_rate": 23.0,
        "cred_hist": 1, "credit_score": 320, "prev_default": "Yes"
    },
    "borderline": {
        "age": 28, "gender": "male", "education": "High School",
        "income": 50000, "emp_exp": 3, "home": "RENT",
        "loan_amt": 10000, "intent": "PERSONAL", "int_rate": 11.0,
        "cred_hist": 5, "credit_score": 650, "prev_default": "No"
    }
}

preset = st.session_state.get("preset", None)
vals = presets[preset] if preset else None

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, vals["age"] if vals else 28)
    gender = st.selectbox("Gender", ["male", "female"])
    education = st.selectbox("Education",
                             ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    income = st.number_input("Income", 10000, 500000,
                             vals["income"] if vals else 50000)
    emp_exp = st.slider("Experience", 0, 20, vals["emp_exp"] if vals else 3)
    home = st.selectbox("Home", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    loan_amt = st.number_input("Loan", 500, 35000,
                               vals["loan_amt"] if vals else 10000)
    intent = st.selectbox("Purpose",
                          ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                           "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    int_rate = st.slider("Interest", 5.0, 24.0,
                          vals["int_rate"] if vals else 11.0)
    cred_hist = st.slider("Credit History", 1, 30,
                           vals["cred_hist"] if vals else 5)
    credit_score = st.slider("Credit Score", 300, 850,
                              vals["credit_score"] if vals else 650)
    prev_default = st.selectbox("Previous Default", ["No", "Yes"])

# ─── PREDICT ─────────────────────────────────────────────
if st.button("Analyze"):

    input_df, loan_pct = build_input(
        age, income, emp_exp, loan_amt, int_rate,
        cred_hist, credit_score, gender, education,
        home, intent, prev_default
    )

    # 🔥 FIXED PART
    default_prob = model.predict_proba(input_df)[0][1]
    prob = 1 - default_prob     # approval probability
    risk = default_prob         # risk = default probability

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

    st.subheader("Results")

    st.metric("Decision", f"{icon} {decision}")
    st.metric("Approval Probability", f"{prob:.0%}")
    st.metric("Risk Score", f"{risk:.0%}")

    if dtype == "success":
        st.success("Approved")
    elif dtype == "warning":
        st.warning("Review")
    else:
        st.error("Rejected")

    st.subheader("Top Factors")
    for r in reasons:
        st.write(r)

    st.subheader("Explanation")
    explanation = get_llm_explanation(prob, risk, decision, reasons)
    st.info(explanation)
```
