import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Loan Risk Analyzer",
    page_icon="🏦",
    layout="wide"
)

# ─── Load & train (cached so it only runs once) ────────────
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
    xgb = XGBClassifier(scale_pos_weight=scale_pos, n_estimators=100,
                        random_state=42, eval_metric='logloss', verbosity=0)
    xgb.fit(X_train, y_train)

    explainer = shap.TreeExplainer(xgb)
    return xgb, explainer, X_train.columns.tolist()

xgb, explainer, feature_names = load_model()

# ─── Groq client ───────────────────────────────────────────
import os
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY", "YOUR_API_KEY"))

# ─── Helper functions ──────────────────────────────────────
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

Write a clear, professional 3-4 sentence explanation for the applicant.
- If APPROVED: be positive, mention key strengths
- If REJECTED: be empathetic, explain reasons, suggest improvements
- If FLAGGED FOR REVIEW: explain it needs human review and why
- Do NOT use technical ML terms
- Write as if speaking directly to the applicant"""

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

# ─── UI ────────────────────────────────────────────────────
st.title("Loan Risk Analyzer")
st.markdown("AI-powered loan decision system with explainable risk assessment")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Applicant Details")

    age = st.slider("Age", 18, 70, 28)
    gender = st.selectbox("Gender", ["male", "female"])
    education = st.selectbox("Education", 
                  ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    income = st.number_input("Annual Income (₹)", 
                  min_value=10000, max_value=500000, value=50000, step=5000)
    emp_exp = st.slider("Employment Experience (years)", 0, 20, 3)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    st.subheader("Loan Details")

    loan_amt = st.number_input("Loan Amount (₹)", 
                  min_value=500, max_value=35000, value=10000, step=500)
    intent = st.selectbox("Loan Purpose", 
                  ["PERSONAL", "EDUCATION", "MEDICAL", 
                   "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    int_rate = st.slider("Interest Rate (%)", 5.0, 24.0, 11.0, 0.1)
    loan_pct = round(loan_amt / income, 2)
    st.metric("Loan % of Income", f"{loan_pct:.0%}")
    cred_hist = st.slider("Credit History Length (years)", 1, 30, 5)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    prev_default = st.selectbox("Previous Loan Defaults", ["No", "Yes"])

st.divider()

# ─── Predict button ────────────────────────────────────────
if st.button(" Analyze Loan Application", use_container_width=True):

    # Build input matching training columns
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

    input_df = pd.DataFrame([input_dict])[feature_names]

    # Predict
    prob = xgb.predict_proba(input_df)[0][1]
    risk = 1 - prob
    decision, icon, dtype = get_decision(risk)

    # SHAP
    shap_vals = explainer.shap_values(input_df)[0]
    top_idx = abs(shap_vals).argsort()[-3:][::-1]
    reasons = []
    for i in top_idx:
        arrow = "increases risk" if shap_vals[i] > 0 else "decreases risk"
        reasons.append(f"{feature_names[i]} = {input_df.iloc[0][feature_names[i]]:.2f} ({arrow})")

    # ─── Results ───────────────────────────────────────────
    st.subheader("Analysis Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Decision", f"{icon} {decision}")
    r2.metric("Approval Probability", f"{prob:.0%}")
    r3.metric("Risk Score", f"{risk:.0%}")

    # Decision banner
    if dtype == "success":
        st.success(f"✅ Loan Approved — Low risk applicant")
    elif dtype == "warning":
        st.warning(f"⚠️ Flagged for Manual Review")
    else:
        st.error(f"❌ Loan Rejected — High risk applicant")

    # SHAP reasons
    st.subheader("Top Risk Factors")
    for i, reason in enumerate(reasons):
        st.write(f"**{i+1}.** {reason}")

    # LLM explanation
    st.subheader("AI Explanation")
    with st.spinner("Generating explanation..."):
        explanation = get_llm_explanation(prob, risk, decision, reasons)
    st.info(explanation)