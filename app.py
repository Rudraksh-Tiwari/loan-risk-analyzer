import json
import os
import warnings

import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Loan Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Theme: "Ledger" — light, institutional, dense ──
st.html("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; }

.stApp { background: #f3f2ef !important; }

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] { display: none !important; }

.block-container {
    padding: 1.4rem 2rem 80px 2rem !important;
    max-width: 1220px !important;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #1c1c1a !important;
}

/* Buttons — base (preset / secondary) */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid #e3e1da !important;
    color: #46453f !important;
    border-radius: 7px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 10px 8px !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: #1d3a5f !important;
    color: #1d3a5f !important;
    background: #f6f8fa !important;
}
.stButton > button:focus { box-shadow: none !important; outline: none !important; }

/* Primary (Analyze) */
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: #1d3a5f !important;
    border: 1px solid #1d3a5f !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 13px !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: #16314f !important;
    border-color: #16314f !important;
    color: #ffffff !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #e3e1da !important;
    border-radius: 7px !important;
}
[data-baseweb="select"] span {
    color: #1c1c1a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}
[data-baseweb="popover"] > div { background: #ffffff !important; }
[data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1px solid #e3e1da !important;
}
li[role="option"] {
    color: #1c1c1a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}
li[role="option"]:hover { background: #f1f4f7 !important; }

/* Number input */
input[type="number"] {
    background: #ffffff !important;
    border: 1px solid #e3e1da !important;
    color: #1c1c1a !important;
    border-radius: 7px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}
.stNumberInput button {
    background: #f6f5f2 !important;
    border-color: #e3e1da !important;
    color: #8a887f !important;
}

/* Widget labels — sentence case, muted (no uppercase tracking) */
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
    color: #8a887f !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid #e6e4dd !important;
    margin: 18px 0 !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e6e4dd !important;
    border-radius: 8px !important;
    padding: 13px 16px !important;
}
[data-testid="stMetricLabel"] > div {
    color: #8a887f !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
[data-testid="stMetricValue"] > div {
    color: #1c1c1a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 24px !important;
    font-weight: 500 !important;
}

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #1d3a5f !important; }

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Column padding */
[data-testid="column"] {
    padding-left: 8px !important;
    padding-right: 8px !important;
}
</style>
""")

# ─── MODEL ───
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
        scale_pos_weight=scale_pos, n_estimators=100,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    xgb.fit(X_train, y_train)
    explainer = shap.TreeExplainer(xgb)
    return xgb, explainer, X_train.columns.tolist()

xgb, explainer, feature_names = load_model()

HARDCODED_GROQ_API_KEY = ""


def get_groq_api_key():
    key = HARDCODED_GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if key:
        return key
    # st.secrets raises if no secrets.toml exists anywhere — don't let that
    # crash the app; AI summaries just stay disabled until a key is provided.
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


GROQ_API_KEY = get_groq_api_key()
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ─── HELPERS ─────
def get_decision(risk):
    if risk < 0.3:   return "Approved",            "success"
    elif risk < 0.7: return "Flagged for review",  "warning"
    else:            return "Rejected",            "error"

def get_llm_explanation(prob, risk, decision, reasons):
    prompt = f"""You are a loan officer. Write 2-3 SHORT sentences directly to the applicant. No greetings, no sign-offs, no placeholders like [Name] or [Bank].

Decision: {decision}
Approval chance: {prob:.0%}
Top factors: {reasons[0]} | {reasons[1]} | {reasons[2]}

Rules:
- APPROVED: mention 1-2 strengths, keep it warm and brief
- REJECTED: name the main reason, suggest one concrete improvement
- FLAGGED: say a human will review it and roughly why
- No formal letter format. No ML jargon. Plain English only. Max 3 sentences."""

    if not client_groq:
        return (
            "AI explanation unavailable because GROQ_API_KEY is not configured. "
            "Set GROQ_API_KEY in your environment before starting the app."
        )

    try:
        response = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "AI explanation could not be generated. Verify GROQ_API_KEY and network access, "
            "then restart the app."
        )


TRAINING_RANGES = {
    "Age": (20, 144),
    "Annual income": (8000, 7200766),
    "Employment experience": (0, 125),
    "Loan amount": (500, 35000),
    "Interest rate": (5.42, 20.0),
    "Credit score": (390, 850),
    "Credit history": (2, 30),
    "Loan percent of income": (0.0, 0.66),
}


def loan_percent_income(loan_amt, income):
    if income <= 0:
        return 0
    return round(loan_amt / income, 4)


def range_note(label, value):
    low, high = TRAINING_RANGES[label]
    if value < low:
        return f"{label} is below the training range ({low:g}-{high:g})."
    if value > high:
        return f"{label} is above the training range ({low:g}-{high:g})."
    return None


def collect_input_notes(age, income, emp_exp, loan_amt, int_rate,
                        cred_hist, credit_score, loan_pct):
    checks = {
        "Age": age,
        "Annual income": income,
        "Employment experience": emp_exp,
        "Loan amount": loan_amt,
        "Interest rate": int_rate,
        "Credit score": credit_score,
        "Credit history": cred_hist,
        "Loan percent of income": loan_pct,
    }
    notes = [note for label, value in checks.items()
             if (note := range_note(label, value))]

    if emp_exp > max(age - 14, 0):
        notes.append("Employment experience is unusually high for the applicant age.")
    if cred_hist > max(age - 18, 0):
        notes.append("Credit history is unusually high for the applicant age.")
    if loan_amt > income:
        notes.append("Loan amount is greater than annual income.")

    return notes


def build_input(age, income, emp_exp, loan_amt, int_rate,
                cred_hist, credit_score, gender, education,
                home, intent, prev_default):
    loan_pct = loan_percent_income(loan_amt, income)
    input_dict = {
        'person_age': age, 'person_income': income,
        'person_emp_exp': emp_exp, 'loan_amnt': loan_amt,
        'loan_int_rate': int_rate, 'loan_percent_income': loan_pct,
        'cb_person_cred_hist_length': cred_hist, 'credit_score': credit_score,
        'person_gender_male': 1 if gender == "male" else 0,
        'person_education_Bachelor':   1 if education == "Bachelor"   else 0,
        'person_education_Doctorate':  1 if education == "Doctorate"  else 0,
        'person_education_High School':1 if education == "High School"else 0,
        'person_education_Master':     1 if education == "Master"     else 0,
        'person_home_ownership_OTHER':  1 if home == "OTHER"    else 0,
        'person_home_ownership_OWN':    1 if home == "OWN"      else 0,
        'person_home_ownership_RENT':   1 if home == "RENT"     else 0,
        'loan_intent_DEBTCONSOLIDATION':1 if intent == "DEBTCONSOLIDATION" else 0,
        'loan_intent_EDUCATION':        1 if intent == "EDUCATION"         else 0,
        'loan_intent_HOMEIMPROVEMENT':  1 if intent == "HOMEIMPROVEMENT"   else 0,
        'loan_intent_MEDICAL':          1 if intent == "MEDICAL"           else 0,
        'loan_intent_PERSONAL':         1 if intent == "PERSONAL"          else 0,
        'loan_intent_VENTURE':          1 if intent == "VENTURE"           else 0,
        'previous_loan_defaults_on_file_Yes': 1 if prev_default == "Yes" else 0,
    }
    df = pd.DataFrame([input_dict])
    for col in feature_names:
        if col not in df: df[col] = 0
    return df[feature_names], loan_pct


def round_offer_amount(amount):
    if amount < 500:
        return max(int(amount), 1)
    return int(round(amount / 500) * 500)


def estimate_risk_for_amount(age, income, emp_exp, loan_amt, int_rate,
                             cred_hist, credit_score, gender, education,
                             home, intent, prev_default):
    offer_df, _ = build_input(
        age, income, emp_exp, loan_amt, int_rate,
        cred_hist, credit_score, gender, education,
        home, intent, prev_default
    )
    return float(xgb.predict_proba(offer_df)[0][1])


def calculate_loan_offer(age, income, emp_exp, requested_amt, int_rate,
                         cred_hist, credit_score, gender, education,
                         home, intent, prev_default):
    if requested_amt <= 1:
        return 1, estimate_risk_for_amount(
            age, income, emp_exp, 1, int_rate, cred_hist, credit_score,
            gender, education, home, intent, prev_default
        ), "approved"

    candidates = np.linspace(1, requested_amt, 80)
    scored = []
    for amount in candidates:
        rounded_amount = min(round_offer_amount(amount), requested_amt)
        risk_at_amount = estimate_risk_for_amount(
            age, income, emp_exp, rounded_amount, int_rate,
            cred_hist, credit_score, gender, education, home, intent,
            prev_default
        )
        scored.append((rounded_amount, risk_at_amount))

    approved = [row for row in scored if row[1] < 0.3]
    if approved:
        return max(approved, key=lambda row: row[0]) + ("approved",)

    review = [row for row in scored if row[1] < 0.7]
    if review:
        return max(review, key=lambda row: row[0]) + ("review",)

    fallback_offer = min(
        round_offer_amount(min(requested_amt, max(income * 0.10, 1))),
        requested_amt
    )
    fallback_risk = estimate_risk_for_amount(
        age, income, emp_exp, fallback_offer, int_rate,
        cred_hist, credit_score, gender, education, home, intent, prev_default
    )
    return fallback_offer, fallback_risk, "manual_review"


def get_llm_loan_offer(requested_amt, offer_amt, offer_risk, offer_status,
                       decision, risk):
    prompt = f"""You are a bank loan officer. Write 2-3 short sentences to the applicant.

Original decision: {decision}
Requested loan amount: ₹{requested_amt:,.0f}
Requested loan risk score: {risk:.0%}
Recommended offer amount: ₹{offer_amt:,.0f}
Recommended offer risk score: {offer_risk:.0%}
Offer status: {offer_status}

Rules:
- If the requested amount is not suitable, clearly say the bank cannot offer the full requested amount.
- State the recommended amount the bank may offer for this situation.
- Mention that the offer is based on the applicant profile and risk assessment.
- Do not use ML jargon. No greeting or sign-off. Max 3 sentences."""

    if not client_groq:
        if offer_amt < requested_amt:
            return (
                f"We cannot offer the full requested amount of ₹{requested_amt:,.0f} "
                f"for this profile. Based on the current risk assessment, the bank may "
                f"offer approximately ₹{offer_amt:,.0f} instead."
            )
        return (
            f"The requested amount of ₹{requested_amt:,.0f} is within the amount "
            "the bank may offer for this profile."
        )

    try:
        response = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=140
        )
        return response.choices[0].message.content
    except Exception:
        return (
            f"The bank may offer approximately ₹{offer_amt:,.0f} for this situation, "
            f"but cannot confirm the full requested amount of ₹{requested_amt:,.0f}."
        )


def section(label):
    """Renders a styled section header (sentence-case, hairline rule)."""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:20px 0 13px">
        <span style="font-size:11px;letter-spacing:0.04em;
                     color:#8a887f;font-family:'IBM Plex Mono',monospace;white-space:nowrap">
            {label}
        </span>
        <div style="flex:1;height:1px;background:#e6e4dd"></div>
    </div>
    """, unsafe_allow_html=True)

# ─── PRESETS ──────────────────────────────────────────────────────────────────
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
vals   = presets[preset] if preset else None


# ─── TOP BAR ──────────────────────────────────────────────────────────────────
st.html("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:6px 0 16px;border-bottom:1px solid #e3e1da;margin-bottom:6px;">
    <div style="display:flex;align-items:center;gap:12px">
        <div style="width:38px;height:38px;border-radius:9px;flex-shrink:0;
                    background:#1d3a5f;display:flex;align-items:center;justify-content:center;">
            <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#ffffff"
                 stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 21h18"/><path d="M3 10h18"/><path d="M5 6l7-3 7 3"/>
                <path d="M4 10v11"/><path d="M20 10v11"/>
                <path d="M8 14v3"/><path d="M12 14v3"/><path d="M16 14v3"/>
            </svg>
        </div>
        <div>
            <div style="font-size:18px;font-weight:600;color:#1c1c1a;
                        font-family:'IBM Plex Sans',sans-serif;line-height:1.2">
                Loan Risk Analyzer
            </div>
            <div style="font-size:11px;color:#8a887f;font-family:'IBM Plex Sans',sans-serif">
                Explainable credit scoring
            </div>
        </div>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#46453f;
                border:1px solid #e3e1da;border-radius:7px;padding:6px 12px;background:#ffffff">
        AUC-ROC 0.978 &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; 45k applicants
    </div>
</div>
""")

if not GROQ_API_KEY:
    st.warning(
        "GROQ_API_KEY is not configured — the model still runs, but AI summaries are "
        "disabled. Add it to .streamlit/secrets.toml or set it as an environment variable."
    )


# ─── TWO-PANE LAYOUT: inputs (left) · results (right) ─────────────────────────
left, right = st.columns([5, 7], gap="large")

with left:
    # QUICK LOAD
    section("Quick load")
    b1, b2, b3 = st.columns(3)
    if b1.button("Good applicant"):
        st.session_state.preset = "good";       st.rerun()
    if b2.button("Bad applicant"):
        st.session_state.preset = "bad";        st.rerun()
    if b3.button("Borderline"):
        st.session_state.preset = "borderline"; st.rerun()

    st.divider()

    # APPLICANT DETAILS
    section("Applicant details")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age",
            min_value=18, step=1,
            value=vals["age"] if vals else 28)
    with c2:
        gender = st.selectbox("Gender", ["male", "female"],
            index=0 if not vals else ["male","female"].index(vals["gender"]))

    c3, c4 = st.columns(2)
    with c3:
        education = st.selectbox("Education",
            ["High School","Associate","Bachelor","Master","Doctorate"],
            index=0 if not vals else ["High School","Associate","Bachelor","Master","Doctorate"].index(vals["education"]))
    with c4:
        home = st.selectbox("Home ownership", ["RENT","OWN","MORTGAGE","OTHER"],
            index=0 if not vals else ["RENT","OWN","MORTGAGE","OTHER"].index(vals["home"]))

    income = st.number_input("Annual income (₹)",
        min_value=1, step=5000,
        value=vals["income"] if vals else 50000)

    emp_exp = st.number_input("Employment experience (years)",
        min_value=0, step=1,
        value=vals["emp_exp"] if vals else 3)

    st.divider()

    # LOAN DETAILS
    section("Loan details")

    loan_amt = st.number_input("Loan amount (₹)",
        min_value=1, step=500,
        value=vals["loan_amt"] if vals else 10000)

    c5, c6 = st.columns(2)
    with c5:
        intent = st.selectbox("Loan purpose",
            ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"],
            index=0 if not vals else ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"].index(vals["intent"]))
    with c6:
        prev_default = st.selectbox("Previous defaults", ["No","Yes"],
            index=0 if not vals else ["No","Yes"].index(vals["prev_default"]))

    int_rate = st.number_input("Interest rate (%)",
        min_value=0.0, max_value=50.0, step=0.1,
        value=float(vals["int_rate"] if vals else 11.0))

    credit_score = st.number_input("Credit score",
        min_value=300, max_value=900, step=1,
        value=vals["credit_score"] if vals else 650)

    c7, c8 = st.columns(2)
    with c7:
        cred_hist = st.number_input("Credit history (years)",
            min_value=0, step=1,
            value=vals["cred_hist"] if vals else 5)
    with c8:
        loan_pct_display = loan_percent_income(loan_amt, income)
        st.metric("Loan % of income", f"{loan_pct_display:.0%}")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    analyze = st.button("Analyze application", use_container_width=True, type="primary")


with right:
    if not analyze:
        # ── Idle state: model card (uses the space, surfaces the stats) ──
        st.markdown("""
        <div style="border:1px solid #e3e1da;border-radius:10px;background:#ffffff;padding:22px 24px">
            <div style="font-size:11px;color:#8a887f;font-family:'IBM Plex Mono',monospace;margin-bottom:16px">
                Model
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
                <div>
                    <div style="font-size:27px;font-weight:600;color:#1d3a5f;font-family:'IBM Plex Sans',sans-serif">0.978</div>
                    <div style="font-size:11px;color:#8a887f;font-family:'IBM Plex Mono',monospace;margin-top:2px">AUC-ROC</div>
                </div>
                <div>
                    <div style="font-size:27px;font-weight:600;color:#1c1c1a;font-family:'IBM Plex Sans',sans-serif">45,000</div>
                    <div style="font-size:11px;color:#8a887f;font-family:'IBM Plex Mono',monospace;margin-top:2px">Applicants</div>
                </div>
                <div>
                    <div style="font-size:27px;font-weight:600;color:#1c1c1a;font-family:'IBM Plex Sans',sans-serif">XGBoost</div>
                    <div style="font-size:11px;color:#8a887f;font-family:'IBM Plex Mono',monospace;margin-top:2px">Model</div>
                </div>
            </div>

            <hr style="border:none;border-top:1px solid #ece9e2;margin:20px 0">

            <div style="font-size:11px;color:#8a887f;font-family:'IBM Plex Mono',monospace;margin-bottom:13px">
                Decision thresholds
            </div>
            <div style="display:flex;flex-direction:column;gap:11px">
                <div style="display:flex;align-items:center;gap:11px;font-size:13px;color:#46453f">
                    <span style="width:8px;height:8px;border-radius:50%;background:#2e6b4f;flex-shrink:0"></span>
                    <span style="width:70px">Approve</span>
                    <span style="color:#8a887f;font-family:'IBM Plex Mono',monospace">Risk below 30%</span>
                </div>
                <div style="display:flex;align-items:center;gap:11px;font-size:13px;color:#46453f">
                    <span style="width:8px;height:8px;border-radius:50%;background:#b5790a;flex-shrink:0"></span>
                    <span style="width:70px">Review</span>
                    <span style="color:#8a887f;font-family:'IBM Plex Mono',monospace">Risk 30% – 70%</span>
                </div>
                <div style="display:flex;align-items:center;gap:11px;font-size:13px;color:#46453f">
                    <span style="width:8px;height:8px;border-radius:50%;background:#a3392f;flex-shrink:0"></span>
                    <span style="width:70px">Reject</span>
                    <span style="color:#8a887f;font-family:'IBM Plex Mono',monospace">Risk above 70%</span>
                </div>
            </div>

            <hr style="border:none;border-top:1px solid #ece9e2;margin:20px 0">

            <div style="font-size:11px;color:#8a887f;font-family:'IBM Plex Mono',monospace;margin-bottom:11px">
                How it works
            </div>
            <div style="font-size:13px;color:#46453f;line-height:1.7;font-family:'IBM Plex Sans',sans-serif">
                Enter the applicant's profile, then XGBoost scores default risk, SHAP surfaces
                the factors behind the score, and Llama 3.3 70B writes a plain-language summary.
            </div>

            <div style="margin-top:18px;font-size:13px;color:#1d3a5f;font-family:'IBM Plex Sans',sans-serif">
                Fill in the details on the left, then press Analyze application.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        input_df, loan_pct = build_input(
            age, income, emp_exp, loan_amt, int_rate,
            cred_hist, credit_score, gender, education,
            home, intent, prev_default
        )
        input_notes = collect_input_notes(
            age, income, emp_exp, loan_amt, int_rate,
            cred_hist, credit_score, loan_pct
        )

        default_prob = xgb.predict_proba(input_df)[0][1]
        risk  = default_prob
        prob  = 1 - default_prob
        decision, dtype = get_decision(risk)
        offer_amt, offer_risk, offer_status = calculate_loan_offer(
            age, income, emp_exp, loan_amt, int_rate,
            cred_hist, credit_score, gender, education,
            home, intent, prev_default
        )

        if dtype == "success":
            color, bg   = "#2e6b4f", "#eef4ef"
            border      = "#cfe3d6"
            badge_label = "Low risk"
        elif dtype == "warning":
            color, bg   = "#b5790a", "#fbf3e3"
            border      = "#ecdcb3"
            badge_label = "Needs review"
        else:
            color, bg   = "#a3392f", "#fbeeec"
            border      = "#eccfc9"
            badge_label = "High risk"

        section("Decision")

        if input_notes:
            st.warning(
                "Result generated, but some values are outside or near the edge of "
                "the training data: " + " ".join(input_notes)
            )

        # ── Decision banner ──
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;
                    padding:20px 24px;margin-bottom:16px;
                    display:flex;align-items:center;justify-content:space-between">
            <div>
                <div style="font-size:11px;color:#8a887f;
                            font-family:'IBM Plex Mono',monospace;margin-bottom:6px">
                    Decision
                </div>
                <div style="font-size:32px;font-weight:600;color:{color};
                            font-family:'IBM Plex Sans',sans-serif;line-height:1">
                    {decision}
                </div>
            </div>
            <div style="padding:7px 16px;border-radius:7px;background:#ffffff;
                        border:1px solid {border};font-size:12px;color:{color};
                        font-family:'IBM Plex Mono',monospace">
                {badge_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ──
        m1, m2 = st.columns(2)
        m1.metric("Approval probability", f"{prob:.0%}")
        m2.metric("Risk score",           f"{risk:.0%}")

        # ── Risk bar ──
        st.markdown(f"""
        <div style="margin:14px 0 4px">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                <span style="font-size:11px;font-family:'IBM Plex Mono',monospace;color:#a3a097">
                    Low risk
                </span>
                <span style="font-size:11px;font-family:'IBM Plex Mono',monospace;color:#a3a097">
                    High risk
                </span>
            </div>
            <div style="height:6px;background:#ece9e2;border-radius:4px;overflow:hidden">
                <div style="height:100%;width:{risk*100:.1f}%;border-radius:4px;background:{color}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── SHAP factors ──
        shap_vals      = explainer.shap_values(input_df)[0]
        shap_vals_risk = -shap_vals
        top_idx        = abs(shap_vals_risk).argsort()[-3:][::-1]

        reasons = []
        for i in top_idx:
            d = "increases risk" if shap_vals_risk[i] > 0 else "decreases risk"
            reasons.append(
                f"{feature_names[i]} = {input_df.iloc[0][feature_names[i]]:.2f} ({d})"
            )

        section("What's driving this · SHAP")

        top8_idx = abs(shap_vals_risk).argsort()[-8:][::-1]

        def clean_name(fn):
            return (fn.replace("person_","").replace("loan_","")
                      .replace("cb_","").replace("_"," ").title())

        chart_rows = []
        for i in top8_idx:
            sv   = float(shap_vals_risk[i])
            rv   = float(input_df.iloc[0][feature_names[i]])
            dv   = f"{int(rv):,}" if rv == int(rv) else f"{rv:.2f}"
            chart_rows.append({
                "label": clean_name(feature_names[i]),
                "value": dv,
                "shap":  round(sv, 4),
            })

        base_val   = float(explainer.expected_value)
        final_val  = float(xgb.predict_proba(input_df)[0][1])
        max_abs    = max(abs(r["shap"]) for r in chart_rows) or 1

        rows_json     = json.dumps(chart_rows)
        base_json     = json.dumps(round(base_val, 4))
        final_json    = json.dumps(round(final_val, 4))
        maxabs_json   = json.dumps(round(max_abs, 4))

        shap_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: transparent;
    font-family: 'IBM Plex Mono', monospace;
    color: #46453f;
    padding: 0 2px 28px;
  }}

  .chart-wrap {{ width: 100%; }}

  .row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
    position: relative;
    cursor: default;
  }}
  .row:hover .bar-fill {{ filter: brightness(0.92); }}
  .row:hover .tooltip  {{ opacity: 1; pointer-events: auto; }}

  .feat-label {{
    font-size: 11px;
    color: #6b6a64;
    width: 148px;
    min-width: 148px;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: 0.02em;
  }}

  .bar-track {{
    flex: 1;
    height: 26px;
    background: #ece9e2;
    border-radius: 5px;
    position: relative;
    overflow: visible;
  }}

  .bar-track::after {{
    content: '';
    position: absolute;
    left: 50%;
    top: 0; bottom: 0;
    width: 1px;
    background: #d8d5cc;
    transform: translateX(-50%);
  }}

  .bar-fill {{
    position: absolute;
    top: 3px; bottom: 3px;
    border-radius: 3px;
    transition: width 0.55s cubic-bezier(.22,.61,.36,1),
                left  0.55s cubic-bezier(.22,.61,.36,1);
  }}
  .bar-fill.risk  {{ background: #b5483a; }}
  .bar-fill.safe  {{ background: #2e6b4f; }}

  .shap-val {{
    font-size: 10px;
    min-width: 52px;
    text-align: left;
    letter-spacing: 0.04em;
    font-weight: 500;
  }}
  .shap-val.risk {{ color: #b5483a; }}
  .shap-val.safe {{ color: #2e6b4f; }}

  .tooltip {{
    position: absolute;
    left: 162px;
    top: -38px;
    background: #ffffff;
    border: 1px solid #e3e1da;
    border-radius: 7px;
    padding: 7px 12px;
    font-size: 10px;
    color: #46453f;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
    z-index: 99;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
  }}
  .tooltip span {{ color: #1d3a5f; }}

  .summary {{
    display: flex;
    align-items: center;
    gap: 0;
    margin: 18px 0 4px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e6e4dd;
    background: #faf9f6;
  }}
  .s-block {{
    flex: 1;
    padding: 10px 14px;
    border-right: 1px solid #ece9e2;
  }}
  .s-block:last-child {{ border-right: none; }}
  .s-label {{
    font-size: 9px;
    letter-spacing: 0.06em;
    color: #a3a097;
    margin-bottom: 4px;
  }}
  .s-val {{
    font-size: 18px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
  }}
  .s-val.base   {{ color: #8a887f; }}
  .s-val.pos    {{ color: #b5483a; }}
  .s-val.neg    {{ color: #2e6b4f; }}
  .s-val.final  {{ color: #1c1c1a; }}

  .legend {{
    display: flex;
    gap: 20px;
    margin-bottom: 14px;
    justify-content: flex-end;
  }}
  .leg-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 9px;
    color: #8a887f;
    letter-spacing: 0.04em;
  }}
  .leg-dot {{
    width: 8px; height: 8px;
    border-radius: 2px;
  }}
</style>
</head>
<body>
<div class="chart-wrap">

  <div class="legend">
    <div class="leg-item">
      <div class="leg-dot" style="background:#b5483a"></div>
      Raises risk
    </div>
    <div class="leg-item">
      <div class="leg-dot" style="background:#2e6b4f"></div>
      Lowers risk
    </div>
  </div>

  <div id="rows"></div>

  <div class="summary" id="summary"></div>

</div>

<script>
const rows     = {rows_json};
const baseVal  = {base_json};
const finalVal = {final_json};
const maxAbs   = {maxabs_json};

const container = document.getElementById('rows');

rows.forEach(r => {{
  const pct    = (Math.abs(r.shap) / maxAbs) * 46;
  const isRisk = r.shap > 0;
  const cls    = isRisk ? 'risk' : 'safe';
  const sign   = isRisk ? '+' : '';

  const left  = isRisk ? '50%' : `${{50 - pct}}%`;
  const width = `${{pct}}%`;

  const div = document.createElement('div');
  div.className = 'row';
  div.innerHTML = `
    <div class="feat-label" title="${{r.label}}">${{r.label}}</div>
    <div class="bar-track">
      <div class="bar-fill ${{cls}}"
           style="left:${{left}};width:0%"
           data-left="${{left}}" data-width="${{width}}">
      </div>
      <div class="tooltip">
        ${{r.label}} = <span>${{r.value}}</span> &nbsp;|&nbsp; SHAP: <span>${{sign}}${{r.shap.toFixed(4)}}</span>
      </div>
    </div>
    <div class="shap-val ${{cls}}">${{sign}}${{r.shap.toFixed(3)}}</div>
  `;
  container.appendChild(div);
}});

requestAnimationFrame(() => {{
  setTimeout(() => {{
    document.querySelectorAll('.bar-fill').forEach(el => {{
      el.style.width = el.dataset.width;
      el.style.left  = el.dataset.left;
    }});
  }}, 60);
}});

const posSum = rows.filter(r=>r.shap>0).reduce((a,r)=>a+r.shap,0);
const negSum = rows.filter(r=>r.shap<0).reduce((a,r)=>a+r.shap,0);
const summary = document.getElementById('summary');
summary.innerHTML = `
  <div class="s-block">
    <div class="s-label">Base rate</div>
    <div class="s-val base">${{(baseVal*100).toFixed(1)}}%</div>
  </div>
  <div class="s-block">
    <div class="s-label">Risk factors</div>
    <div class="s-val pos">+${{(posSum*100).toFixed(1)}}%</div>
  </div>
  <div class="s-block">
    <div class="s-label">Safe factors</div>
    <div class="s-val neg">${{(negSum*100).toFixed(1)}}%</div>
  </div>
  <div class="s-block">
    <div class="s-label">Final risk score</div>
    <div class="s-val final">${{(finalVal*100).toFixed(1)}}%</div>
  </div>
`;
</script>
</body>
</html>
"""

        components.html(shap_html, height=420, scrolling=False)

        st.divider()

        # ── Summary ──
        section("Summary")
        with st.spinner("Generating…"):
            explanation = get_llm_explanation(prob, risk, decision, reasons)

        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e3e1da;
                    border-radius:9px;padding:18px 20px">
            <div style="font-size:11px;color:#8a887f;
                        font-family:'IBM Plex Mono',monospace;margin-bottom:10px">
                Plain-language summary
            </div>
            <div style="font-size:14px;color:#46453f;line-height:1.75;
                        font-family:'IBM Plex Sans',sans-serif">
                {explanation}
            </div>
            <div style="font-size:11px;color:#b3b1a8;
                        font-family:'IBM Plex Mono',monospace;margin-top:12px">
                Generated by Llama 3.3 70B
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        section("Suggested offer")

        o1, o2 = st.columns(2)
        o1.metric("Requested amount", f"₹ {loan_amt:,.0f}")
        o2.metric("Suggested offer",  f"₹ {offer_amt:,.0f}")

        with st.spinner("Preparing offer…"):
            offer_explanation = get_llm_loan_offer(
                loan_amt, offer_amt, offer_risk, offer_status, decision, risk
            )

        st.markdown(f"""
        <div style="background:#eef3f8;border:1px solid #d9e2ec;
                    border-radius:9px;padding:18px 20px">
            <div style="font-size:11px;color:#3a5a7a;
                        font-family:'IBM Plex Mono',monospace;margin-bottom:10px">
                Recommendation
            </div>
            <div style="font-size:14px;color:#3d4654;line-height:1.75;
                        font-family:'IBM Plex Sans',sans-serif">
                {offer_explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)
