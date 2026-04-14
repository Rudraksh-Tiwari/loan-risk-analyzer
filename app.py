import streamlit as st
import pandas as pd
import numpy as np
import shap
from groq import Groq
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="LoanRisk AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS injection — use one markdown call for HTML and style ───────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; }

/* ── App background ── */
.stApp { background: #080c10 !important; }

/* ── Subtle grid overlay ── */
.stApp > div:first-child::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(0,212,180,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,180,0.035) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Remove default padding ── */
.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Global font ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    color: #e8edf2 !important;
}

/* ── Preset buttons ── */
.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #6b7f8e !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: rgba(0,212,180,0.4) !important;
    color: #00d4b4 !important;
    background: rgba(0,212,180,0.05) !important;
    box-shadow: none !important;
}
.stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #e8edf2 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    background: transparent !important;
}
[data-baseweb="popover"] > div { background: #0e1520 !important; }
[data-baseweb="menu"] { background: #0e1520 !important; border: 1px solid rgba(0,212,180,0.12) !important; }
li[role="option"] { color: #e8edf2 !important; font-family: 'DM Mono', monospace !important; font-size: 13px !important; }
li[role="option"]:hover { background: rgba(0,212,180,0.08) !important; }

/* ── Number input ── */
input[type="number"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #e8edf2 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}
.stNumberInput button {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #6b7f8e !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background: rgba(255,255,255,0.08) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #00d4b4 !important;
    border-color: #00d4b4 !important;
    box-shadow: 0 0 10px rgba(0,212,180,0.4) !important;
}
[data-testid="stThumbValue"] {
    color: #00d4b4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    background: transparent !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #2a3a4a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
}

/* ── Labels (slider, select, number input) ── */
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"],
.stSlider label {
    color: #4a5a6a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,212,180,0.08) !important;
    margin: 18px 0 !important;
}

/* ── Metric card ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] > div {
    color: #4a5a6a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] > div {
    color: #e8edf2 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-color: rgba(0,212,180,0.2) !important;
    border-top-color: #00d4b4 !important;
}

/* ── Column gap fix ── */
[data-testid="column"] { padding-left: 6px !important; padding-right: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ─── MODEL ────────────────────────────────────────────────────────────────────
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

import os
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY", "your_key_here"))

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_decision(risk):
    if risk < 0.3:
        return "APPROVED", "success"
    elif risk < 0.7:
        return "FLAGGED FOR REVIEW", "warning"
    else:
        return "REJECTED", "error"

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

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content

def build_input(age, income, emp_exp, loan_amt, int_rate,
                cred_hist, credit_score, gender, education,
                home, intent, prev_default):
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
        'loan_intent_DEBTCONSOLIDATION': 1 if intent == "DEBTCONSOLIDATION" else 0,
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
    return df[feature_names], loan_pct

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

# ─── TOPBAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    display:flex; align-items:center; justify-content:space-between;
    padding:14px 36px;
    border-bottom:1px solid rgba(0,212,180,0.12);
    background:rgba(8,12,16,0.97);
    position:sticky; top:0; z-index:999;
">
    <div style="display:flex;align-items:center;gap:12px">
        <div style="
            width:36px;height:36px;border-radius:8px;
            background:linear-gradient(135deg,#00d4b4,#0066ff);
            display:flex;align-items:center;justify-content:center;
            font-size:14px;font-weight:800;color:#080c10;
            font-family:'Syne',sans-serif;flex-shrink:0;
        ">LR</div>
        <div>
            <div style="font-size:16px;font-weight:700;color:#fff;font-family:'Syne',sans-serif;line-height:1.2">LoanRisk AI</div>
            <div style="font-size:9px;color:#00d4b4;letter-spacing:0.2em;font-family:'DM Mono',monospace;text-transform:uppercase">Credit Intelligence Platform</div>
        </div>
    </div>
    <div style="
        display:flex;align-items:center;gap:8px;
        padding:6px 16px;border-radius:20px;
        border:1px solid rgba(0,212,180,0.2);
        background:rgba(0,212,180,0.04);
    ">
        <div style="
            width:7px;height:7px;border-radius:50%;
            background:#00d4b4;box-shadow:0 0 8px #00d4b4;
            animation:none;
        "></div>
        <span style="font-size:10px;color:#6b7f8e;font-family:'DM Mono',monospace;letter-spacing:0.12em">
            MODEL LIVE &nbsp;·&nbsp; XGBoost + SHAP + Groq
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── LAYOUT HELPERS ───────────────────────────────────────────────────────────
def sec(label):
    st.markdown(f"""
    <div style="font-size:9px;letter-spacing:0.22em;text-transform:uppercase;
                color:#00d4b4;font-family:'DM Mono',monospace;
                display:flex;align-items:center;gap:10px;
                margin-bottom:14px;margin-top:6px;">
        {label}
        <span style="flex:1;height:1px;background:rgba(0,212,180,0.15);display:inline-block"></span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
# QUICK LOAD — full width
# ═════════════════════════════════════════════════════════
sec("Quick load sample")
b1, b2, b3 = st.columns(3)
if b1.button("✓  Good applicant"):
    st.session_state.preset = "good";       st.rerun()
if b2.button("✕  Bad applicant"):
    st.session_state.preset = "bad";        st.rerun()
if b3.button("⚠  Borderline"):
    st.session_state.preset = "borderline"; st.rerun()

st.divider()

# ═════════════════════════════════════════════════════════
# FORM — Applicant Details (left) | Loan Details (right)
# ═════════════════════════════════════════════════════════
left, right = st.columns(2, gap="large")

with left:
    sec("Applicant details")

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 70, vals["age"] if vals else 28)
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
        min_value=10000, max_value=500000, step=5000,
        value=vals["income"] if vals else 50000)

    emp_exp = st.slider("Employment experience (years)", 0, 20,
        vals["emp_exp"] if vals else 3)

with right:
    sec("Loan details")

    loan_amt = st.number_input("Loan amount (₹)",
        min_value=500, max_value=35000, step=500,
        value=vals["loan_amt"] if vals else 10000)

    c5, c6 = st.columns(2)
    with c5:
        intent = st.selectbox("Loan purpose",
            ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"],
            index=0 if not vals else ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"].index(vals["intent"]))
    with c6:
        prev_default = st.selectbox("Previous defaults", ["No","Yes"],
            index=0 if not vals else ["No","Yes"].index(vals["prev_default"]))

    int_rate = st.slider("Interest rate (%)", 5.0, 24.0,
        vals["int_rate"] if vals else 11.0, 0.1)

    credit_score = st.slider("Credit score", 390, 850,
        vals["credit_score"] if vals else 650)

    c7, c8 = st.columns(2)
    with c7:
        cred_hist = st.slider("Credit history (yrs)", 1, 30,
            vals["cred_hist"] if vals else 5)
    with c8:
        loan_pct_display = round(min(loan_amt / income, 0.66), 2)
        st.metric("Loan % of income", f"{loan_pct_display:.0%}")

st.divider()

# ── Analyze button — full width ───────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="secondary"]:last-of-type {
    background: linear-gradient(135deg, #00d4b4 0%, #0066ff 100%) !important;
    border: none !important; color: #080c10 !important;
    font-weight: 700 !important; font-size: 13px !important;
    letter-spacing: 0.1em !important; padding: 14px !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)
analyze = st.button("⟡  ANALYZE APPLICATION", use_container_width=True)

# ═════════════════════════════════════════════════════════
# RESULTS — full width below form
# ═════════════════════════════════════════════════════════
if analyze:
    input_df, loan_pct = build_input(
        age, income, emp_exp, loan_amt, int_rate,
        cred_hist, credit_score, gender, education,
        home, intent, prev_default
    )

    default_prob = xgb.predict_proba(input_df)[0][1]
    risk  = default_prob
    prob  = 1 - default_prob
    decision, dtype = get_decision(risk)

    if dtype == "success":
        color, bg   = "#00d4b4", "rgba(0,212,180,0.06)"
        border      = "rgba(0,212,180,0.22)"
        badge_label = "LOW RISK"
    elif dtype == "warning":
        color, bg   = "#ffb400", "rgba(255,180,0,0.06)"
        border      = "rgba(255,180,0,0.22)"
        badge_label = "NEEDS REVIEW"
    else:
        color, bg   = "#ff5050", "rgba(255,80,80,0.06)"
        border      = "rgba(255,80,80,0.22)"
        badge_label = "HIGH RISK"

    st.divider()
    sec("Analysis results")

    # Decision banner — full width
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:10px;
                padding:22px 28px;margin-bottom:16px;
                display:flex;align-items:center;justify-content:space-between">
        <div>
            <div style="font-size:9px;letter-spacing:0.2em;text-transform:uppercase;
                        color:#4a5a6a;font-family:'DM Mono',monospace;margin-bottom:6px">
                Decision
            </div>
            <div style="font-size:34px;font-weight:800;color:{color};
                        font-family:'Syne',sans-serif;letter-spacing:0.03em;line-height:1">
                {decision}
            </div>
        </div>
        <div style="padding:8px 20px;border-radius:20px;background:rgba(0,0,0,0.2);
                    border:1px solid {border};font-size:10px;font-weight:600;
                    color:{color};font-family:'DM Mono',monospace;letter-spacing:0.14em">
            {badge_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics — 4 cols across full width
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Approval probability", f"{prob:.0%}")
    m2.metric("Risk score", f"{risk:.0%}")
    m3.metric("Interest rate", f"{int_rate:.1f}%")
    m4.metric("Loan % of income", f"{loan_pct_display:.0%}")

    # Risk bar
    st.markdown(f"""
    <div style="margin:16px 0">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span style="font-size:9px;font-family:'DM Mono',monospace;
                         color:#2a3a4a;letter-spacing:0.1em;text-transform:uppercase">Low risk</span>
            <span style="font-size:9px;font-family:'DM Mono',monospace;
                         color:#2a3a4a;letter-spacing:0.1em;text-transform:uppercase">High risk</span>
        </div>
        <div style="height:5px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden">
            <div style="height:100%;width:{risk*100:.1f}%;border-radius:3px;
                        background:linear-gradient(90deg,#00d4b4,{color})"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # SHAP + AI side by side
    shap_vals      = explainer.shap_values(input_df)[0]
    shap_vals_risk = -shap_vals
    top_idx        = abs(shap_vals_risk).argsort()[-3:][::-1]

    reasons = []
    for i in top_idx:
        d = "increases risk" if shap_vals_risk[i] > 0 else "decreases risk"
        reasons.append(
            f"{feature_names[i]} = {input_df.iloc[0][feature_names[i]]:.2f} ({d})"
        )

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        sec("Top risk factors")
        for idx, i in enumerate(top_idx):
            up    = shap_vals_risk[i] > 0
            tc    = "#ff5050" if up else "#00d4b4"
            tbg   = "rgba(255,80,80,0.08)"  if up else "rgba(0,212,180,0.08)"
            tbd   = "rgba(255,80,80,0.18)"  if up else "rgba(0,212,180,0.18)"
            arrow = "▲ raises risk"         if up else "▼ lowers risk"
            clean = (feature_names[i]
                     .replace("person_","").replace("loan_","").replace("cb_","")
                     .replace("_"," ").title())
            rv    = input_df.iloc[0][feature_names[i]]
            dv    = f"{int(rv):,}" if rv == int(rv) else f"{rv:.2f}"

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.05);
                        border-radius:7px;padding:12px 16px;margin-bottom:8px">
                <span style="font-size:10px;font-family:'DM Mono',monospace;
                             color:#2a3a4a;min-width:22px;font-weight:500">0{idx+1}</span>
                <span style="font-size:12px;font-family:'DM Mono',monospace;
                             color:#6b7f8e;flex:1">{clean}</span>
                <span style="font-size:12px;font-family:'DM Mono',monospace;
                             color:#c8d4e0;font-weight:500">{dv}</span>
                <span style="font-size:9px;padding:3px 10px;border-radius:4px;
                             background:{tbg};color:{tc};border:1px solid {tbd};
                             font-family:'DM Mono',monospace;
                             letter-spacing:0.06em;white-space:nowrap">{arrow}</span>
            </div>
            """, unsafe_allow_html=True)

    with res_right:
        sec("AI explanation")
        with st.spinner("Generating..."):
            explanation = get_llm_explanation(prob, risk, decision, reasons)

        st.markdown(f"""
        <div style="background:rgba(0,102,255,0.05);
                    border:1px solid rgba(0,102,255,0.14);
                    border-radius:8px;padding:18px 20px;height:100%">
            <div style="font-size:9px;letter-spacing:0.18em;text-transform:uppercase;
                        color:#4477cc;font-family:'DM Mono',monospace;margin-bottom:12px">
                ◆ &nbsp;Groq · Llama 3.3 70B
            </div>
            <div style="font-size:14px;color:#8a9ab0;line-height:1.85;
                        font-family:'Syne',sans-serif">
                {explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
