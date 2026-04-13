#  Loan Risk Analyzer

An AI-powered loan approval prediction system that combines Machine Learning, 
Explainable AI (SHAP), and LLM integration to make and explain loan decisions 
like a real bank.

---

##  What This Project Does

Takes a loan applicant's details → predicts approval probability → explains 
the decision in plain English using an LLM.

**Example output:**
> *"I regret to inform you that we are unable to approve your loan application 
> at this time. Our review revealed that your current income may not be 
> sufficient to support the loan payments..."*

---

##  Live Demo
> Coming soon — Streamlit Cloud deployment

---

##  Tech Stack

| Layer            | Technology                                  |
| ---------------- | ------------------------------------------- |
| Programming      | Python 3.11                                 |
| Machine Learning | Logistic Regression, Random Forest, XGBoost |
| Explainability   | SHAP                                        |
| LLM Integration  | Llama 3.3 70B via Groq API                  |
| User Interface   | Streamlit                                   |
| Dataset          | 45,000 loan applicants                      |


---

##  Model Results

| Model               | AUC-ROC | F1 Score (Approved) | F1 Score (Rejected) |
| ------------------- | ------- | ------------------- | ------------------- |
| Logistic Regression | 0.9525  | 0.74                | 0.90                |
| Random Forest       | 0.9743  | 0.83                | 0.95                |
| XGBoost             | 0.9782  | 0.83                | 0.95                |


XGBoost selected as final model — highest AUC-ROC and best minority class F1.

---

##  How It Works


Applicant Data
↓
XGBoost Model (AUC-ROC: 0.9782)
↓
Risk Score = 1 - Approval Probability
↓
┌─────────────────────────────────┐
│ Risk < 30%  → APPROVE           │
│ Risk 30-70% → REVIEW            │
│ Risk > 70%  → REJECT            │
└─────────────────────────────────┘
↓
SHAP Values → Top 3 Risk Factors
↓
Groq LLM → Plain English Explanation


---

##  Key Features

- **Imbalanced data handling** — 78/22 class split handled via 
  `scale_pos_weight` in XGBoost
- **SHAP explainability** — every decision explained by top 3 contributing 
  factors (regulatory requirement in real banking)
- **LLM integration** — Llama 3.3 70B converts ML output into professional 
  bank-style letters
- **Risk engine** — 3-tier decision system (Approve / Review / Reject) 
  instead of binary prediction
- **Real dataset** — 45,000 applicants with credit score, income, 
  employment history, loan intent

---

##  How To Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/loan-risk-analyzer.git
cd loan-risk-analyzer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your Groq API key**
```bash
# Windows
set GROQ_API_KEY=your_key_here

# Mac/Linux
export GROQ_API_KEY=your_key_here
```

**4. Run the app**
```bash
python -m streamlit run app.py
```

---

##  Project Structure


loan-risk-analyzer/
│
├── app.py                  # Streamlit UI
├── loan_risk_model.ipynb   # Full ML pipeline notebook
├── loan_data.csv           # Dataset (45k applicants)
├── requirements.txt        # Dependencies
└── README.md               # This file


---

##  Requirements

Create a `requirements.txt` file:
pandas
numpy
scikit-learn
xgboost
shap
streamlit
groq
matplotlib
seaborn

---

##  What I Learned

- Handling class imbalance in real financial datasets
- Why AUC-ROC matters more than accuracy for imbalanced problems
- SHAP values for regulatory-grade model explainability
- Integrating LLMs into ML pipelines for human-readable outputs
- End-to-end ML deployment with Streamlit

---

##  Author
**Rudraksh**  
[GitHub](https://github.com/Rudraksh-Tiwari) •  
[LinkedIn](https://www.linkedin.com/in/rudraksh-tiwari-b6083129b/)
