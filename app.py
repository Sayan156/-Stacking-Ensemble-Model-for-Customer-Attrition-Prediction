"""
ChurnShield — Bank Customer Attrition Predictor
Run:  streamlit run app.py
Deps: pip install streamlit scikit-learn xgboost pandas numpy plotly
Place stacking_model.pkl in the same folder as app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnShield · Bank Attrition Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #07090f; --surface: #0f1420; --surface2: #151c2e;
    --border: #1e2d4a; --gold: #d4a84b; --gold-dim: #9b7733;
    --cyan: #38bdf8; --danger: #f43f5e; --safe: #22d3a5;
    --text: #e2e8f0; --muted: #64748b; --radius: 12px;
}
html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; font-family: 'IBM Plex Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.main .block-container { padding: 2rem 3rem 4rem !important; max-width: 1300px !important; }

.hero { display:flex; align-items:center; gap:1.5rem; padding:2.2rem 2.5rem; background:linear-gradient(135deg,#0f1420,#0d1829,#0a1220); border:1px solid var(--border); border-radius:var(--radius); margin-bottom:2rem; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; inset:0; background:radial-gradient(ellipse 60% 80% at 90% 50%,rgba(212,168,75,0.06),transparent 70%); pointer-events:none; }
.hero-icon { font-size:3.2rem; filter:drop-shadow(0 0 18px rgba(212,168,75,0.6)); flex-shrink:0; }
.hero-title { font-family:'Playfair Display',serif !important; font-size:2.2rem !important; font-weight:900 !important; color:var(--gold) !important; margin:0 0 0.3rem !important; }
.hero-sub { font-family:'IBM Plex Mono',monospace !important; font-size:0.72rem !important; color:var(--muted) !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }
.hero-badge { margin-left:auto; background:rgba(212,168,75,0.08); border:1px solid var(--gold-dim); border-radius:8px; padding:0.6rem 1.2rem; text-align:center; flex-shrink:0; }
.hero-badge-label { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; display:block; margin-bottom:0.2rem; }
.hero-badge-val { font-family:'IBM Plex Mono',monospace; font-size:0.95rem; color:var(--gold); font-weight:500; }

.section-label { font-family:'IBM Plex Mono',monospace !important; font-size:0.62rem !important; letter-spacing:0.18em !important; text-transform:uppercase !important; color:var(--gold) !important; border-left:3px solid var(--gold); padding-left:0.7rem; margin-bottom:1rem; margin-top:0.5rem; display:block; }

.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.5rem 1.8rem; margin-bottom:1.2rem; transition:border-color 0.2s; }
.card:hover { border-color:var(--gold-dim); }

div[data-testid="stNumberInput"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stSlider"] > label {
    font-family:'IBM Plex Mono',monospace !important; font-size:0.7rem !important;
    color:var(--muted) !important; text-transform:uppercase !important; letter-spacing:0.07em !important;
}
div[data-baseweb="select"] { background:var(--surface2) !important; }

div.stButton > button {
    width:100% !important; padding:1rem 0 !important;
    background:linear-gradient(135deg,#b8892a,#d4a84b,#c9963a) !important;
    color:#07090f !important; font-family:'IBM Plex Mono',monospace !important;
    font-size:0.9rem !important; font-weight:500 !important; letter-spacing:0.12em !important;
    text-transform:uppercase !important; border:none !important; border-radius:10px !important;
    box-shadow:0 4px 24px rgba(212,168,75,0.25) !important;
}
div.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 32px rgba(212,168,75,0.4) !important; }

.derived-chip { display:inline-flex; align-items:center; gap:0.5rem; background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:6px; padding:0.5rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:var(--cyan); margin-top:0.5rem; }

.result-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.8rem 2rem; text-align:center; }
.result-card.danger { border-color:var(--danger); box-shadow:0 0 40px rgba(244,63,94,0.1); }
.result-card.safe   { border-color:var(--safe);   box-shadow:0 0 40px rgba(34,211,165,0.1); }
.result-verdict { font-family:'Playfair Display',serif !important; font-size:2rem !important; font-weight:700 !important; margin:0.5rem 0 0.3rem !important; }
.result-verdict.danger { color:var(--danger) !important; }
.result-verdict.safe   { color:var(--safe)   !important; }

.stat-row { display:flex; gap:0.8rem; justify-content:center; margin-top:1rem; flex-wrap:wrap; }
.stat-chip { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:0.5rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--muted); text-align:center; }
.stat-chip strong { display:block; font-size:1rem; color:var(--text); margin-top:0.2rem; }

.info-note { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:var(--muted); margin-top:0.3rem; }
hr { border-color:var(--border) !important; margin:1.5rem 0 !important; }
</style>
""")
# ─────────────────────────────────────────────────────────────────────────────
# EXACT FEATURE ORDER from features_order.json
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_ORDER = [
    "Customer_Age", "Months_on_book", "Total_Relationship_Count",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Total_Revolving_Bal",
    "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "engagement_score",
    "Education_Level_Doctorate", "Education_Level_Graduate",
    "Education_Level_High School", "Education_Level_Post-Graduate",
    "Education_Level_Uneducated", "Education_Level_Unknown",
    "Income_Category_$40K - $60K", "Income_Category_$60K - $80K",
    "Income_Category_$80K - $120K", "Income_Category_Less than $40K",
    "Income_Category_Unknown", "Gender_M",
    "Marital_Status_Married", "Marital_Status_Single", "Marital_Status_Unknown",
    "Card_Category_Gold", "Card_Category_Platinum", "Card_Category_Silver",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRE-TRAINED MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open("stacking_model.pkl", "rb") as f:
        return joblib.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ `stacking_model.pkl` not found. Place it in the same folder as `app.py`.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER — mirrors notebook get_dummies(drop_first=True) exactly
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(inp: dict) -> pd.DataFrame:
    # Education — drop_first drops 'College'
    edu_cats = ["Doctorate", "Graduate", "High School", "Post-Graduate", "Uneducated", "Unknown"]
    edu_vec  = {f"Education_Level_{v}": int(inp["Education_Level"] == v) for v in edu_cats}

    # Income — drop_first drops '$120K +'
    inc_cats = ["$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K", "Unknown"]
    inc_vec  = {f"Income_Category_{v}": int(inp["Income_Category"] == v) for v in inc_cats}

    # Gender — drop_first drops 'F'
    gender_M = int(inp["Gender"] == "M")

    # Marital — drop_first drops 'Divorced'
    mar_cats = ["Married", "Single", "Unknown"]
    mar_vec  = {f"Marital_Status_{v}": int(inp["Marital_Status"] == v) for v in mar_cats}

    # Card — drop_first drops 'Blue'
    card_cats = ["Gold", "Platinum", "Silver"]
    card_vec  = {f"Card_Category_{v}": int(inp["Card_Category"] == v) for v in card_cats}

    # Derived feature (same as notebook)
    engagement_score = inp["Total_Trans_Ct"] * inp["Avg_Utilization_Ratio"]

    row = {
        "Customer_Age":             inp["Customer_Age"],
        "Months_on_book":           inp["Months_on_book"],
        "Total_Relationship_Count": inp["Total_Relationship_Count"],
        "Months_Inactive_12_mon":   inp["Months_Inactive_12_mon"],
        "Contacts_Count_12_mon":    inp["Contacts_Count_12_mon"],
        "Total_Revolving_Bal":      inp["Total_Revolving_Bal"],
        "Total_Amt_Chng_Q4_Q1":     inp["Total_Amt_Chng_Q4_Q1"],
        "Total_Trans_Amt":          inp["Total_Trans_Amt"],
        "Total_Trans_Ct":           inp["Total_Trans_Ct"],
        "Total_Ct_Chng_Q4_Q1":      inp["Total_Ct_Chng_Q4_Q1"],
        "Avg_Utilization_Ratio":    inp["Avg_Utilization_Ratio"],
        "engagement_score":         engagement_score,
        **edu_vec, **inc_vec,
        "Gender_M":                 gender_M,
        **mar_vec, **card_vec,
    }

    return pd.DataFrame([row])[FEATURES_ORDER]   # ← strict column order enforced here

# ─────────────────────────────────────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_gauge(prob: float, verdict: str) -> go.Figure:
    color = "#f43f5e" if verdict == "ATTRITED" else "#22d3a5"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 38, "color": color, "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1e2d4a",
                     "tickfont": {"color": "#64748b", "size": 10, "family": "IBM Plex Mono"}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "#0f1420", "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(34,211,165,0.06)"},
                {"range": [40, 70], "color": "rgba(212,168,75,0.06)"},
                {"range": [70,100], "color": "rgba(244,63,94,0.06)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8,
                          "value": round(prob*100, 1)},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=230, margin=dict(t=20, b=10, l=20, r=20), font_family="IBM Plex Mono",
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🛡️</div>
  <div>
    <div class="hero-title">ChurnShield</div>
    <div class="hero-sub">Bank Customer Attrition Predictor · Stacking Ensemble</div>
  </div>
  <div class="hero-badge">
    <span class="hero-badge-label">Model</span>
    <span class="hero-badge-val">stacking_model.pkl ✅</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 1], gap="large")

# ════════════════════════════════════════════
# LEFT — Numeric inputs
# ════════════════════════════════════════════
with left:

    st.markdown('<span class="section-label">👤 Customer Profile</span>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=45)
    with c2:
        months_on_book = st.number_input("Months on Book", min_value=1, max_value=60, value=36)

    c3, c4 = st.columns(2)
    with c3:
        total_relationship_count = st.slider("Total Products Held", 1, 6, 3)
    with c4:
        months_inactive = st.slider("Months Inactive (last 12)", 0, 6, 2)

    contacts_count = st.slider("Contacts Count (last 12 months)", 0, 6, 2)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<span class="section-label">💳 Transaction & Balance</span>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    total_revolving_bal = st.number_input(
        "Total Revolving Balance ($)", min_value=0, max_value=3000, value=800, step=50
    )

    c5, c6 = st.columns(2)
    with c5:
        total_trans_amt = st.number_input(
            "Total Transaction Amount ($)", min_value=0, max_value=20000, value=4500, step=100
        )
    with c6:
        total_trans_ct = st.number_input("Total Transaction Count", min_value=0, max_value=150, value=60)

    c7, c8 = st.columns(2)
    with c7:
        total_amt_chng = st.number_input(
            "Amt Change Q4→Q1 Ratio", min_value=0.0, max_value=3.5,
            value=0.75, step=0.01, format="%.3f"
        )
    with c8:
        total_ct_chng = st.number_input(
            "Count Change Q4→Q1 Ratio", min_value=0.0, max_value=3.5,
            value=0.70, step=0.01, format="%.3f"
        )

    avg_utilization = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.25, step=0.01)

    eng_score = round(total_trans_ct * avg_utilization, 4)
    st.markdown(f"""
    <div class="derived-chip">
      ⚡ <strong>engagement_score</strong>&nbsp;=&nbsp;Trans_Ct × Util_Ratio&nbsp;=&nbsp;
      <strong style="color:#e2e8f0">{eng_score}</strong>
      &nbsp;<span style="color:#64748b;font-size:0.68rem">(auto-calculated)</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# RIGHT — Categoricals + Result
# ════════════════════════════════════════════
with right:

    st.markdown('<span class="section-label">📋 Demographic Details</span>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["F", "M"], format_func=lambda x: "Female" if x == "F" else "Male")

    education_level = st.selectbox(
        "Education Level",
        ["College", "Doctorate", "Graduate", "High School", "Post-Graduate", "Uneducated", "Unknown"],
        index=2,
    )
    st.markdown('<div class="info-note">⚑ Reference (baseline): College → all dummies = 0</div>',
                unsafe_allow_html=True)

    marital_status = st.selectbox(
        "Marital Status", ["Divorced", "Married", "Single", "Unknown"], index=1
    )
    st.markdown('<div class="info-note">⚑ Reference (baseline): Divorced → all dummies = 0</div>',
                unsafe_allow_html=True)

    income_category = st.selectbox(
        "Income Category",
        ["$120K +", "$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K", "Unknown"],
        index=2,
    )
    st.markdown('<div class="info-note">⚑ Reference (baseline): $120K + → all dummies = 0</div>',
                unsafe_allow_html=True)

    card_category = st.selectbox("Card Category", ["Blue", "Gold", "Platinum", "Silver"], index=0)
    st.markdown('<div class="info-note">⚑ Reference (baseline): Blue → all dummies = 0</div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── PREDICT BUTTON ──────────────────────────────────────────────────────
    st.markdown('<span class="section-label">🔮 Prediction</span>', unsafe_allow_html=True)
    predict_clicked = st.button("⚡  Predict Churn")

    if predict_clicked:
        inp = {
            "Customer_Age":             customer_age,
            "Months_on_book":           months_on_book,
            "Total_Relationship_Count": total_relationship_count,
            "Months_Inactive_12_mon":   months_inactive,
            "Contacts_Count_12_mon":    contacts_count,
            "Total_Revolving_Bal":      total_revolving_bal,
            "Total_Amt_Chng_Q4_Q1":     total_amt_chng,
            "Total_Trans_Amt":          total_trans_amt,
            "Total_Trans_Ct":           total_trans_ct,
            "Total_Ct_Chng_Q4_Q1":      total_ct_chng,
            "Avg_Utilization_Ratio":    avg_utilization,
            "Education_Level":          education_level,
            "Income_Category":          income_category,
            "Gender":                   gender,
            "Marital_Status":           marital_status,
            "Card_Category":            card_category,
        }

        X_input    = build_feature_vector(inp)
        prob_churn = float(model.predict_proba(X_input)[0, 1])
        prediction = int(model.predict(X_input)[0])

        verdict       = "ATTRITED" if prediction == 1 else "RETAINED"
        verdict_class = "danger"   if prediction == 1 else "safe"
        emoji         = "🔴"       if prediction == 1 else "🟢"
        risk_label    = ("High Risk"   if prob_churn >= 0.70 else
                         "Medium Risk" if prob_churn >= 0.40 else "Low Risk")
        risk_color    = ("#f43f5e" if prob_churn >= 0.70 else
                         "#d4a84b" if prob_churn >= 0.40 else "#22d3a5")

        st.plotly_chart(make_gauge(prob_churn, verdict), use_container_width=True)

        st.markdown(f"""
        <div class="result-card {verdict_class}">
          <div style="font-family:'IBM Plex Mono';font-size:0.68rem;color:var(--muted);
                      text-transform:uppercase;letter-spacing:0.12em">Model Verdict</div>
          <div class="result-verdict {verdict_class}">{emoji} {verdict}</div>
          <div style="font-family:'IBM Plex Mono';font-size:0.75rem;color:var(--muted)">
            Churn probability:
            <strong style="color:{risk_color}">{prob_churn*100:.1f}%</strong> · {risk_label}
          </div>
          <div class="stat-row">
            <div class="stat-chip">Churn Prob<strong>{prob_churn*100:.2f}%</strong></div>
            <div class="stat-chip">Retain Prob<strong>{(1-prob_churn)*100:.2f}%</strong></div>
            <div class="stat-chip">Risk Tier<strong style="color:{risk_color}">{risk_label}</strong></div>
            <div class="stat-chip">Eng Score<strong>{eng_score:.3f}</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem 1rem;border-style:dashed">
          <div style="font-size:2.5rem;margin-bottom:0.8rem">🎯</div>
          <div style="font-family:'IBM Plex Mono';font-size:0.75rem;color:#64748b;
                      text-transform:uppercase;letter-spacing:0.1em">
            Fill in the details and click Predict
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VECTOR DEBUGGER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("🔬 Inspect Feature Vector sent to model (all 30 in order)"):
    debug_inp = {
        "Customer_Age":             customer_age,
        "Months_on_book":           months_on_book,
        "Total_Relationship_Count": total_relationship_count,
        "Months_Inactive_12_mon":   months_inactive,
        "Contacts_Count_12_mon":    contacts_count,
        "Total_Revolving_Bal":      total_revolving_bal,
        "Total_Amt_Chng_Q4_Q1":     total_amt_chng,
        "Total_Trans_Amt":          total_trans_amt,
        "Total_Trans_Ct":           total_trans_ct,
        "Total_Ct_Chng_Q4_Q1":      total_ct_chng,
        "Avg_Utilization_Ratio":    avg_utilization,
        "Education_Level":          education_level,
        "Income_Category":          income_category,
        "Gender":                   gender,
        "Marital_Status":           marital_status,
        "Card_Category":            card_category,
    }
    X_debug = build_feature_vector(debug_inp)
    st.dataframe(X_debug.T.rename(columns={0: "value"}), use_container_width=True, height=700)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-family:'IBM Plex Mono';
            font-size:0.62rem;color:#334155;letter-spacing:0.08em;text-transform:uppercase">
  ChurnShield · Loads stacking_model.pkl · 30 features · engagement_score = Trans_Ct × Util_Ratio
</div>
""", unsafe_allow_html=True)