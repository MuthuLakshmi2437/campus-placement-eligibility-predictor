"""
Campus Placement Eligibility Predictor
---------------------------------------
A single-page Streamlit app that predicts whether a student is eligible
for campus placement, using the RandomForest pipeline trained in
train_model.py (same preprocessing as the original Colab notebook).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="Campus Placement Eligibility Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# ==============================================================
# CUSTOM CSS — professional navy / steel-blue theme
# ==============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root{
    --bg1:#F8FBFD;
    --bg2:#EAF2FA;
    --navy:#2E4F6E;
    --navy-dark:#1D3A54;
    --blue:#4A8FC9;
    --blue-strong:#2E6FA8;
    --text-main:#1E2E3F;
    --text-soft:#55697D;
    --card-bg:#FFFFFF;
    --card-border:#DDE7F2;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background — calm, professional, no visual noise */
.stApp {
    background: linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 100%);
}

/* A little breathing room at the top of the page */
div.block-container {
    padding-top: 2.4rem;
    padding-bottom: 2rem;
}

/* ---- Streamlit's built-in top bar (title/Deploy strip) — blend it with the
   app theme instead of a stray black bar sitting on top of the page ---- */
header[data-testid="stHeader"] {
    background: linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 100%) !important;
    box-shadow: none !important;
    border-bottom: 1px solid var(--card-border);
    position: relative;
}
/* App title placed directly inside the top bar, so the title and the
   Deploy button sit together on one continuous strip instead of feeling
   like two disconnected pieces */
header[data-testid="stHeader"]::before {
    content: "🎓  Campus Placement Eligibility Predictor";
    position: absolute;
    left: 1.3rem;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--navy);
    white-space: nowrap;
    pointer-events: none;
}
/* On narrow screens, hide the injected header title so it never
   overlaps the Deploy / menu buttons — the hero banner below already
   shows the full title clearly on mobile widths */
@media (max-width: 640px) {
    header[data-testid="stHeader"]::before {
        display: none;
    }
}

[data-testid="stDecoration"] {
    background: linear-gradient(90deg, var(--navy-dark), var(--blue), var(--blue-strong)) !important;
    height: 3px;
}
[data-testid="stToolbar"],
[data-testid="stToolbarActions"] {
    background: transparent !important;
}
/* Menu / settings icon (the "..." kebab) — dark so it shows on the light bar */
header[data-testid="stHeader"] svg,
header[data-testid="stHeader"] path {
    fill: var(--navy) !important;
    color: var(--navy) !important;
}
/* "Deploy" button — solid, clearly readable pill instead of the plain
   default black button that looked cut off from the rest of the page */
[data-testid="stAppDeployButton"] button,
header[data-testid="stHeader"] button[kind="header"],
header[data-testid="stHeader"] button {
    background: var(--navy) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    padding: 0.35rem 0.9rem !important;
}
[data-testid="stAppDeployButton"] button *,
header[data-testid="stHeader"] button[kind="header"] *,
header[data-testid="stHeader"] button * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    opacity: 1 !important;
}
[data-testid="stAppDeployButton"] button:hover,
header[data-testid="stHeader"] button:hover {
    background: var(--blue-strong) !important;
}

/* Spacing between stacked elements */
[data-testid="stVerticalBlock"] { gap: 0.85rem; }

/* ---- Base readable text color everywhere (declared first) ---- */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li {
    color: var(--text-main);
}

/* Headings — bold, dark navy, no underline, no badge */
h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif;
    color: var(--navy) !important;
    font-weight: 700;
    border-bottom: none !important;
}

/* Section labels inside each card (plain bold text, dark, no pill/underline) */
.section-label {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.02rem;
    color: var(--blue-strong) !important;
    margin-bottom: 0.9rem;
}

/* ---- Sidebar — same navy family as the header, for a consistent look ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy-dark) 0%, var(--navy) 100%);
}
section[data-testid="stSidebar"] * {
    color: #E4ECF7 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Hero banner — matches sidebar tone for visual consistency */
.hero-banner {
    background: linear-gradient(115deg, var(--navy-dark) 0%, var(--blue-strong) 55%, var(--blue) 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    box-shadow: 0 10px 24px rgba(46, 79, 110, 0.25);
    margin-bottom: 1.6rem;
    overflow: visible;
}
.hero-banner h1, .hero-banner p, .hero-banner span {
    color: #FFFFFF !important;
}
.hero-banner h1 {
    font-size: clamp(1.3rem, 2.6vw, 1.9rem);
    margin-bottom: 0.3rem;
    white-space: normal;
    overflow-wrap: break-word;
    line-height: 1.3;
}
.hero-banner p {
    color: #DCE9FA !important;
    font-size: 0.98rem;
    margin: 0;
    white-space: normal;
    overflow-wrap: break-word;
}

/* Card style container — spacious padding, clean border, no stray bars */
.pink-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-top: 3px solid var(--blue);
    border-radius: 12px;
    padding: 1.6rem 1.7rem 1.8rem 1.7rem;
    box-shadow: 0 4px 14px rgba(46, 79, 110, 0.08);
    margin-bottom: 1rem;
}
.pink-card, .pink-card p, .pink-card span, .pink-card li, .pink-card b {
    color: var(--text-main) !important;
}
.pink-card b { color: var(--blue-strong) !important; }

/* Metric cards */
.metric-box {
    background: linear-gradient(145deg, #F0F5FB, #DCE9F8);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
    box-shadow: 0 3px 10px rgba(46, 79, 110, 0.10);
}
.metric-box .value {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--navy) !important;
    font-family: 'Poppins', sans-serif;
}
.metric-box .label {
    font-size: 0.8rem;
    color: var(--blue-strong) !important;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* Result banners */
.result-eligible {
    background: linear-gradient(120deg, #145C36, #1F8449);
    padding: 1.25rem 1.5rem;
    border-radius: 14px;
    font-size: 1.25rem;
    font-weight: 800;
    text-align: center;
    box-shadow: 0 8px 20px rgba(20, 92, 54, 0.28);
    font-family: 'Poppins', sans-serif;
}
.result-eligible, .result-eligible span { color: #FFFFFF !important; }

.result-not-eligible {
    background: linear-gradient(120deg, #7A1E22, #A6302F);
    padding: 1.25rem 1.5rem;
    border-radius: 14px;
    font-size: 1.25rem;
    font-weight: 800;
    text-align: center;
    box-shadow: 0 8px 20px rgba(122, 30, 34, 0.28);
    font-family: 'Poppins', sans-serif;
}
.result-not-eligible, .result-not-eligible span { color: #FFFFFF !important; }

/* Buttons — clearly clickable: pointer cursor, hover lift, and a
   press-down effect on click so it always feels interactive */
.stButton>button, .stFormSubmitButton>button {
    background: linear-gradient(120deg, var(--navy-dark), var(--blue)) !important;
    color: #FFFFFF !important;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 0.7rem 1.8rem;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
    letter-spacing: 0.01em;
    box-shadow: 0 5px 12px rgba(46, 79, 110, 0.30);
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    cursor: pointer !important;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(46, 79, 110, 0.42);
    filter: brightness(1.08);
    border-color: rgba(255,255,255,0.35);
}
.stButton>button:active, .stFormSubmitButton>button:active {
    transform: translateY(0px) scale(0.98);
    box-shadow: 0 3px 8px rgba(46, 79, 110, 0.35);
    filter: brightness(0.96);
}
.stButton>button:focus-visible, .stFormSubmitButton>button:focus-visible {
    outline: 2px solid var(--blue);
    outline-offset: 2px;
}
.stButton>button *, .stFormSubmitButton>button * { color: #FFFFFF !important; }

/* Sliders */
[data-testid="stSlider"] [role="slider"] { background-color: var(--blue) !important; }
div[data-baseweb="slider"] > div > div { background: var(--blue) !important; }

/* Expander (Project Summary) — darker, bolder title for a more
   professional, report-like heading */
[data-testid="stExpander"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-top: 3px solid var(--navy-dark);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--navy-dark) !important;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
    font-size: 1.05rem;
    letter-spacing: 0.01em;
    padding: 0.9rem 1.1rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--blue-strong) !important;
}

/* Divider */
hr { border-color: var(--card-border) !important; }

/* Footer note */
.footer-note {
    text-align: center;
    color: var(--text-soft);
    font-size: 0.83rem;
    margin-top: 1.8rem;
    opacity: 0.85;
}
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================
# LOAD ARTIFACTS
# ==============================================================
@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "campus_placement_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    metrics = None
    metrics_path = BASE_DIR / "model_metrics.pkl"
    if metrics_path.exists():
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
    return model, encoders, scaler, feature_columns, metrics


try:
    model, encoders, scaler, feature_columns, metrics = load_artifacts()
    ARTIFACTS_OK = True
except FileNotFoundError:
    ARTIFACTS_OK = False


BRANCH_OPTIONS = (
    list(encoders["branch"].classes_) if ARTIFACTS_OK and "branch" in encoders else []
)

REQUIRED_RAW_COLUMNS = [
    "cgpa", "branch", "college_tier", "python_skill", "dsa_skill",
    "ml_skill", "web_dev_skill", "coding_score", "communication_score",
    "aptitude_score", "internships", "projects", "backlogs", "resume_score",
    "skill_score",
]


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================
def preprocess_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning / encoding used during training."""
    df = df_raw.copy()

    drop_cols = ["student_id", "company_type", "job_role",
                 "salary_lpa", "placed", "placement_status"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}")

    df = df[REQUIRED_RAW_COLUMNS].copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    le = encoders["branch"]
    known_classes = set(le.classes_)

    def safe_encode(val):
        val = str(val)
        if val in known_classes:
            return le.transform([val])[0]
        return le.transform([le.classes_[0]])[0]

    df["branch"] = df["branch"].apply(safe_encode)

    return df[feature_columns]


def predict_one(df_processed: pd.DataFrame):
    X_scaled = scaler.transform(df_processed)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[:, 1][0]
    return pred, prob


def metric_box(label, value):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## 🎓 About")
    st.markdown(
        """
        This app predicts **campus placement eligibility** using a
        Random Forest model trained on student academic, skill and
        performance data.
        """
    )
    st.markdown("---")
    st.markdown("### Inputs Considered")
    st.markdown(
        "- CGPA, branch, college tier\n"
        "- Python / DSA / ML / Web Dev skills\n"
        "- Coding, communication & aptitude scores\n"
        "- Internships, projects, backlogs\n"
        "- Resume score & skill score"
    )
    if ARTIFACTS_OK and metrics:
        st.markdown("---")
        st.markdown("### Model Quality")
        st.markdown(f"**Accuracy:** {metrics['accuracy']*100:.1f}%")
        st.markdown(f"**F1 Score:** {metrics['f1']:.3f}")
        st.markdown(f"**ROC-AUC:** {metrics['roc_auc']:.3f}")


# ==============================================================
# HERO
# ==============================================================
st.markdown(
    """
    <div class="hero-banner">
        <h1>🎓 Campus Placement Eligibility Predictor</h1>
        <p>Enter a student's profile below to instantly predict placement eligibility.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ARTIFACTS_OK:
    st.error(
        "⚠️ Model artifacts not found. Please run **train_model.py** first "
        "(it needs `dataset.csv` in the same folder) to generate "
        "`campus_placement_model.pkl`, `label_encoders.pkl`, `scaler.pkl` "
        "and `feature_columns.pkl`."
    )
    st.stop()


# ==============================================================
# SINGLE PREDICTION FORM
# ==============================================================
st.markdown("### Enter Student Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="pink-card">', unsafe_allow_html=True)
        section_label("🎓 Academics")
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.0, 0.01)
        branch = st.selectbox("Branch", BRANCH_OPTIONS)
        college_tier = st.selectbox(
            "College Tier", [1, 2, 3],
            format_func=lambda x: f"Tier {x}" + (" (Top)" if x == 1 else ""),
        )
        backlogs = st.slider("Backlogs", 0, 3, 0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="pink-card">', unsafe_allow_html=True)
        section_label("💻 Skills")
        python_skill = st.selectbox("Python Skill", [0, 1], format_func=lambda x: "Yes" if x else "No")
        dsa_skill = st.selectbox("DSA Skill", [0, 1], format_func=lambda x: "Yes" if x else "No")
        ml_skill = st.selectbox("ML Skill", [0, 1], format_func=lambda x: "Yes" if x else "No")
        web_dev_skill = st.selectbox("Web Dev Skill", [0, 1], format_func=lambda x: "Yes" if x else "No")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="pink-card">', unsafe_allow_html=True)
        section_label("📈 Performance")
        coding_score = st.slider("Coding Score", 0.0, 100.0, 50.0, 0.1)
        communication_score = st.slider("Communication Score", 4.0, 10.0, 7.0, 0.1)
        aptitude_score = st.slider("Aptitude Score", 40.0, 100.0, 70.0, 0.1)
        internships = st.slider("Internships", 0, 3, 1)
        projects = st.slider("Projects", 1, 6, 2)
        resume_score = st.slider("Resume Score", 16.5, 132.1, 60.0, 0.1)
        skill_score = st.slider("Skill Score", 0, 4, 2)
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🔮 Predict Eligibility", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "cgpa": cgpa,
        "branch": branch,
        "college_tier": college_tier,
        "python_skill": python_skill,
        "dsa_skill": dsa_skill,
        "ml_skill": ml_skill,
        "web_dev_skill": web_dev_skill,
        "coding_score": coding_score,
        "communication_score": communication_score,
        "aptitude_score": aptitude_score,
        "internships": internships,
        "projects": projects,
        "backlogs": backlogs,
        "resume_score": resume_score,
        "skill_score": skill_score,
    }])

    processed = preprocess_dataframe(input_df)
    pred, prob = predict_one(processed)

    st.markdown("---")
    res_col, chart_col = st.columns([1, 1], gap="large")

    with res_col:
        if pred == 1:
            st.markdown(
                f'<div class="result-eligible">✅ ELIGIBLE for Placement<br>'
                f'<span style="font-size:1rem; font-weight:600;">Confidence: {prob*100:.2f}%</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-not-eligible">❌ NOT ELIGIBLE for Placement<br>'
                f'<span style="font-size:1rem; font-weight:600;">Confidence: {(1-prob)*100:.2f}%</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            metric_box("Eligible Probability", f"{prob*100:.1f}%")
        with m2:
            metric_box("Not Eligible Probability", f"{(1-prob)*100:.1f}%")

    with chart_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#2E4F6E"}},
            title={"text": "Eligibility Score", "font": {"size": 16, "color": "#2E4F6E"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#2E4F6E"},
                "bar": {"color": "#4A8FC9"},
                "bgcolor": "#F0F5FB",
                "borderwidth": 1,
                "bordercolor": "#DDE7F2",
                "steps": [
                    {"range": [0, 40], "color": "#DCE9F8"},
                    {"range": [40, 70], "color": "#A9CBE8"},
                    {"range": [70, 100], "color": "#5C93CB"},
                ],
                "threshold": {
                    "line": {"color": "#2E4F6E", "width": 4},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(t=35, b=10, l=20, r=20),
            font=dict(color="#2E4F6E"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 What's Driving This Prediction?")
    if metrics and "feature_importance" in metrics:
        fi = metrics["feature_importance"]
        top_feats = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        values = input_df.iloc[0]
        tips = []
        for feat, imp in top_feats:
            tips.append(
                f"<b>{feat.replace('_', ' ').title()}</b> "
                f"(model weight: {imp*100:.1f}%) — this student's value: <b>{values[feat]}</b>"
            )
        st.markdown(
            "<div class='pink-card'>" + "<br><br>".join(tips) + "</div>",
            unsafe_allow_html=True,
        )


# ==============================================================
# PROJECT SUMMARY (optional, for reports / viva)
# ==============================================================
with st.expander("📄 Project Summary"):
    st.markdown('<div class="pink-card">', unsafe_allow_html=True)
    st.markdown(
        """
**Project:** Campus Placement Eligibility Predictor

**Objective:** Predict whether a student is eligible for campus
placement based on academic performance, technical skills, and
soft-skill / resume indicators.

**Dataset:** Student records with CGPA, branch, college tier,
skill flags (Python, DSA, ML, Web Dev), coding/communication/aptitude
scores, internships, projects, backlogs, and resume score.

**Pipeline:**
1. Data cleaning — removed duplicates and irrelevant columns
   (student ID, salary, company type, job role, skill score).
2. Encoding — `branch` label-encoded; other features already numeric.
3. Train/test split (80/20, stratified) followed by SMOTE
   oversampling of the minority class on the training set only.
4. Feature scaling with `StandardScaler`.
5. Model — `RandomForestClassifier` (200 trees, max depth 10,
   class-balanced).
        """
    )
    if metrics:
        st.markdown("**Model Performance (on held-out test set):**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_box("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        with c2:
            metric_box("Precision", f"{metrics['precision']*100:.1f}%")
        with c3:
            metric_box("Recall", f"{metrics['recall']*100:.1f}%")
        with c4:
            metric_box("F1 Score", f"{metrics['f1']*100:.1f}%")
        with c5:
            metric_box("ROC-AUC", f"{metrics['roc_auc']*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================
# FOOTER
# ==============================================================
st.markdown(
    '<div class="footer-note">Campus Placement Eligibility Predictor · Random Forest Classifier</div>',
    unsafe_allow_html=True,
)
