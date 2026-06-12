"""
AegisIQ - Intelligent Risk Assessment Platform
================================================
Multi-page Streamlit app leveraging native Streamlit components and Plotly.
"""

import os
import json
from pdf_generator import generate_risk_report
import base64
import pickle
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
try:
    BACKEND_URL = st.secrets.get("BACKEND_URL", os.environ.get("BACKEND_URL", "http://localhost:8000"))
except Exception:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
PREDICT_URL = f"{BACKEND_URL}/predict"
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "dataset" / "insurance_data_500.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

OCCUPATIONS = [
    "private_sector", "public_sector", "self_employed", "freelancer",
    "student", "retired", "unemployed", "business_owner",
]

OCCUPATION_LABELS = {
    "private_sector": "🏢  Private Sector",
    "public_sector": "🏛️  Public Sector",
    "self_employed": "💼  Self Employed",
    "freelancer": "🎨  Freelancer",
    "student": "🎓  Student",
    "retired": "🏖️  Retired",
    "unemployed": "📋  Unemployed",
    "business_owner": "🏭  Business Owner",
}

THEME = {
    "Low":    {"color": "#6CAEF9", "emoji": "✅", "label": "LOW RISK"},
    "Medium": {"color": "#FFAB00", "emoji": "⚠️", "label": "MEDIUM RISK"},
    "High":   {"color": "#C85A64", "emoji": "🚨", "label": "HIGH RISK"},
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AegisIQ - Intelligent Risk Assessment",
    page_icon="💠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Load Assets ─────────────────────────────────────────────────────────────
@st.cache_data
def load_css_asset(filename):
    with open(BASE_DIR / "assets" / filename, "r", encoding="utf-8") as f:
        return f.read()

CSS_CONTENT = load_css_asset("style.css")
SHIELD_SVG = load_css_asset("shield.svg")
_shield_b64 = base64.b64encode(SHIELD_SVG.strip().encode()).decode()

st.markdown(f"<style>{CSS_CONTENT}</style>", unsafe_allow_html=True)

# ── Cached loaders ──────────────────────────────────────────────────────────
@st.cache_data
def load_csv():
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ── Helper functions ─────────────────────────────────────────────────────────
def render_hero():
    st.markdown(f"""
    <div class="hero">
        <h1>AegisIQ <span style="font-size: 1.3em; vertical-align: middle;">💠</span></h1>
        <p style="color:#8888aa;font-size:1.1rem;margin-top:-0.5rem;">Intelligent Risk Assessment &middot; Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

def plot_confidence_gauge(confidence, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        number = {"suffix": "%", "font": {"color": color, "size": 48, "family": "JetBrains Mono"}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white", "visible": False},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "shape": "angular"
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#8888aa", "family": "Inter"}
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0 1rem;">
        <h2 style="margin:0.5rem 0 0 0;font-weight:900;">AegisIQ</h2>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["⚡ Predict", "🔬 How It Works", "📊 Model Performance", "🗂️ Data Explorer"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#888;font-size:0.8rem;'>Built with Streamlit & Plotly</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 - PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "⚡ Predict":
    render_hero()
    
    st.subheader("📝 Your Details")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", min_value=18, max_value=80, value=30)
            weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)
            height = st.number_input("Height (m)", min_value=1.40, max_value=2.20, value=1.70, step=0.01, format="%.2f")
            income_lpa = st.number_input("Annual Income (LPA)", min_value=1.0, max_value=100.0, value=8.0, step=0.5)
        with col2:
            smoker = st.checkbox("🚬 Smoker")
            city = st.text_input("🏙️ City", value="Mumbai")
            occ_display = st.selectbox("👔 Occupation", options=list(OCCUPATION_LABELS.values()))
            occ_reverse = {v: k for k, v in OCCUPATION_LABELS.items()}
            occupation = occ_reverse[occ_display]
        
        submitted = st.form_submit_button("⚡ Predict My Premium", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "age": age, "weight": weight, "height": height,
            "income_lpa": income_lpa, "smoker": smoker,
            "city": city.strip(), "occupation": occupation,
        }
        try:
            with st.spinner("Analyzing profile... (Note: First prediction may take up to 50s to wake up the cloud server)"):
                resp = requests.post(PREDICT_URL, json=payload, timeout=60)
            
            if resp.status_code == 200:
                data = resp.json()
                category = data["predicted_category"]
                confidence = data["confidence"]
                probs = data["class_probabilities"]
                t = THEME.get(category, THEME["Medium"])

                st.divider()
                st.subheader(f"🎯 Prediction Result")
                
                # Big result badge using native metric with custom color
                rc1, rc2, rc3 = st.columns([1, 2, 1])
                with rc2:
                    st.markdown(f"""
                    <div style="background-color: {t['color']}20; border: 2px solid {t['color']}; 
                                border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 2rem;">
                        <h2 style="color: {t['color']}; margin: 0;">{t['emoji']} {category.upper()} RISK</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Category", category)
                c2.metric("Confidence", f"{confidence*100:.1f}%")
                c3.metric("Calculated BMI", f"{round(weight / (height**2), 1)}")

                st.plotly_chart(plot_confidence_gauge(confidence, t['color']), use_container_width=True)

                st.divider()
                st.subheader("📊 Class Probabilities")
                
                chart_df = pd.DataFrame(
                    {"Probability (%)": [probs.get(c, 0)*100 for c in ["Low", "Medium", "High"]], "Category": ["Low", "Medium", "High"]}
                )
                
                # Plotly Horizontal Bar
                fig = px.bar(
                    chart_df, x="Probability (%)", y="Category", orientation='h',
                    color="Category", color_discrete_map={"Low": THEME["Low"]["color"], "Medium": THEME["Medium"]["color"], "High": THEME["High"]["color"]},
                    text="Probability (%)"
                )
                fig.update_traces(texttemplate='%{text:.1f}')
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        range=[0, 100], 
                        showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                        showline=True, linewidth=1, linecolor="rgba(255,255,255,0.2)", mirror=True
                    ),
                    yaxis=dict(
                        showgrid=False,
                        showline=True, linewidth=1, linecolor="rgba(255,255,255,0.2)", mirror=True
                    ),
                    showlegend=False,
                    height=200,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("📥 Export")
                
                pdf_bytes = generate_risk_report(payload, category, confidence * 100, THEME)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="AegisIQ_Risk_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            else:
                try: err_detail = resp.json().get("error", resp.text)
                except: err_detail = resp.text
                st.error(f"API Error ({resp.status_code}): {err_detail}")
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Failed - Cannot reach backend at `{BACKEND_URL}`. Is the FastAPI server running?")
        except requests.exceptions.ReadTimeout:
            st.error("The cloud backend took too long to wake up from sleep mode. It should be awake now—please click 'Predict' again!")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 - HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 How It Works":
    render_hero()
    
    st.header("🧬 Feature Engineering")
    st.markdown("How raw user inputs are transformed into ML-ready features before hitting the Random Forest.")

    # Native Markdown Table
    st.markdown("""
    | Raw Input | Derived Feature | Transformation Logic |
    | :--- | :--- | :--- |
    | ⚖️ weight + 📏 height | `bmi` | weight / height² (Body Mass Index) |
    | 🎂 age | `age_group` | `young` (<25) · `adult` (<45) · `middle_aged` (<60) · `senior` (60+) |
    | 🚬 smoker + BMI | `lifestyle_risk` | `high` (smoker & BMI>30) · `medium` (smoker & BMI>27) · `low` (else) |
    | 🏙️ city | `city_tier` | `Tier 1` (metros) · `Tier 2` (major cities) · `Tier 3` (others) |
    | 💰 income_lpa | `income_lpa` | Passed directly - annual income in lakhs per annum |
    | 👔 occupation | `occupation` | One-hot encoded - 8 categorical columns |
    """)

    st.divider()

    st.header("💡 Why These Features Matter")
    with st.expander("⚖️ BMI (Body Mass Index)", expanded=True):
        st.write("The strongest health indicator used by insurers. Higher BMI tightly correlates with diabetes, heart disease, and joint problems - dramatically increasing claim likelihood.")
    with st.expander("🎂 Age Group"):
        st.write("Insurance risk increases non-linearly with age. Bucketing into groups captures life-stage patterns: young people have fewer claims, while seniors have chronic conditions.")
    with st.expander("🚬 Lifestyle Risk"):
        st.write("Combines smoking status with BMI to create a compound risk factor. A smoker with high BMI faces exponentially higher health risks than either factor alone.")
    with st.expander("🏙️ City Tier"):
        st.write("Metro cities (Tier 1) have substantially higher medical costs and hospital bills. City tier acts as a direct proxy for healthcare cost and accessibility.")

    st.divider()

    st.header("🌲 The Model - Random Forest")
    st.write("An ensemble of **200 decision trees** that vote on the final prediction.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**What is an Ensemble?**\nInstead of relying on a single decision tree (which overfits), a Random Forest trains 200 independent trees, each on a random subset of the data.")
    with col2:
        st.info("**How Does Voting Work?**\nEach tree makes its own prediction. The final answer is determined by majority vote. The confidence score is the percentage of trees that voted for the winning class.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 - MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.write("Evaluation metrics from the final trained RandomForest model.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", "91.0%")
    c2.metric("F1 Weighted", "0.9086")
    c3.metric("CV Mean Accuracy", "83.6%", delta="± 2.06%", delta_color="off")

    tab1, tab2, tab3 = st.tabs(["Per-Class Breakdown", "Confusion Matrix", "Feature Importance"])

    with tab1:
        st.markdown("""
        | Class | Precision | Recall | F1-Score | Support |
        | :--- | :--- | :--- | :--- | :--- |
        | 🔴 **High** | 0.93 | 0.76 | 0.84 | 17 |
        | 🟢 **Low** | 0.90 | 0.90 | 0.90 | 21 |
        | 🟡 **Medium** | 0.91 | 0.95 | 0.93 | 62 |
        """)

    with tab2:
        st.write("Actual vs Predicted Categories")
        cm_df = pd.DataFrame(
            [[13, 0, 4], [0, 19, 2], [1, 2, 59]],
            index=["Actual: High", "Actual: Low", "Actual: Medium"],
            columns=["Pred: High", "Pred: Low", "Pred: Medium"]
        )
        st.dataframe(cm_df, use_container_width=True)

    with tab3:
        try:
            model = load_model()
            clf = model.named_steps["classifier"]
            preprocessor = model.named_steps["preprocessor"]

            cat_features = preprocessor.transformers_[0][2]
            ohe = preprocessor.transformers_[0][1]
            cat_names = ohe.get_feature_names_out(cat_features).tolist()
            num_names = preprocessor.transformers_[1][2]
            all_names = cat_names + list(num_names)

            fi_df = pd.DataFrame({"Importance": clf.feature_importances_}, index=all_names)
            fi_df = fi_df.sort_values("Importance", ascending=True)

            fig = px.bar(fi_df, x="Importance", y=fi_df.index, orientation='h', title="Feature Importances")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load feature importances: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 - DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Data Explorer":
    st.title("🗂️ Data Explorer")
    st.write("Explore the underlying insurance dataset used to train the model.")
    
    df = load_csv()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{len(df)}")
    c2.metric("Features", "8")
    c3.metric("Target Classes", "3")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Premium Category Distribution")
        fig_pie = px.pie(
            df, names='insurance_premium_category', hole=0.4, 
            color='insurance_premium_category',
            color_discrete_map={"Low": THEME["Low"]["color"], "Medium": THEME["Medium"]["color"], "High": THEME["High"]["color"]}
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Age Distribution")
        fig_hist = px.histogram(df, x="age", nbins=20, color_discrete_sequence=["#00E5FF"])
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Income by Premium Category")
        fig_box = px.box(
            df, x="insurance_premium_category", y="income_lpa", 
            color="insurance_premium_category",
            color_discrete_map={"Low": THEME["Low"]["color"], "Medium": THEME["Medium"]["color"], "High": THEME["High"]["color"]}
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        st.subheader("Smoker Proportion")
        fig_smoker = px.pie(df, names='smoker', color_discrete_sequence=["#FF1744", "#00E676"])
        fig_smoker.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_smoker, use_container_width=True)
