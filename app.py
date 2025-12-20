import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ==============================
# 页面设置
# ==============================
st.set_page_config(
    page_title="Stroke Outcome Prediction",
    layout="centered"
)

st.title("Stroke Outcome Prediction after EVT")
st.caption("Machine learning–based prognostic model for mRS assessment")

# ==============================
# 加载模型与元数据（建议使用缓存）
# ==============================
@st.cache_resource
def load_assets():
    model = joblib.load("best_model_Voting_infer.pkl")
    final_features = joblib.load("final_features_list.pkl")
    return model, final_features

model, final_features = load_assets()

# ==============================
# 输入区域
# ==============================
st.sidebar.header("Input Parameters")

age = st.sidebar.slider("Age (years)", 18, 100, 65)
sbp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 90, 180, 120)
asitnsir = st.sidebar.slider("ASITN/SIR Score", 0, 4, 2)
baselinenihss = st.sidebar.slider("Baseline NIHSS", 0, 30, 5)

occlusion_site = st.sidebar.selectbox(
    "Occlusion Site",
    options=[
        ("Middle segment", 1),
        ("Proximal segment", 2),
        ("Distal segment", 3),
    ],
    format_func=lambda x: x[0]
)[1]

onset_admission = st.sidebar.slider("Onset-to-admission time (min)", 0, 300, 60)
pcaspects = st.sidebar.slider("PC-ASPECTS", 0, 10, 7)

# ==============================
# 构造模型输入
# ==============================
user_input = pd.DataFrame({
    "age": [age],
    "sbp": [sbp],
    "asitnsir": [asitnsir],
    "baselinenihss": [baselinenihss],
    "occlusion_site": [occlusion_site],
    "onset_admission": [onset_admission],
    "pcaspects": [pcaspects],
})

# 确保特征对齐
user_input = user_input[final_features]

# ==============================
# 预测计算
# ==============================
prob = model.predict_proba(user_input)[0, 1]
pred = model.predict(user_input)[0]

# ==============================
# 结果展示
# ==============================
st.subheader("Prediction Result")

col1, col2 = st.columns([1, 1])

with col1:
    st.metric(
        label="Prob. of Good Outcome",
        value=f"{prob:.2%}"
    )

with col2:
    if pred == 1:
        st.success("Result: Good Recovery")
    else:
        st.error("Result: Poor Recovery")

# --- 核心：风险分层部分 ---
st.markdown("---")
st.subheader("Risk Stratification")

if prob >= 0.70:
    st.success("### 🟢 Low Risk of Poor Outcome")
    st.write("Predicted probability suggests a highly favorable functional prognosis (mRS 0–3).")
elif 0.40 <= prob < 0.70:
    st.warning("### 🟡 Intermediate Risk")
    st.write("The prognosis is uncertain. Clinical vigilance and individualized management are recommended.")
else:
    st.error("### 🔴 High Risk of Poor Outcome")
    st.write("Predicted probability indicates a high risk of poor functional recovery (mRS 4–6). Intensified monitoring may be warranted.")

# 可视化进度条，增强直观感
st.progress(prob)

# ==============================
# 模型解释 & 性能
# ==============================
st.markdown("---")
st.subheader("Model Explainability & Performance")

tab1, tab2 = st.tabs(["SHAP (Feature Importance)", "Model Performance (ROC)"])

with tab1:
    if os.path.exists("SHAP_summary_Top7.png"):
        st.image("SHAP_summary_Top7.png", caption="SHAP Summary Plot")
    else:
        st.info("SHAP plot not found.")

with tab2:
    if os.path.exists("ROC_best_model.png"):
        st.image("ROC_best_model.png", caption="ROC Curve")
    else:
        st.info("ROC plot not found.")
