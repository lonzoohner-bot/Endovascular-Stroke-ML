import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# 基础路径（关键！解决找不到文件问题）
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# 页面设置
# ==============================
st.set_page_config(
    page_title="90-Day Outcome Prediction After EVT for BAO",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 90-Day Outcome Prediction After EVT for BAO")
st.caption("Machine learning–based prognostic model for unfavorable functional outcome (mRS 4–6)")

# ==============================
# 加载模型与元数据
# ==============================
@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, "best_single_model_SVM_7f.pkl")
    features_path = os.path.join(BASE_DIR, "final_features_list.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    model = joblib.load(model_path)
    final_features = joblib.load(features_path)

    final_features = list(final_features)
    return model, final_features


try:
    model, final_features = load_assets()
except Exception as e:
    st.error(f"❌ Failed to load model assets: {e}")
    st.stop()

# ==============================
# 构造输入函数
# ==============================
def build_input_dataframe(
    age, sbp, asitnsir, baselinenihss,
    onset_admission, pcaspects, occlusion_site_label
):

    occlusion_map = {
        "Distal segment": 0,
        "Middle segment": 1,
        "Proximal segment": 2,
    }

    base_input = {
        "age": age,
        "sbp": sbp,
        "asitnsir": asitnsir,
        "baselinenihss": baselinenihss,
        "onset_admission": onset_admission,
        "pcaspects": pcaspects,
        "occlusion_site": occlusion_map[occlusion_site_label],
    }

    input_df = pd.DataFrame([{col: 0 for col in final_features}])

    for col, value in base_input.items():
        if col in input_df.columns:
            input_df.at[0, col] = value

    return input_df


def get_prediction(model, X):
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    idx_good = classes.index(0)
    idx_poor = classes.index(1)

    prob_good = float(proba[idx_good])
    prob_poor = float(proba[idx_poor])
    pred = int(model.predict(X)[0])

    return prob_poor, prob_good, pred


# ==============================
# 输入区域
# ==============================
st.sidebar.header("Input Parameters")

age = st.sidebar.slider("Age (years)", 18, 100, 65)
sbp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 90, 220, 140)
asitnsir = st.sidebar.slider("ASITN/SIR collateral score", 0, 4, 2)
baselinenihss = st.sidebar.slider("Baseline NIHSS", 0, 40, 13)
onset_admission = st.sidebar.slider("Onset-to-admission time (min)", 0, 1440, 300)
pcaspects = st.sidebar.slider("PC-ASPECTS", 0, 10, 8)

occlusion_site_label = st.sidebar.selectbox(
    "Occlusion site",
    ["Distal segment", "Middle segment", "Proximal segment"]
)

# ==============================
# 预测
# ==============================
user_input = build_input_dataframe(
    age, sbp, asitnsir, baselinenihss,
    onset_admission, pcaspects, occlusion_site_label
)

prob_poor, prob_good, pred = get_prediction(model, user_input)

# ==============================
# 展示结果
# ==============================
st.subheader("Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("Prob. of Poor Outcome (mRS 4–6)", f"{prob_poor:.2%}")

with col2:
    if pred == 1:
        st.error("⚠️ Poor Recovery")
    else:
        st.success("✅ Favorable Recovery")

st.caption(f"Prob. of favorable outcome: {prob_good:.2%}")

# ==============================
# 风险分层
# ==============================
st.subheader("Risk Stratification")

if prob_poor < 0.30:
    st.success("🟢 Low Risk")
elif prob_poor < 0.60:
    st.warning("🟡 Intermediate Risk")
else:
    st.error("🔴 High Risk")

st.progress(prob_poor)

# ==============================
# 图像展示（修复路径问题）
# ==============================
st.subheader("Model Explainability & Performance")

tab1, tab2 = st.tabs(["SHAP", "ROC"])

with tab1:
    shap_path = os.path.join(BASE_DIR, "SHAP_summary_Top7.png")
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Summary Plot")
    else:
        st.info("SHAP plot not found")

with tab2:
    roc_path = os.path.join(BASE_DIR, "ROC_external_SVM_7f.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC Curve")
    else:
        st.info("ROC plot not found")

# ==============================
# 页脚
# ==============================
st.caption("Disclaimer: For research use only.")
