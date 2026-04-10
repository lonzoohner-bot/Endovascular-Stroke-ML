import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


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
    model_path = "best_single_model_SVM_7f.pkl"
    features_path = "final_features_list.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    model = joblib.load(model_path)
    final_features = joblib.load(features_path)

    if not isinstance(final_features, (list, tuple, np.ndarray, pd.Series)):
        raise ValueError("final_features_list.pkl must contain a sequence of feature names.")

    final_features = list(final_features)
    return model, final_features


try:
    model, final_features = load_assets()
except Exception as e:
    st.error(f"❌ Failed to load model assets: {e}")
    st.stop()


# ==============================
# 工具函数
# ==============================
def build_input_dataframe(
    age: int,
    sbp: int,
    asitnsir: int,
    baselinenihss: int,
    onset_admission: int,
    pcaspects: int,
    occlusion_site_label: str,
    final_features: list[str]
) -> pd.DataFrame:
    """
    根据训练时特征名自动构造输入：
    1) 若 final_features 中包含原始 'occlusion_site'，则直接填数值编码
    2) 若 final_features 中包含 one-hot 列，如 occlusion_site_distal，则自动填充
    """

    # 这里的编码请与你训练数据保持一致
    # 你现在图表和文稿中使用的是 distal / middle / proximal
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

    # 先建一个全0框架
    input_df = pd.DataFrame([{col: 0 for col in final_features}])

    # 填普通数值列
    for col, value in base_input.items():
        if col in input_df.columns:
            input_df.at[0, col] = value

    # 自动处理 one-hot 编码情况
    occlusion_onehot_candidates = [c for c in final_features if c.startswith("occlusion_site_")]
    if occlusion_onehot_candidates:
        # 与训练时命名保持兼容
        onehot_name_map = {
            "Distal segment": ["occlusion_site_distal", "occlusion_site_0", "occlusion_site_Distal segment"],
            "Middle segment": ["occlusion_site_middle", "occlusion_site_1", "occlusion_site_Middle segment"],
            "Proximal segment": ["occlusion_site_proximal", "occlusion_site_2", "occlusion_site_Proximal segment"],
        }
        for candidate in onehot_name_map[occlusion_site_label]:
            if candidate in input_df.columns:
                input_df.at[0, candidate] = 1

    return input_df


def get_probabilities_and_prediction(model, X: pd.DataFrame):
    """
    返回：
    - prob_poor: 不良结局概率（假设标签 1 = poor outcome）
    - prob_good: 良好结局概率（标签 0 = good outcome）
    - pred: 预测类别（0=good, 1=poor）
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba().")

    proba = model.predict_proba(X)[0]

    classes = list(getattr(model, "classes_", [0, 1]))
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Unexpected model classes: {classes}. Expected binary classes [0, 1].")

    idx_good = classes.index(0)   # 0 = favorable outcome
    idx_poor = classes.index(1)   # 1 = unfavorable outcome

    prob_good = float(proba[idx_good])
    prob_poor = float(proba[idx_poor])

    pred = int(model.predict(X)[0])
    return prob_poor, prob_good, pred


# ==============================
# 输入区域
# ==============================
st.sidebar.header("Input Parameters")

age = st.sidebar.slider("Age (years)", min_value=18, max_value=100, value=65)
sbp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", min_value=90, max_value=220, value=140)
asitnsir = st.sidebar.slider("ASITN/SIR collateral score", min_value=0, max_value=4, value=2)
baselinenihss = st.sidebar.slider("Baseline NIHSS", min_value=0, max_value=40, value=13)
onset_admission = st.sidebar.slider("Onset-to-admission time (min)", min_value=0, max_value=1440, value=300)
pcaspects = st.sidebar.slider("PC-ASPECTS", min_value=0, max_value=10, value=8)

occlusion_site_label = st.sidebar.selectbox(
    "Occlusion site",
    options=["Distal segment", "Middle segment", "Proximal segment"],
    index=1
)


# ==============================
# 构造输入并预测
# ==============================
try:
    user_input = build_input_dataframe(
        age=age,
        sbp=sbp,
        asitnsir=asitnsir,
        baselinenihss=baselinenihss,
        onset_admission=onset_admission,
        pcaspects=pcaspects,
        occlusion_site_label=occlusion_site_label,
        final_features=final_features
    )

    prob_poor, prob_good, pred = get_probabilities_and_prediction(model, user_input)

except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()


# ==============================
# 患者信息展示
# ==============================
st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Age", f"{age} years")
    st.metric("SBP", f"{sbp} mmHg")

with col2:
    st.metric("Baseline NIHSS", baselinenihss)
    st.metric("ASITN/SIR", asitnsir)

with col3:
    st.metric("Occlusion site", occlusion_site_label)
    st.metric("PC-ASPECTS", pcaspects)

st.metric("Onset-to-admission time", f"{onset_admission} min")

st.divider()


# ==============================
# 预测结果
# ==============================
st.subheader("Prediction Result")

col1, col2 = st.columns([1, 1])

with col1:
    st.metric(
        label="Prob. of Poor Outcome (mRS 4–6)",
        value=f"{prob_poor:.2%}"
    )

with col2:
    if pred == 1:
        st.error("⚠️ Result: Poor Recovery (mRS 4–6)")
    else:
        st.success("✅ Result: Favorable Recovery (mRS 0–3)")

st.caption(f"Predicted probability of favorable outcome (mRS 0–3): {prob_good:.2%}")

st.divider()


# ==============================
# 风险分层（仅界面展示）
# ==============================
st.subheader("Risk Stratification")

if prob_poor < 0.30:
    st.success("### 🟢 Low Risk of Poor Outcome")
    st.write("Predicted probability suggests a relatively favorable functional prognosis.")
elif prob_poor < 0.60:
    st.warning("### 🟡 Intermediate Risk of Poor Outcome")
    st.write("The prognosis is uncertain and should be interpreted together with clinical assessment.")
else:
    st.error("### 🔴 High Risk of Poor Outcome")
    st.write("Predicted probability indicates a relatively high risk of poor functional recovery (mRS 4–6).")

st.progress(prob_poor)
st.caption(f"Estimated risk of poor outcome (mRS 4–6): {prob_poor:.2%}")
st.caption("Risk categories are displayed for interface interpretation only and should not be considered definitive clinical thresholds.")

st.divider()


# ==============================
# 模型解释 & 性能
# ==============================
st.subheader("Model Explainability & Performance")

tab1, tab2 = st.tabs(["SHAP (Feature Importance)", "Model Performance (ROC)"])

with tab1:
    shap_candidates = [
        "SHAP_summary_internal_SVM_7f.png",
        "SHAP_summary_Top7.png",
        "SHAP_summary_SVM.png"
    ]
    shap_path = next((p for p in shap_candidates if os.path.exists(p)), None)

    if shap_path:
        st.image(shap_path, caption="SHAP Summary Plot")
    else:
        st.info("SHAP summary plot not found.")

with tab2:
    roc_candidates = [
        "ROC_external_SVM_7f.png",
        "ROC_best_model.png",
        "ROC_SVM.png"
    ]
    roc_path = next((p for p in roc_candidates if os.path.exists(p)), None)

    if roc_path:
        st.image(roc_path, caption="ROC Curve")
    else:
        st.info("ROC curve not found.")

st.divider()


# ==============================
# 页脚
# ==============================
st.caption("Disclaimer: This tool is for research purposes only and should not replace clinical judgment.")
