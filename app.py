from __future__ import annotations

import json
import math
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "model_spec.json"
ASSETS = ROOT / "assets"


@st.cache_data
def load_model_spec() -> dict:
    with SPEC_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def standardise(value: float, centre: float, scale: float) -> float:
    return (float(value) - float(centre)) / float(scale)


def predict_probability(inputs: dict, spec: dict, intercept_shift: float = 0.0) -> tuple[float, float]:
    eta = float(spec["intercept"]) + float(intercept_shift)

    for key, meta in spec["continuous_terms"].items():
        transformed = standardise(inputs[key], meta["centre"], meta["scale"])
        eta += float(meta["coefficient"]) * transformed

    occlusion_site = inputs["occlusion_site"]
    eta += float(spec["occlusion_site_terms"][occlusion_site])
    return eta, logistic(eta)


def contribution_table(inputs: dict, spec: dict) -> list[dict]:
    rows = []
    for key, meta in spec["continuous_terms"].items():
        transformed = standardise(inputs[key], meta["centre"], meta["scale"])
        contribution = float(meta["coefficient"]) * transformed
        rows.append(
            {
                "Variable": meta["display_name"],
                "Input": inputs[key],
                "Coefficient": round(float(meta["coefficient"]), 6),
                "Contribution": round(contribution, 6),
            }
        )

    occlusion_site = inputs["occlusion_site"]
    rows.append(
        {
            "Variable": "Occlusion site",
            "Input": occlusion_site,
            "Coefficient": round(float(spec["occlusion_site_terms"][occlusion_site]), 6),
            "Contribution": round(float(spec["occlusion_site_terms"][occlusion_site]), 6),
        }
    )
    return sorted(rows, key=lambda item: abs(item["Contribution"]), reverse=True)


def risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Lower estimated risk"
    if probability < 0.60:
        return "Intermediate estimated risk"
    return "Higher estimated risk"


def risk_colour(probability: float) -> str:
    if probability < 0.30:
        return "#166534"
    if probability < 0.60:
        return "#92400e"
    return "#991b1b"


def image_or_message(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure file not found: {path.name}")


def main() -> None:
    spec = load_model_spec()

    st.set_page_config(
        page_title="BAO EVT 90-day outcome calculator",
        page_icon="",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1.0rem;
            margin-bottom: 1.2rem;
        }
        .risk-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.0rem 1.1rem;
            background: #ffffff;
        }
        .risk-number {
            font-size: 3.0rem;
            font-weight: 700;
            line-height: 1;
            margin: 0.15rem 0 0.45rem 0;
        }
        .small-muted {
            color: #6b7280;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">90-Day Outcome Prediction After EVT for BAO</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Final seven-variable logistic regression model</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Model inputs")
        age = st.number_input("Age (years)", min_value=18.0, max_value=110.0, value=65.0, step=1.0)
        baselinenihss = st.number_input("Baseline NIHSS", min_value=0.0, max_value=42.0, value=22.0, step=1.0)
        sbp = st.number_input("SBP (mmHg)", min_value=60.0, max_value=260.0, value=160.0, step=1.0)
        pcaspects = st.number_input("PC-ASPECTS", min_value=0.0, max_value=10.0, value=8.0, step=1.0)
        onset_admission = st.number_input("OTA (min)", min_value=0.0, max_value=1440.0, value=360.0, step=10.0)
        asitnsir = st.number_input("ASITN/SIR collateral score", min_value=0.0, max_value=4.0, value=1.0, step=1.0)
        occlusion_site = st.selectbox("Occlusion site", ["middle", "proximal", "distal"], index=1)

        st.divider()
        intercept_shift = st.number_input(
            "Optional local intercept adjustment",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.01,
            help="Use 0.00 unless a local recalibration intercept has been estimated.",
        )

    inputs = {
        "age": age,
        "baselinenihss": baselinenihss,
        "sbp": sbp,
        "pcaspects": pcaspects,
        "onset_admission": onset_admission,
        "asitnsir": asitnsir,
        "occlusion_site": occlusion_site,
    }
    eta, probability = predict_probability(inputs, spec, intercept_shift=intercept_shift)

    calculator_tab, interpretation_tab, validation_tab = st.tabs(["Calculator", "Model interpretation", "Validation figures"])

    with calculator_tab:
        col1, col2 = st.columns([1.0, 1.15])
        with col1:
            st.subheader("Prediction result")
            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="small-muted">Probability of 90-day unfavourable outcome (mRS 4-6)</div>
                    <div class="risk-number" style="color:{risk_colour(probability)}">{probability * 100:.1f}%</div>
                    <div>{risk_band(probability)}</div>
                    <div class="small-muted">Linear predictor: {eta:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.subheader("Model transparency")
            st.markdown(
                """
                - Model type: final seven-variable logistic regression model
                - External validation AUC: 0.854
                - Outcome: 90-day unfavourable functional outcome, mRS 4-6
                - Intended use: research-use risk estimation and local validation
                """
            )
            st.warning(
                "This calculator is not a treatment-decision rule and should not replace clinician judgement.",
            )

        st.subheader("Entered values")
        st.table(
            {
                "Variable": [
                    "Age (years)",
                    "Baseline NIHSS",
                    "SBP (mmHg)",
                    "PC-ASPECTS",
                    "OTA (min)",
                    "ASITN/SIR collateral score",
                    "Occlusion site",
                ],
                "Value": [
                    age,
                    baselinenihss,
                    sbp,
                    pcaspects,
                    onset_admission,
                    asitnsir,
                    occlusion_site,
                ],
            }
        )

        st.subheader("Current-input contributions")
        st.table(contribution_table(inputs, spec))

    with interpretation_tab:
        st.subheader("Logistic regression coefficients")
        image_or_message(ASSETS / "LR_coefficients_7f.png", "Final logistic regression coefficients")
        st.subheader("SHAP summary")
        image_or_message(ASSETS / "SHAP_summary_LR_7f.png", "SHAP summary regenerated from final logistic regression coefficients and external-validation features")

    with validation_tab:
        st.subheader("External validation ROC")
        image_or_message(ASSETS / "ROC_external_LR_7f.png", "External validation ROC curve for the final logistic regression model")
        with st.expander("Model specification"):
            public_spec = {
                "model": spec["model"],
                "outcome": spec["outcome"],
                "external_validation_auc": spec["external_validation_auc"],
                "intercept": spec["intercept"],
                "continuous_terms": spec["continuous_terms"],
                "occlusion_site_terms": spec["occlusion_site_terms"],
                "probability_formula": spec["probability_formula"],
                "smote_rule": spec["smote_rule"],
            }
            st.json(public_spec)


if __name__ == "__main__":
    main()
