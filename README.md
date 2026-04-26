# Endovascular-Stroke-ML: Interpretable Outcome Prediction After EVT in Acute Basilar Artery Occlusion

This repository contains the source code and web-based deployment files for an interpretable machine learning model developed to predict 90-day unfavorable functional outcome after endovascular treatment (EVT) in patients with acute basilar artery occlusion (BAO).

The model was developed using data from the ATTENTION registry and externally validated in an independent multicenter cohort. The final deployed model is a support vector machine (SVM) model selected based on discrimination, calibration, robustness, and clinical interpretability.

## Overview

Acute basilar artery occlusion is associated with high mortality and disability, and clinical outcomes after EVT remain heterogeneous even after successful angiographic reperfusion. This project aims to provide an interpretable risk prediction tool that estimates the individualized probability of 90-day unfavorable functional outcome after EVT.

The primary outcome is:

- **Unfavorable functional outcome:** modified Rankin Scale (mRS) score of 4–6 at 90 days.

The model uses routinely available clinical and imaging variables to support individualized risk stratification.

## Web Application

The web-based calculator is publicly available at:

**https://endovascular-stroke-ml-hqfwtqqdryxzgwzouvyhlf.streamlit.app/**

The application allows users to input patient-level clinical and imaging variables and returns an individualized predicted probability of 90-day unfavorable functional outcome.

## Input Variables

The final model uses the following seven predictors:

1. Age  
2. Baseline NIHSS score  
3. Systolic blood pressure  
4. PC-ASPECTS  
5. Onset-to-admission time  
6. Occlusion site  
7. ASITN/SIR collateral score  

These variables were selected based on model-based feature ranking, LASSO regression, bootstrap-based stability analysis, and clinical interpretability.

## Model Development

Multiple supervised machine learning algorithms were evaluated, including:

- Logistic regression
- Random forest
- Support vector machine
- XGBoost
- LightGBM
- CatBoost
- Voting ensemble
- Stacking ensemble

The final SVM model was selected based on overall predictive performance, calibration, robustness, and model simplicity.

## Model Performance

In the manuscript, the final SVM model achieved:

- **Development cohort AUC:** 0.819  
- **External validation cohort AUC:** 0.852  
- **Improved calibration after recalibration-in-the-large in the external cohort**

Model performance was assessed using:

- Area under the receiver operating characteristic curve
- Accuracy
- Sensitivity
- Specificity
- F1 score
- Calibration curves
- Calibration intercept and slope
- Brier score
- Decision curve analysis

## Model Interpretability

SHapley Additive exPlanations (SHAP) were used to interpret model predictions.

The main interpretability outputs include:

- Global feature importance
- SHAP summary plots
- SHAP dependence plots
- Individual-level prediction explanations

In the final model, baseline NIHSS score, PC-ASPECTS, and collateral status were among the most influential predictors.

SHAP values should be interpreted as measures of feature contribution within the fitted model and should not be interpreted as causal effects.

## Repository Structure

```text
Endovascular-Stroke-ML/
│
├── app.py                         # Streamlit web application
├── best_single_model_SVM_7f.pkl   # Saved final SVM model
├── final_features_list.pkl        # List of final model input features
├── ROC_external_SVM_7f.png        # ROC curve for external validation
├── SHAP_summary_Top7.png          # SHAP summary plot
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .devcontainer/                 # Development container configuration


