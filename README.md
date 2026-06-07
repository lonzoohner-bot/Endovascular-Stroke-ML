# BAO EVT 90-day outcome calculator

This repository implements the final seven-variable logistic regression model for estimating 90-day unfavourable functional outcome after endovascular treatment for acute basilar artery occlusion.

The previous deployment files should be replaced with the files in this folder. This package implements only the final logistic regression model.

## Final model

- Model: seven-variable logistic regression
- Outcome: 90-day unfavourable functional outcome, mRS 4-6
- External validation AUC: 0.854
- Predictors: Age (years), Baseline NIHSS, SBP (mmHg), PC-ASPECTS, OTA (min), Occlusion site, ASITN/SIR collateral score

## Files

- `app.py`: Streamlit calculator using the logistic regression formula directly.
- `model_spec.json`: Intercept, coefficients, training-set centre/scale, and occlusion-site coding.
- `final_lr_coefficients.csv`: Tabular coefficient export for independent verification.
- `final_features_list.json`: Final seven predictors and display labels.
- `assets/ROC_external_LR_7f.png`: External ROC curve regenerated from prediction outputs.
- `assets/LR_coefficients_7f.png`: Logistic regression coefficient plot regenerated from the coefficient table.
- `assets/SHAP_summary_LR_7f.png`: Logistic-regression SHAP summary regenerated from model coefficients and external-validation features.
- `assets/SHAP_values_LR_7f.csv`: Source data for the SHAP summary plot.
- `.streamlit/config.toml`: Streamlit theme configuration.
- `requirements.txt`: Minimal Python dependencies.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Probability formula

The calculator computes:

```text
p = 1 / (1 + exp(-eta))
eta = intercept + sum(beta_j * transformed_feature_j)
```

Continuous or ordinal predictors are standardised using the training-set centre and scale in `model_spec.json`.
Occlusion site is represented by one-hot indicator coefficients for middle, proximal, and distal locations.

SMOTE was used only during model fitting and is not applied to new patients at prediction time.

This calculator is intended for research-use risk estimation and local validation. It should not replace clinician judgement or be used as a sole basis for treatment decisions.
