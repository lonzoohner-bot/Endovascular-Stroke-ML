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
├── model/                         # Saved model files
├── data/                          # Example or synthetic data, if available
├── figures/                       # Model performance and SHAP figures
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # License file
The exact repository structure may vary depending on deployment requirements.

## Local Installation

To run the web application locally, clone this repository:

git clone https://github.com/lonzoohner-bot/Endovascular-Stroke-ML.git
cd Endovascular-Stroke-ML

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate

For Windows users:

python -m venv venv
venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

The application will be available locally at:

http://localhost:8501
Requirements

The main dependencies include:

Python
Streamlit
scikit-learn
pandas
numpy
matplotlib
SHAP
XGBoost
LightGBM
CatBoost
imbalanced-learn

Please refer to requirements.txt for the complete list of dependencies and package versions.

## Data Availability

Individual-level patient data are not publicly available because of data ownership, institutional restrictions, and privacy considerations related to the ATTENTION registry and participating centers.

Data may be available from the corresponding author upon reasonable request and with permission from the ATTENTION registry steering committee and relevant participating institutions.

## Code Availability

The source code for model development, validation, interpretability analysis, and web-based deployment is available in this repository:

https://github.com/lonzoohner-bot/Endovascular-Stroke-ML

The deployed web-based calculator is available at:

https://endovascular-stroke-ml-hqfwtqqdryxzgwzouvyhlf.streamlit.app/

Intended Use

This tool is intended for research and clinical decision-support purposes only. It is designed to assist individualized risk estimation and clinician–patient communication.

It should not be used as a standalone basis for treatment decisions. Clinical judgment, institutional protocols, and multidisciplinary assessment remain essential.

## Privacy Statement

The web application does not intentionally store patient-level input data. Users should avoid entering personally identifiable information into the application.

## Limitations

This model was developed using registry-based data and externally validated in an independent multicenter cohort. Although it demonstrated good discrimination and acceptable calibration, several limitations should be considered:

The model identifies prognostic associations rather than causal effects.
The model should not be used as a standalone treatment decision tool.
Model performance may vary across healthcare systems, populations, imaging protocols, and EVT workflows.
Further prospective evaluation is needed before routine clinical implementation.
SHAP-based interpretation explains model behavior and does not establish causal relationships or actionable clinical thresholds.
## Citation

If you use this code or web application, please cite the associated manuscript:

Du J, Zhao Y, Deng S, et al. An Interpretable and Externally Validated Machine Learning Model for Predicting 90-Day Outcomes After Endovascular Treatment in Acute Basilar Artery Occlusion.

A formal citation will be added after publication.

## License

Please refer to the LICENSE file for details.

## Contact

For questions about the model, code, or data availability, please contact:

Guodong Xiao
Department of Neurology and Clinical Research Center of Neurological Disease
The Second Affiliated Hospital of Soochow University
Suzhou, Jiangsu Province, China
Email: gdxiao@suda.edu.cn

To run the web application locally, clone this repository:

```bash
git clone https://github.com/lonzoohner-bot/Endovascular-Stroke-ML.git
cd Endovascular-Stroke-ML
