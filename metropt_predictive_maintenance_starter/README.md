# Explainable Predictive Maintenance Framework for MetroPT-3

MSc Data Science dissertation project developing an explainable machine learning framework for early failure detection in a metro train air-compressor system using the MetroPT-3 dataset.

The project combines time-series feature engineering, chronological model evaluation, alarm-burden analysis, SHAP explainability and an interactive Streamlit dashboard to support predictive-maintenance decision making.

## Live Dashboard

The deployed Streamlit dashboard is available at:

https://metropt-predictive-maintenance-system.streamlit.app/

## Project Overview

The project uses sensor data from the MetroPT-3 Air Production Unit to investigate whether abnormal operating behaviour can be identified before documented air-leak failures.

The implementation covers:

- data auditing and preprocessing
- leakage-safe time-series feature engineering
- early-warning label construction
- chronological train, validation and test splitting
- comparison of supervised and anomaly-detection models
- validation-based threshold and persistence selection
- event-level failure detection
- lead-time and false-alarm evaluation
- SHAP-based model explainability
- prototype maintenance-risk categorisation
- interactive Streamlit dashboard development

## Models

Four machine-learning approaches were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- Isolation Forest

Model evaluation was not based only on conventional classification metrics. The framework also considers:

- F2-score
- failure-event detection
- first-alert lead time
- false-alarm episodes per day
- percentage of time in alarm
- persistence of warning signals

## Final Model Selection

Model and alarm-rule selection was performed using the validation period only.

The final validation-selected rule was:

| Item | Result |
|---|---|
| Model | Logistic Regression |
| Threshold | 0.99 |
| Persistence | 5 minutes |
| Validation event detected | F3: 1/1 |
| Validation lead time | approximately 7.7 minutes |
| False-alarm episodes/day | 0.000 |
| Validation alarm time | 0.012269% |
| F2-score | 0.012303 |

The selected Logistic Regression rule was then applied unchanged to the held-out F4 test period.

It did not detect F4.

Among the independently validation-locked comparator rules, Isolation Forest showed the strongest held-out behaviour:

| Item | Result |
|---|---|
| Model | Isolation Forest |
| Locked rule | 0.42 threshold / 5-minute persistence |
| F4 detected | 1/1 |
| Lead time | 11.998 hours |
| F2-score | 0.627968 |
| False-alarm episodes/day | 1.373218 |
| Alarm time | 4.417684% |

The Isolation Forest result is reported as held-out comparator evidence and was not used to retrospectively change the validation-selected model.

## Time-Series Feature Engineering

The feature-engineering pipeline was designed to preserve temporal order and avoid future-information leakage.

The final modelling matrix contains 114 predictors and includes:

- original compressor sensor measurements
- pressure-difference features
- 15-minute rolling statistics
- 60-minute rolling statistics
- 5-minute lag information
- 15-minute lag information
- 60-minute lag information
- 5-minute rate-of-change features

The import-generated `Unnamed: 0` / `Unnamed__0` field was excluded from modelling.

## Early-Warning Framework

Failure-warning labels were generated before documented failure intervals.

The main modelling target uses a 12-hour pre-failure warning window while excluding observations occurring during the actual failure interval.

The data was divided chronologically so that later observations could not influence earlier modelling stages.

This preserves the time-dependent nature of predictive maintenance and avoids random train/test leakage.

## Explainability

SHAP was used to explain the selected Logistic Regression model.

The final explanation process uses:

- `SHAP LinearExplainer`
- training-derived background data
- the same fitted preprocessing pipeline used by the model
- held-out test observations for explanation

Important influential features included longer-duration behaviour in:

- oil temperature
- TP2
- H1
- motor current

particularly 60-minute rolling statistics.

SHAP values are interpreted as model attributions rather than evidence of mechanical causation.

## Prototype Risk Framework

The dashboard translates the selected model score into five prototype decision-support categories:

- Normal
- Low
- Medium
- High
- Critical

The risk framework is linked to the selected Logistic Regression threshold of `0.99` and five-minute persistence requirement.

These categories are research-prototype outputs and are not engineering-certified railway maintenance instructions.

## Dashboard

The Streamlit application provides five main pages:

### Overview

Introduces the project, framework and main findings.

### Model and Alarm Evaluation

Displays model comparison, validation-selected alarm rules and held-out evaluation results.

### Interactive Alarm Simulator

Allows thresholds, persistence periods, model scores and date ranges to be explored interactively.

### Timeline Explorer

Displays model scores, warning labels and alarm behaviour across selected time periods.

### Explainability and Risk Framework

Presents SHAP-based explanations together with the prototype maintenance-risk framework.

The dashboard loads saved outputs from the completed analysis pipeline. It does not retrain the machine-learning models during a user session.

## Repository Structure

```text
metropt_predictive_maintenance_starter/
│
├── dashboard/
│   └── streamlit_app.py
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_preprocessing_eda_RESET.ipynb
│   ├── 03_feature_engineering_RESET.ipynb
│   ├── 04_label_creation_RESET.ipynb
│   ├── 05_chronological_split_RESET.ipynb
│   ├── 06_model_training_comparison_RESET.ipynb
│   ├── 07_threshold_event_evaluation.ipynb
│   ├── 07B_alarm_burden_sensitivity_analysis.ipynb
│   └── 08_shap_risk_framework.ipynb
│
├── outputs/
│
├── outputs_deploy/
│
├── requirements.txt
├── streamlit_app.py
└── README.md
```

Some notebook filenames may contain version suffixes in the repository where intermediate development versions were retained.

## Notebook Workflow

The main experimental workflow follows this order:

1. **Data Audit**  
   Load and inspect the MetroPT-3 dataset.

2. **Preprocessing and EDA**  
   Prepare timestamped sensor data and examine operating behaviour.

3. **Feature Engineering**  
   Generate leakage-safe rolling, lag, rate-of-change and pressure-difference features.

4. **Early-Warning Labels**  
   Construct pre-failure warning targets around documented failure events.

5. **Chronological Split**  
   Divide the dataset into train, validation and held-out test periods.

6. **Model Training and Comparison**  
   Train Logistic Regression, Random Forest, XGBoost and Isolation Forest.

7. **Threshold and Event Evaluation**  
   Evaluate candidate warning thresholds and event-level behaviour.

8. **Alarm-Burden Sensitivity Analysis**  
   Select threshold/persistence rules using validation data and operational burden constraints.

9. **SHAP and Risk Framework**  
   Explain the selected model and generate prototype maintenance-risk outputs.

## Installation

Clone the repository:

```bash
git clone https://github.com/UjjjwalDawadi/metropt_predictive_maintenance_starter.git
cd metropt_predictive_maintenance_starter
```

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Dashboard Locally

Run:

```bash
python -m streamlit run dashboard/streamlit_app.py
```

The application will open in a local browser.

## Dataset

This project uses the MetroPT-3 dataset from the UCI Machine Learning Repository.

The raw dataset is not included in this repository because of its size.

Place the downloaded raw data in the appropriate `data/` location before running the complete experimental pipeline.

The deployed dashboard does not require the complete raw dataset because it uses selected saved outputs stored in `outputs_deploy/`.

## Reproducibility

The project separates:

- raw data
- processed data
- engineered features
- model outputs
- prediction files
- evaluation results
- figures
- dashboard-ready deployment outputs

The held-out test period was not used to select or retune the final model or alarm rule.

Random seeds and saved preprocessing/model artefacts are used where applicable to support reproducibility.

## Limitations

MetroPT-3 contains a large number of timestamped observations but only a small number of documented failure events.

The final chronological test evaluation therefore contains only one held-out failure event.

Results should consequently be interpreted as evidence from a research prototype rather than proof of general performance across different compressors, operating conditions or railway systems.

Further work should evaluate the framework using additional failure events and compressors and incorporate engineering validation of alarm thresholds and maintenance actions.

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Streamlit

## Academic Project

This repository was developed as part of an MSc Data Science dissertation.

**Project:** Explainable Predictive Maintenance Framework for MetroPT-3  
**Author:** Ujjwal Dawadi  
**Year:** 2026

The software and dashboard are intended for research and demonstration purposes and should not be interpreted as a safety-certified railway maintenance system.