# MetroPT Streamlit Dashboard

## Purpose

This dashboard is a lightweight UI layer for the MetroPT-3 predictive maintenance dissertation project.

It does not retrain models. It reads the saved output files from your pipeline and displays:

- project overview
- model comparison
- alarm timeline
- alarm-burden analysis
- SHAP explainability
- risk framework
- evidence file checklist

## Where to put the file

Copy the `dashboard` folder into your project root:

```text
metropt_predictive_maintenance_starter/
├── dashboard/
│   └── streamlit_app.py
├── notebooks/
├── outputs/
├── models/
└── data/
```

## Install Streamlit

Activate your virtual environment, then run:

```powershell
pip install streamlit
```

or inside a notebook:

```python
%pip install streamlit
```

## Run the dashboard

From the project root:

```powershell
streamlit run dashboard/streamlit_app.py
```

## Important

Run the technical notebooks first, especially:

- Step 6 model training
- Step 7 threshold/event evaluation
- Step 7B alarm-burden sensitivity analysis
- Step 8 SHAP/risk framework

The dashboard depends on the CSV and figure files created by those steps.
