# MetroPT Streamlit Dashboard with Interactive Alarm Simulator

## What changed

This version adds an **Interactive Alarm Simulator** page.

The user can change:

- model score
- alarm threshold
- minimum persistent alarm duration
- date range

The dashboard then recalculates:

- event detected or missed
- lead time
- false alarm episodes per day
- total alarm time
- alarm time percentage
- precision and recall

This does not retrain the model and does not require raw dataset upload. It uses the saved prediction outputs from your existing pipeline.

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

If a previous dashboard file already exists, replace it with this one.

## Install Streamlit

Activate your virtual environment, then run:

```powershell
pip install streamlit
```

## Run the dashboard

From the correct project root, where you can see `notebooks`, `outputs`, `data`, and `dashboard`, run:

```powershell
streamlit run dashboard/streamlit_app.py
```

## Important

Run these technical notebooks first:

- Step 6 model training
- Step 7 threshold/event evaluation
- Step 7B alarm-burden sensitivity analysis
- Step 8 SHAP/risk framework

The dashboard depends on the CSV and figure files created by those steps.
