# Project Roadmap

## Stage 1: Data audit

- Load MetroPT-3 dataset
- Confirm shape, columns and timestamp range
- Check missing values and duplicate timestamps
- Save summary tables

## Stage 2: Preprocessing and EDA

- Clean timestamps
- Sort chronologically
- Handle missing values using leakage-safe rules
- Create sensor plots and correlation heatmap

## Stage 3: Feature engineering and labels

- Create 15-minute and 60-minute rolling features
- Create 5-minute, 15-minute and 60-minute lag features
- Create pressure-difference features
- Create 6h, 12h and 24h early-warning labels

## Stage 4: Modelling and evaluation

- Train Logistic Regression, Random Forest, XGBoost and Isolation Forest
- Use chronological validation
- Evaluate record-level and event-level metrics
- Tune threshold using validation data only

## Stage 5: Explainability and risk support

- Apply SHAP explanations
- Analyse true positives, false positives and false negatives
- Create provisional maintenance risk categories
