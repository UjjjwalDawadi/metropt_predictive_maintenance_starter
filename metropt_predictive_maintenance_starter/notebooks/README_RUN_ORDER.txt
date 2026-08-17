# MetroPT RESET files: Steps 02 to 06

Use these files to restart the implementation from the confirmed full raw dataset.

Keep:
- data/raw/MetroPT3(AirCompressor).csv
- notebooks/

Delete old generated folders before rerunning:
- data/processed/
- outputs/tables/
- outputs/figures/
- outputs/predictions/
- models/

Run in this order using the Python (FYP MetroPT) kernel:
1. 02_preprocessing_eda_RESET.ipynb
2. 03_feature_engineering_RESET.ipynb
3. 04_label_creation_RESET.ipynb
4. 05_chronological_split_RESET.ipynb
5. 06_model_training_comparison_RESET.ipynb

Expected checks:
- Step 2: cleaned data ends around 2020-09-01, not April.
- Step 4: F1, F2, F3 and F4 appear in warning_12h_event_id.
- Step 5: train, validation and test are all non-empty.
- Step 6: models and prediction CSV files are saved.
