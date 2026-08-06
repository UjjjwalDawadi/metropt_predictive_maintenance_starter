# Data Folder

The raw MetroPT-3 dataset is not included in this repository because of file size.

Download the dataset from the UCI Machine Learning Repository and place the raw file inside:

```text
data/raw/
```

The notebook `notebooks/01_data_audit.ipynb` automatically searches `data/raw/` for a CSV, TXT or Excel file.

Do not upload the raw dataset to GitHub. The `data/raw/` folder is ignored by Git.
