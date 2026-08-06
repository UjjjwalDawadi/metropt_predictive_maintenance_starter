# MetroPT-3 Predictive Maintenance Project

This repository contains my MSc Data Science dissertation project on early failure detection in a metro train air compressor using the MetroPT-3 dataset.

The project is currently at the methodology and early implementation stage. The aim is to build an explainable machine learning pipeline that can identify early warning patterns before documented compressor failures and support maintenance decision-making.

## Current contents

- `docs/methodology/` contains the final methodology chapter.
- `notebooks/01_data_audit.ipynb` starts the implementation by loading and auditing the dataset.
- `data/README.md` explains where to place the raw MetroPT-3 dataset.
- `outputs/` stores generated tables, metrics and figures.

## Dataset

The raw MetroPT-3 dataset is not included in this repository because of file size. Download it from the UCI Machine Learning Repository and place the raw file inside:

```text
data/raw/
```

## Setup

Recommended Python version: 3.10, 3.11 or 3.12.

On Windows, run:

```bash
setup_windows.bat
```

Then open the notebook and select the kernel:

```text
Python (FYP MetroPT)
```

## First notebook

Run the notebook below first:

```text
notebooks/01_data_audit.ipynb
```

It checks dataset shape, columns, timestamp range, missing values, duplicate timestamps and documented failure intervals.

## Ethical note

This project uses public machine sensor data only. It does not involve human participants, interviews, questionnaires or personal data.
