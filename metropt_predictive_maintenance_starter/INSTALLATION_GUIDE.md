# Installation Guide

## Windows setup

1. Extract this ZIP file.
2. Open the extracted folder in VS Code.
3. Open Terminal in VS Code.
4. Run:

```bash
setup_windows.bat
```

5. Open `notebooks/01_data_audit.ipynb`.
6. Select kernel: `Python (FYP MetroPT)`.
7. Run the first test cell.

## If the kernel does not appear

Run this manually in the VS Code terminal:

```bash
.venv\Scriptsctivate
python -m ipykernel install --user --name fyp-metropt --display-name "Python (FYP MetroPT)"
```

Then reopen the notebook and select the kernel again.
