@echo off
setlocal

echo Creating project virtual environment...
python -m venv .venv

if errorlevel 1 (
    echo Failed to create virtual environment. Check that Python is installed and added to PATH.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required packages...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo Package installation failed.
    pause
    exit /b 1
)

echo Registering Jupyter kernel...
python -m ipykernel install --user --name fyp-metropt --display-name "Python (FYP MetroPT)"

echo.
echo Setup complete.
echo Open notebooks/01_data_audit.ipynb and select kernel: Python (FYP MetroPT)
echo.
pause
