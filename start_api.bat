@echo off
echo Starting Synthetic Data Generation API...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install API requirements if needed
echo Installing API requirements...
pip install -r requirements-api.txt

REM Start the API server
echo.
echo Starting API server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python run_api.py --env development --reload

pause