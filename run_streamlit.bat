@echo off
REM Batch script to activate backt conda environment and run Streamlit app

echo ====================================
echo BackT Streamlit Backtest Runner
echo ====================================
echo.

REM Activate the backt conda environment
echo Activating backt environment...
call conda activate backt

if errorlevel 1 (
    echo ERROR: Failed to activate backt environment
    echo Please make sure conda is installed and backt environment exists
    echo.
    pause
    exit /b 1
)

echo Environment activated successfully!
echo.

REM Change to the project directory (where this batch file is located)
cd /d "%~dp0"

REM Run Streamlit app
echo Starting Streamlit application...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_backtest_runner.py

REM If streamlit command fails
if errorlevel 1 (
    echo.
    echo ERROR: Failed to run Streamlit
    echo Make sure Streamlit is installed: pip install streamlit
    echo.
    pause
)
