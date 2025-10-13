@echo off
REM Batch script to activate backt conda environment and run Streamlit app

echo ====================================
echo BackT Streamlit Backtest Runner
echo ====================================
echo.

REM Try to find and initialize conda
echo Looking for conda installation...

REM Common conda installation paths
set CONDA_PATHS="%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\.conda" "C:\ProgramData\Anaconda3" "C:\ProgramData\Miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3"

set CONDA_FOUND=0
for %%p in (%CONDA_PATHS%) do (
    if exist "%%~p\Scripts\activate.bat" (
        echo Found conda at %%~p
        call "%%~p\Scripts\activate.bat" "%%~p"
        set CONDA_FOUND=1
        goto :activate_env
    )
)

:activate_env
if %CONDA_FOUND%==0 (
    echo ERROR: Could not find conda installation
    echo Please make sure Anaconda or Miniconda is installed
    echo Common locations: %USERPROFILE%\anaconda3 or %USERPROFILE%\miniconda3
    echo.
    pause
    exit /b 1
)

REM Activate the backt environment
echo Activating backt environment...
call conda activate backt

if errorlevel 1 (
    echo ERROR: Failed to activate backt environment
    echo Please make sure the 'backt' environment exists
    echo Create it with: conda create -n backt python=3.10
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
