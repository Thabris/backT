@echo off
echo ================================================
echo   BackT - Professional Backtesting Framework
echo ================================================
echo.
echo Starting Streamlit web interface...
echo.
echo Your browser will open at: http://localhost:8501
echo.
cd streamlit_apps
"C:\Users\maxim\.conda\envs\backt\Scripts\streamlit.exe" run streamlit_backtest_runner.py
