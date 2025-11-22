@echo off
echo ================================================
echo   Streamlit Setup for BackT
echo ================================================
echo.
echo This will install Streamlit in your backt environment
echo.
pause

echo.
echo Activating backt environment...
call C:\Users\maxim\.conda\condabin\conda.bat activate backt

echo.
echo Installing Streamlit...
pip install streamlit

echo.
echo ================================================
echo   Setup Complete!
echo ================================================
echo.
echo You can now run: launch_streamlit.bat
echo.
pause
