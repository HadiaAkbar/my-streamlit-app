@echo off
title MH[FG] Fake News Detector - Quick Start
color 0A
cls

echo ============================================
echo    MH[FG] FAKE NEWS DETECTOR
echo    Quick Start - No Installation Needed
echo ============================================
echo.

set "PROJECT_PATH=C:\Users\LTC\OneDrive\Desktop\MH[FG]"
cd /d "%PROJECT_PATH%"

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting Fake News Detector...
echo The app will open in your browser at: http://localhost:8501
echo Keep this window open while using the app.
echo Press Ctrl+C to stop when done.
echo.

timeout /t 2 /nobreak >nul

REM Start the application
streamlit run app.py --server.port=8501

echo.
echo ============================================
echo Application stopped.
echo ============================================
echo.
pause