@echo off
title MH[FG] Fake News Detector
color 0A
cls

echo ============================================
echo    MH[FG] FAKE NEWS DETECTOR
echo    Your AI Shield Against Fake News
echo ============================================
echo.

set "PROJECT_PATH=C:\Users\LTC\OneDrive\Desktop\MH[FG]"
cd /d "%PROJECT_PATH%"

echo [1/4] Checking Python version...
python --version

REM CHECK FOR PYTHON 3.12.7 (LOCAL) OR ANY VERSION
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYVER=%%i"
echo Detected Python version: %PYVER%

REM Extract major.minor version
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do set "PY_MAJOR_MINOR=%%a.%%b"

if not "%PY_MAJOR_MINOR%"=="3.12" (
    echo.
    echo ⚠️  WARNING: Python 3.12.x recommended for local development!
    echo Current version: %PYVER%
    echo Expected: 3.12.7 (or any 3.12.x)
    echo Note: Streamlit Cloud uses Python 3.9
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" (
        pause
        exit /b 1
    )
    echo.
)

echo Python %PYVER% detected, proceeding...
echo.

echo [2/4] Setting up virtual environment...

REM Clean up old installation
if exist "venv" (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo Creating new virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel

echo [3/4] Installing packages for local development...

REM Use your new requirements for LOCAL development
echo Creating requirements for local use...
(
echo streamlit==1.28.1
echo pandas==1.5.3
echo numpy==1.21.6
echo scikit-learn==1.0.2
echo nltk==3.8.1
echo joblib==1.2.0
echo matplotlib==3.7.2
echo seaborn==0.12.2
echo plotly==5.17.0
echo textblob==0.17.1
echo beautifulsoup4==4.12.2
echo requests==2.31.0
) > requirements_local.txt

pip install -r requirements_local.txt --no-cache-dir
del requirements_local.txt

echo [4/4] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
echo NLTK data downloaded successfully.

echo.
echo ============================================
echo READY! Starting MH[FG] Fake News Detector...
echo ============================================
echo.
echo The app will open in your browser shortly.
echo KEEP THIS WINDOW OPEN while using the app.
echo.
echo Press Ctrl+C to stop the application.
echo.

timeout /t 3 /nobreak >nul

REM Start the application
streamlit run app.py --server.port=8501

echo.
echo ============================================
echo Application stopped.
echo ============================================
echo.
pause