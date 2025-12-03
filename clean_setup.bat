@echo off
echo COMPLETE CLEAN SETUP FOR MH[FG]
echo.

cd /d "C:\Users\LTC\OneDrive\Desktop\MH[FG]"

echo Step 1: Removing old virtual environment...
if exist venv rmdir /s /q venv

echo Step 2: Creating new virtual environment...
python -m venv venv

echo Step 3: Activating and upgrading pip...
call venv\Scripts\activate.bat
python -m ensurepip --upgrade
python -m pip install --upgrade pip

echo Step 4: Installing ALL packages...
pip install numpy==1.26.4 pandas==2.2.0 scikit-learn==1.5.0 scipy==1.12.0
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.38.0 streamlit==1.32.0 nltk==3.8.1
pip install plotly==5.20.0 matplotlib==3.8.3 requests==2.31.0 beautifulsoup4==4.12.3 joblib==1.3.2

echo Step 5: Testing installation...
python -c "import numpy, pandas, sklearn, torch, transformers, streamlit, nltk; print('SUCCESS! All packages installed.')"

echo.
echo SETUP COMPLETE!
echo Now use mhfg_final.bat to run the application.
pause