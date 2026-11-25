@echo off
cd /d "%~dp0"
echo Installing required libraries...
py -m pip install -r requirements.txt
if errorlevel 1 python -m pip install -r requirements.txt
echo Launching AIR WATCH CHENNAI...
python -m streamlit run app.py
pause
