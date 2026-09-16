@echo off
cd /d "%~dp0"
set "RECON_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "RECON_PYTHON=.venv\Scripts\python.exe"
if not exist ".venv\Scripts\python.exe" if exist "..\.venv\Scripts\python.exe" set "RECON_PYTHON=..\.venv\Scripts\python.exe"
"%RECON_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :end
"%RECON_PYTHON%" -m streamlit run app.py
:end
pause
