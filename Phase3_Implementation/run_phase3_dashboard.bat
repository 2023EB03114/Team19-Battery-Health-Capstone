@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
"%PYTHON_CMD%" -m streamlit run app.py
