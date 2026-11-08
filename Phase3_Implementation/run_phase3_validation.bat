@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
"%PYTHON_CMD%" run_phase3_validation.py
if errorlevel 1 exit /b 1
"%PYTHON_CMD%" verify_outputs.py
if errorlevel 1 exit /b 1
echo Phase-3 validation completed successfully.
