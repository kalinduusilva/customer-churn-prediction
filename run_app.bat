@echo off
title ChurnGuard - Customer Churn Intelligence System
cd /d "%~dp0"

echo ======================================================================
echo   Starting ChurnGuard: Customer Churn Intelligence System
echo   Developer: Kalindu Mihiranga Silva
echo ======================================================================
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv\
    echo Please create the virtual environment and install dependencies first:
    echo    python -m venv .venv
    echo    .\.venv\Scripts\activate.bat
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Free port 8501 if an existing or hung instance is holding it
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1

:: Activate environment and run Streamlit
echo [1/2] Activating Python virtual environment...
call .venv\Scripts\activate.bat

echo [2/2] Launching Streamlit dashboard...
echo.
python -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application terminated unexpectedly.
    pause
)
