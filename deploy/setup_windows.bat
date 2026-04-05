@echo off
REM GoldSignalAI — Windows local setup
REM Usage: double-click or run from project root in cmd/PowerShell

echo === GoldSignalAI — Windows setup ===

REM Check Python 3.12 is available
python --version 2>nul | findstr /i "3.12" >nul
IF ERRORLEVEL 1 (
    echo ERROR: Python 3.12 not found. Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtualenv if missing
IF NOT EXIST "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Upgrade pip and install dependencies
echo Installing dependencies...
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt

REM Copy .env template if .env is missing
IF NOT EXIST ".env" (
    copy deploy\.env.template .env
    echo Copied deploy\.env.template -^> .env  ^(fill in your keys^)
)

REM Create required directories
IF NOT EXIST "logs\"          mkdir logs
IF NOT EXIST "models\"        mkdir models
IF NOT EXIST "data\historical\" mkdir data\historical
IF NOT EXIST "reports\"       mkdir reports
IF NOT EXIST "state\"         mkdir state
IF NOT EXIST "database\"      mkdir database

echo.
echo Setup complete. Run the bot with:
echo   venv\Scripts\python main.py
echo   venv\Scripts\python main.py --health-check
pause
