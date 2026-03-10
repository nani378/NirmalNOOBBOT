@echo off
:: ============================================================
::  MoodBot — One-click launcher for Windows
::  Installs dependencies (first run) and starts the app.
:: ============================================================

echo ============================================
echo   MoodBot — AI Emotion Companion
echo ============================================

:: Move to the script's own directory
cd /d "%~dp0"

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

:: Create venv if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install / update dependencies
echo [SETUP] Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

:: Check .env file exists
if not exist ".env" (
    echo.
    echo [ERROR] .env file not found!
    echo   Create a .env file with:  GROQ_API_KEY=your_key_here
    echo.
    pause
    exit /b 1
)

:: Run the app
echo.
echo [START] Launching MoodBot...
echo   Press Q in the video window to quit.
echo.
python main.py

pause
