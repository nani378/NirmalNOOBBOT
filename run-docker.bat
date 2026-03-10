@echo off
:: ============================================================
::  MoodBot — Docker launcher for Windows
::  Requires: Docker Desktop + VcXsrv (for display)
::  Usage:  git clone <repo> && cd NirmalNOOBBOT && run-docker.bat
:: ============================================================

echo ============================================
echo   MoodBot — Docker Launch (Windows)
echo ============================================

cd /d "%~dp0"

:: Check Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found. Install Docker Desktop from https://docker.com/products/docker-desktop
    pause
    exit /b 1
)

:: Check .env file
if not exist ".env" (
    echo.
    echo [ERROR] .env file not found!
    echo   Create a .env file with:  GROQ_API_KEY=your_key_here
    echo.
    pause
    exit /b 1
)

:: Create data dir
if not exist "data" mkdir data

:: Check for VcXsrv / X server
where vcxsrv >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARN] VcXsrv not found on PATH.
    echo   For the webcam window to display, install VcXsrv:
    echo   https://sourceforge.net/projects/vcxsrv/
    echo   Launch it with "Disable access control" checked.
    echo.
)

:: Get host IP for X11 display forwarding
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do set HOST_IP=%%b
)

echo [BUILD] Building MoodBot Docker image...
docker build -t moodbot .

echo.
echo [START] Launching MoodBot...
echo   Press Q in the video window to quit.
echo.

docker run --rm -it ^
    --env-file .env ^
    -e DISPLAY=%HOST_IP%:0.0 ^
    --device //./COM1 ^
    -v "%cd%\data":/app/data ^
    moodbot

pause
