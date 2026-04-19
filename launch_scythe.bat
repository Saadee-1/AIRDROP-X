@echo off
title AIRDROP-X Launcher
cd /d "%~dp0"

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
)

REM Try to find Python
set PYTHON_CMD=
where python >nul 2>nul && set PYTHON_CMD=python
if "%PYTHON_CMD%"=="" where py >nul 2>nul && set PYTHON_CMD=py
if "%PYTHON_CMD%"=="" where python3 >nul 2>nul && set PYTHON_CMD=python3

if "%PYTHON_CMD%"=="" (
    echo ERROR: Python not found. Please install Python 3.8+ and ensure it's in PATH.
    pause
    exit /b 1
)

REM Desktop window mode (PySide6, offline)
echo Starting AIRDROP-X...
%PYTHON_CMD% qt_app\main.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start.
    echo.
    echo Possible issues:
    echo 1. PySide6 not installed - run: pip install -r requirements.txt
    echo 2. Missing dependencies - run: pip install -r requirements.txt
    echo 3. Python version too old - requires Python 3.8+
    echo.
    pause
    exit /b 1
)
