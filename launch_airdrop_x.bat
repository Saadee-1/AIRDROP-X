@echo off
title AIRDROP-X Launcher
cd /d "%~dp0"

REM Desktop window mode (pywebview) — web pe nahi, alag window mein
where python >nul 2>nul && python main.py || py -3 main.py

if errorlevel 1 pause
