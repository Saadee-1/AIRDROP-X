@echo off
title AIRDROP-X Launcher
cd /d "%~dp0"

REM Desktop window mode (pywebview) — web pe nahi, alag window mein
python main.py

if errorlevel 1 pause
