# AIRDROP-X Simulation Framework

Probabilistic guidance and decision-support simulation for precision
unpowered payload delivery from UAV platforms.

## Overview
This repository contains a physics-based simulation framework for
modeling unpowered payload trajectories under uncertainty and
evaluating drop/no-drop decisions using probabilistic metrics.

## Current Status
- Monte Carlo trajectory simulation
- Hit probability estimation
- User-defined decision threshold
- Web-based preview interface
- Qt desktop application

## Quick Start

### Web Preview (Recommended)
Run the interactive web interface:
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

### Desktop Application
For the Qt-based desktop application:
```bash
python qt_app.py
```

### Command Line
For basic CLI simulation:
```bash
python main.py
```

## Features
- Real-time Monte Carlo simulation with configurable parameters
- Interactive decision threshold adjustment
- Impact dispersion visualization
- Advisory guidance system
- Reproducible results with seed control

## Disclaimer
This project is intended for research and simulation purposes only.
No autonomous release or weapon control functionality is implemented.
