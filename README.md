# SCYTHE Simulation Framework

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
- Offline PySide6 desktop application

## Quick Start

### Desktop Application
For the Qt-based desktop application:
```bash
pip install -r requirements.txt
python qt_app/main.py
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
- Fully offline operation (no browser, no HTTP server)
- GCS-ready desktop UI: runs alongside any ground control station without embarrassment
- Payload Library: select category/payload and apply mass/Cd/area to mission
- Telemetry source: Mock (synthetic) or File playback (CSV replay) for LIVE mode

## Disclaimer
This project is intended for research and simulation purposes only.
No autonomous release or weapon control functionality is implemented.
