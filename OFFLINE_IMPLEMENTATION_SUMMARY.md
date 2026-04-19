# SCYTHE Offline Implementation Summary

Date: 2026-02-18

## Completed
- Migrated UI to native PySide6 desktop app (`qt_app/main.py`).
- Removed legacy web entrypoint and related configuration.
- Updated launcher and README to desktop-only startup path.
- Updated requirements to desktop runtime stack.
- Parity pass: left panel labels/units (Reference Area m², Wind σ in LIVE), tab content and section order aligned with Streamlit reference, confidence index in Mission Overview, System Status section order.
- Payload Library: category/payload selector and **Apply to mission** to push mass/Cd/area into mission config.
- Telemetry source option: **Mock** (default) or **File playback** (CSV replay). In LIVE mode, left panel **TELEMETRY SOURCE** allows choosing source and file path; Apply restarts the worker.

## Runtime Characteristics
- No browser dependency.
- No HTTP server requirement.
- No network connectivity requirement for local execution.
- GCS-ready: desktop app can sit alongside any ground control station; fully offline and self-contained.
