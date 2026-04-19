# SCYTHE Offline Analysis Report

Date: 2026-02-18

## Status
- Desktop UI is fully PySide6-based.
- Legacy web UI stack has been removed from runtime paths.
- No browser or HTTP server startup path remains in active launch flow.

## Launch
Use:

```bash
python qt_app/main.py
```

## Notes
- Engine logic remains unchanged.
- UI, telemetry, and plotting operate in-process as native desktop components.
