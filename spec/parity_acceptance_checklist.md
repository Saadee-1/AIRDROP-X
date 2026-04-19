# SCYTHE Qt Parity Acceptance Checklist

Use this as pass/fail gate against `spec/streamlit_baseline/app.py.reference`.

## Global
- [x] App opens with `python qt_app/main.py`
- [x] No browser opens
- [x] No localhost/http dependency
- [x] Left config panel visible
- [x] 5 tabs visible in exact order
- [x] Operator default selected

## Evaluate Lifecycle
- [x] Initial state editable
- [x] Evaluate creates snapshot ID format `AX-YYYYMMDD-HHMMSS`
- [x] Inputs lock after evaluate
- [x] Button text changes to `Modify & Re-run`
- [x] Clicking `Modify & Re-run` unlocks inputs without rerun

## Mode Behavior
- [x] Operator tab style active when selected
- [x] Engineering tab style active when selected
- [x] Mode switch does not rerun simulation
- [x] Operator plot layers minimal
- [x] Engineering plot layers full

## Telemetry and Auto-Evaluate
- [x] Telemetry updates without UI freeze
- [x] Auto Evaluate OFF/1s/2s works
- [x] Auto evaluate runs only when snapshot is active
- [x] No overlapping simulation runs

## Tab Content Parity
- [x] Mission Overview blocks match Streamlit structure (banner, KEY METRICS, ADVISORY, dispersion, caption; THRESH, Confidence Index)
- [x] Payload Library controls match Streamlit structure (category/payload selector + Apply to mission)
- [x] Sensor & Telemetry blocks match Streamlit structure (mode badge, NAVIGATION/ATTITUDE/ENVIRONMENT panels)
- [x] Analysis card layout matches Streamlit structure (P(HIT) vs…, CEP SUMMARY, IMPACT DYNAMICS; Read-only · No recomputation; IMPACT DISPERSION — badge)
- [x] System Status sections match Streamlit structure (ENGINE IDENTITY, REPRODUCIBILITY, NUMERICAL STABILITY, LIMITATIONS, WARNINGS order)

## Visual Parity
- [x] Label texts and units match (Reference Area m², Wind σ in LIVE, THRESH, em dash in badges)
- [x] Section order matches (left panel; System Status section order)
- [ ] Colors and border emphasis near-identical (theme applied; optional screenshot pass)
- [ ] Legends and plot annotations match placement intent (optional screenshot pass)
