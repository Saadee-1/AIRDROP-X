# SCYTHE Qt Parity Check Report

Date: 2026-02-18
Reference: `spec/parity_acceptance_checklist.md`

## Status Legend
- PASS: Implemented and statically verified from code path.
- PENDING-RUNTIME: Requires live app execution validation.
- PARTIAL: Implemented but needs visual/manual parity pass.

## Global
- PASS: No browser/localhost/webview runtime path in active Qt launch.
- PASS: Left config panel exists.
- PASS: 5 tabs exist in exact order.
- PASS: Operator default selected.
- PASS: Runtime smoke validation passed with PySide6 venv (offscreen instantiation).

## Evaluate Lifecycle
- PASS: Initial state editable (snapshot unlocked).
- PASS: Snapshot ID format `AX-YYYYMMDD-HHMMSS`.
- PASS: Inputs lock after evaluate.
- PASS: Evaluate button switches to `Modify & Re-run`.
- PASS: Unlock path does not rerun simulation.

## Mode Behavior
- PASS: Operator/Engineering active state visuals present.
- PASS: Mode switch does not rerun simulation.
- PASS: Mode-specific zoom ranges (Operator/Engineering) implemented.
- PASS: Operator/Engineering layer behavior now uses shared `plot_impact_dispersion`.
- PENDING-RUNTIME: Manual visual confirmation of layer prominence in app window.

## Telemetry and Auto-Evaluate
- PASS: Telemetry worker thread updates panel fields.
- PASS: Auto Evaluate OFF/1s/2s controls implemented (LIVE only).
- PASS: Auto evaluate runs only when snapshot active.
- PASS: Overlap guard via `simulation_running`.
- PASS: Performance auto-pause (`>1.5s`) with UI indicator.
- PENDING-RUNTIME: Confirm no UI hitch under sustained auto-evaluate.

## Tab Content Parity
- PASS: Mission Overview structure aligned (banner with Confidence Index, THRESH, KEY METRICS, ADVISORY, dispersion badge).
- PARTIAL: Payload Library renderer-backed; Phase 2 will add Apply-to-mission interaction.
- PASS: Sensor & Telemetry blocks aligned (mode badge Live vs Snapshot Mode – Not Live Feed; panel layout).
- PASS: Analysis card layout aligned (P(HIT) vs…, CEP SUMMARY, IMPACT DYNAMICS; IMPACT DISPERSION — badge; Read-only · No recomputation).
- PASS: System Status sections aligned (ENGINE IDENTITY, REPRODUCIBILITY, NUMERICAL STABILITY, LIMITATIONS, WARNINGS order).

## Visual Parity
- PASS: Label texts and units aligned (Reference Area m², Wind σ in LIVE, THRESH, em dash in badges).
- PASS: Section order aligned (left panel; System Status).
- PARTIAL: Colors and borders; theme applied; optional screenshot pass for pixel-level match.
- PARTIAL: Legends/annotations; optional screenshot pass for placement.

## Next Closing Tasks
1. (Optional) Run manual screenshot comparison for Mission Overview and Analysis (operator + engineering).
2. Run one full interactive checklist session in target environment.
3. Phase 2: Payload Library Apply-to-mission; Phase 3: Real telemetry source option.
