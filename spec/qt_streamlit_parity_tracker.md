# SCYTHE Streamlit -> PySide6 Parity Tracker

Date: 2026-02-18  
Mode: Spec-first migration (no redesign)

## 1) Source of Truth (Locked)
- Streamlit reference code: `spec/streamlit_baseline/app.py.reference`
- Streamlit config reference: `spec/streamlit_baseline/config.toml.reference`
- Active Qt entrypoint: `qt_app/main.py`
- Active Qt shell/controller: `qt_app/main_window.py`

Rule: Jab tak parity pass nahi hoti, UI structure ya interaction behavior me creative change allowed nahi.

## 2) Current Snapshot (Ground Truth)
- Done:
  - PySide6 app shell + engine adapter integration
  - Evaluate lifecycle (lock/unlock snapshot)
  - Operator/Engineering mode switching
  - Telemetry worker thread (non-blocking)
  - Controlled auto-evaluate
  - Mission Overview tab now renderer-backed (`product/ui/tabs/mission_overview.py`)
  - Analysis tab now renderer-backed (`product/ui/tabs/analysis.py`)
  - Payload Library tab now renderer-backed (`product/ui/tabs/payload_library.py`)
  - Sensor & Telemetry tab now renderer-backed (`product/ui/tabs/sensor_telemetry.py`)
  - System Status tab now renderer-backed (`product/ui/tabs/system_status.py`)
  - Added `SNAPSHOT/LIVE` panel mode with state-driven control enable/disable rules
  - Added LIVE-only auto-evaluate visibility and gating
  - Added mode-aware Evaluate button labeling parity (`Evaluate Simulation` vs `Evaluate with Current Telemetry`)
  - Added Streamlit-style panel header/subtitle/snapshot meta in left configuration panel
  - Added richer LIVE telemetry mapping (UAV altitude + wind std updates)
  - Added mode-specific zoom control parity (`View Zoom`) with operator/engineering ranges
  - Wired zoom into Mission Overview + Analysis impact-dispersion rendering
  - Added Mission Overview mode badge text (`IMPACT DISPERSION - OPERATOR/ENGINEERING MODE`)
  - Added snapshot metadata caption strip (`Snapshot | Seed | Samples`) on Mission Overview and Analysis
  - Added LIVE badges (`Live`, `Telemetry-driven`) and dynamic `Wind σ` label in left panel
  - Added performance-based auto-evaluate pause indicator in LIVE mode
  - Added `User-defined assumption` badge for non-default Cd when editable
  - Updated system mode selector labels to Streamlit-style (`Snapshot`, `Live telemetry`)
  - Locked panel subtitle copy to baseline wording for Snapshot/Live states
  - Tuned Analysis tab headings to Streamlit-style labels (`P(HIT) vs ...`)
  - Added Analysis impact-dispersion mode badge (`IMPACT DISPERSION - OPERATOR/ENGINEERING MODE`)
  - Aligned left-panel input ranges/steps with Streamlit defaults (samples, altitude, wind step, threshold bounds)
  - Switched Mission Overview dispersion view to shared `plot_impact_dispersion` for mode-layer parity
  - Fixed probability scaling for ellipse color logic (% vs 0..1) in Mission/Analysis render paths
  - Improved Sensor & Telemetry tab with mode-aware live values (speed, altitude, wind, freshness/confidence)
  - Added System Status warning injection for LIVE auto-evaluate performance pause state
- Gap (closed in Phase 1 parity pass):
  - Left panel: labels/units aligned (Reference Area m², Wind σ in LIVE).
  - Tab content: Mission Overview (Confidence Index, THRESH), Sensor & Telemetry (mode badge), Analysis (badge/caption/CEP footer), System Status (section order).
  - Remaining: Payload Library interaction (Phase 2), optional screenshot polish.

## 3) Parity Matrix (Spec vs Qt)

| Area | Streamlit Spec | Current Qt | Gap Level | Action |
|---|---|---|---|---|
| Global layout | Left config + right content + 5 tabs | Present | Low | Spacing/geometry fine-tune against reference screenshots |
| Tab set/order | Mission Overview, Payload Library, Sensor & Telemetry, Analysis, System Status | Present | Low | Keep locked, no reorder |
| Top mode toggle | Operator/Engineering with active/inactive states | Present | Low | Validate exact color/weight parity |
| Evaluate flow | Evaluate -> snapshot lock -> Modify & Re-run unlock | Present | Low | Final behavior checklist run |
| Left panel sections | Payload, UAV, Target, Environment, Simulation, Telemetry Health, Auto-Evaluate, Decision Threshold | Present | Low | Done |
| Left panel field labels | Streamlit labels and units exact (m², Wind σ in LIVE) | Present | Low | Done |
| Left panel control states | Live/snapshot disable logic | Present | Low | Done |
| Mission Overview content | KPIs + decision cards + dispersion + Confidence Index, THRESH | Present | Low | Done |
| Analysis content | 2x top + 3x bottom + dispersion badge, CEP footer | Present | Low | Done |
| Payload Library tab | Category/payload selector + parameter blocks | Partial | Medium | Phase 2: Apply to mission |
| Sensor & Telemetry tab | Snapshot/live badge, panels | Present | Low | Done |
| System Status tab | Identity, Reproducibility, Numerical Stability, Limitations, Warnings order | Present | Low | Done |
| Plot layering rules | Operator minimal vs Engineering full | Present | Medium | Validate legend/layer visibility against Streamlit |
| Plot interaction | Zoom/pan available and stable | Present | Low | Confirm same default limits behavior |
| Styling/theme | Streamlit military HUD tone | Partial | Medium | Align fonts, borders, card backgrounds, tab visuals |

## 4) Catch-up Execution Plan

## Phase A - Baseline and Measurement (today)
1. Freeze Streamlit reference artifacts (done).
2. Prepare screenshot checklist for all tabs/states.
3. Create acceptance checklist with binary pass/fail.

## Phase B - Structural Parity (next)
1. Replace placeholder tabs with real tab widgets/content blocks.
2. Match tab-wise section order and heading text exactly.
3. Match left panel order and missing controls.

## Phase C - Behavioral Parity
1. Implement exact disabled/read-only transitions.
2. Match snapshot labels, telemetry status labels, and mode text.
3. Verify no unexpected rerun on mode/tab switch.

## Phase D - Visual Parity
1. Align typography, borders, spacing, card chrome.
2. Align plot annotation placement and legend positions.
3. Compare screenshots and close remaining deltas.

## Phase E - QA Gate
1. Run parity checklist (all tabs, both modes, lock/unlock states).
2. Run regression checks (telemetry thread, auto evaluate, mode switch, close event).
3. Mark migration as parity-pass only after checklist 100%.

## 5) Acceptance Criteria (Hard)
- Same tabs, same order, same labels.
- Same control order and default values.
- Same lock/unlock behavior and status messaging.
- Same mode-specific plotting behavior.
- No Streamlit runtime dependency.
- No browser/localhost behavior in active launch path.

## 6) Next Immediate Build Slice
Phase 1 parity pass completed (left panel labels, tab content/structure, System Status order, checklist updates).
Next:
1. Phase 2: Payload Library Apply-to-mission (category/payload selector → push mass/Cd/area to panel).
2. Phase 3: Real telemetry source option (Mock / Serial / UDP / File).
3. Phase 4: QA run + docs update; mark parity-pass when checklist 100%.
