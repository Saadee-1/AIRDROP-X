# SCYTHE Data Propagation Diagnosis Report

**Date:** 2025-02-22  
**Scope:** Pipeline trace — Run Simulation → Engine → Evaluation Object → Snapshot Storage → UI Binding → Render  
**Method:** Trace, print, report only. No fixes applied.

---

## 1) Simulation Function Name

| Trigger | Function | Location |
|---------|----------|----------|
| **"New Simulation"** (card click) | `_on_new_simulation_clicked()` | `qt_app/main_window.py:926` |
| **"Run Simulation"** (READY block click, Evaluate button, auto-evaluate) | `_start_simulation(trigger)` | `qt_app/main_window.py:1407` |

**Execution path for Run Simulation:**
1. `_start_simulation(trigger)` → creates `SimulationWorker(cfg, trigger)`
2. `SimulationWorker.run()` → calls `run_simulation_snapshot(config_override=cfg, include_advisory=True)` (adapter)
3. `adapter.run_simulation_snapshot()` → calls `get_impact_points_and_metrics()` and `evaluate_advisory()` (engine)
4. Adapter returns `dict` → stored in variable `snapshot` in `SimulationWorker.run()`
5. `simulation_done.emit(snapshot, trigger)` → `_on_simulation_done(snapshot, trigger)`

**Note:** "New Simulation" resets state and navigates to Mission Config; it does **not** run the engine.

---

## 2) Engine Return Structure

**Engine entry point:** `qt_app/adapter.py` → `run_simulation_snapshot()`

**Engine calls:**
- `product/guidance/advisory_layer.py::get_impact_points_and_metrics(mission_state, random_seed)`
- `product/guidance/advisory_layer.py::evaluate_advisory(mission_state, threshold_pct, random_seed)`

**Return type:** `Dict[str, Any]` (not an object with attributes)

**Variable storing result:** `snapshot` (in SimulationWorker), then `snapshot` (in `_on_simulation_done`)

**Returned dict keys:**
```
impact_points, hits, P_hit, cep50, target_position, target_radius,
advisory, wind_vector, impact_velocity_stats, snapshot_id, confidence_index,
telemetry, n_samples, random_seed, threshold_pct, ci_low, ci_high, p_hat,
decision, decision_reason, doctrine_mode, doctrine_description
```

**Does NOT include:** `snapshot_type`, `evaluation`, `system_mode`, `mission_mode`

---

## 3) Snapshot Structure

**Storage variable:** `self._latest_snapshot` (dict, `MainWindow`)

**Structure (post-`_on_simulation_done`):**
- Adapter dict + `snapshot_type: "EVALUATION"` (added in `_on_simulation_done`)
- `mission_mode` (from `config_state.data`)

**Where snapshot is updated:**
- `_on_simulation_done()` — after SimulationWorker returns (line ~1429)
- `_on_mission_config_committed()` — CONFIG snapshot (line ~610)
- `_on_new_simulation_clicked()` — minimal `{"threshold_pct": th}` (line 934)
- `_handle_evaluation_result()` — LIVE mode evaluation (line 1626)

**Evaluation object inside snapshot?** No. The snapshot dict **is** the evaluation data (flat). `snapshot.get("evaluation")` is always `None`.

---

## 4) Decision Block Source

**UI component:** `decision_state_card`, `decision_label` (in `_build_mission_tab_operator`)

**Update method:** `_render_mission_tab_operator(snapshot, decision, p_hit, cep50, ...)`

**Variable read:** `decision` (parameter)

**Source chain:**
1. `_render_mission_tab()` → `decision = snapshot.get("decision")` (when EVALUATION and `app_state == EVALUATED`)
2. Or `decision = "READY"` (CONFIG)
3. Or `decision = "PAUSED"` (paused state)
4. Passed to `_render_mission_tab_operator(..., decision, ...)`

**Confirmation:** Decision block reads `snapshot.decision` (via `_render_mission_tab`). It does **not** read `snapshot.system_mode`. `snapshot.system_mode` is never set.

---

## 5) Stats Field Sources

| Field | Update method | Variable source |
|-------|---------------|-----------------|
| **HIT %** | `_render_mission_tab_operator` | `p_hit` ← `snapshot.get("P_hit")` |
| **HITS** | `_render_mission_tab_operator` | `snapshot.get("hits")`, `snapshot.get("n_samples")` |
| **Stability** | `_render_mission_tab_operator` | `snapshot.get("confidence_index")` |
| **95% CI** | `_render_mission_tab_operator` | `snapshot.get("ci_low")`, `snapshot.get("ci_high")` |
| **CI width** | `_render_mission_tab_operator` | `(ci_high - ci_low) * 100` |
| **Sample count** | `_render_mission_tab_operator` | `snapshot.get("n_samples")` |
| **CEP50** | `_render_mission_tab_operator` | `cep50` ← `snapshot.get("cep50")` |

All stats derive from `snapshot`; none read directly from `left_panel` or telemetry in EVALUATION mode.

---

## 6) Analytical Renderer Source

**Renderer:** `product/ui/tabs/analysis.py::render()`

**Caller:** `_render_analysis_tab()` in `main_window.py` (line ~952)

**Data flow:**
- `impact_points` ← `snapshot.get("impact_points", [])`
- `p_hit` ← `snapshot.get("P_hit")`
- `cep50` ← `snapshot.get("cep50")`
- `target_position`, `target_radius` ← snapshot
- `dispersion_mode` ← `self.current_mode` (operator/engineering)

**Source:** Snapshot. Both operator and analytical modes use the same `_latest_snapshot`; `impact_points` come from `snapshot.impact_points`.

**Operator vs Analytical:** Same dataset. Mode only affects display density (operator = scatter, engineering = density overlay).

---

## 7) Exact Location Where Propagation Stops

### A) New Simulation path

**Location:** `_on_new_simulation_clicked()` → `_render_mission_tab()` (line 940)

**Failure:** `_latest_snapshot = {"threshold_pct": th}` has no `snapshot_type`.  
`_render_mission_tab()` expects `snapshot_type in ("CONFIG", "EVALUATION")`.  
`snapshot.get("snapshot_type")` → `None` → `RuntimeError: snapshot_type must be 'CONFIG' or 'EVALUATION'; got None`

**Result:** App crashes when "New Simulation" is clicked.

---

### B) Run Simulation path (SimulationWorker → _on_simulation_done)

**Location:** `_on_simulation_done()` (line 1427)

**Behavior:** Adapter returns dict without `snapshot_type`. `_on_simulation_done` adds `snapshot_type = "EVALUATION"` and `mission_mode` before `_render_mission_tab()`. Propagation continues.

**Result:** Run Simulation path propagates correctly once `snapshot_type` is added.

---

### C) LIVE mode (EvaluationWorker → _handle_evaluation_result)

**Location:** `_handle_evaluation_result()` (line 1593)

**Behavior:** EvaluationWorker emits dict with `snapshot_type: "EVALUATION"`. `_handle_evaluation_result` builds full snapshot and assigns to `_latest_snapshot`. Propagation continues.

**Result:** LIVE mode propagates correctly.

---

## Summary

| Path | Propagation status |
|------|--------------------|
| Run Simulation (manual/auto) | OK (snapshot_type added in `_on_simulation_done`) |
| LIVE EvaluationWorker | OK |
| **New Simulation** | FAILS — minimal snapshot without `snapshot_type` → RuntimeError in `_render_mission_tab` |

**Root cause for New Simulation failure:** `_on_new_simulation_clicked` sets `_latest_snapshot = {"threshold_pct": th}` and then calls `_render_mission_tab()`, which assumes `snapshot_type in ("CONFIG", "EVALUATION")`.

---

## Debug Print Locations Added

- **Phase 1:** `SimulationWorker.run()` — EVAL TYPE, HITS, P_HIT, CI LOW/HIGH, SAMPLES, IMPACT SHAPE, DECISION
- **Phase 2:** `_on_simulation_done()` — SNAPSHOT AFTER UPDATE, SNAPSHOT.EVALUATION, SNAPSHOT.SYSTEM_MODE
- **Phase 3:** `_render_mission_tab_operator()` — DECISION SOURCE VARIABLE, DECISION VALUE
- **Phase 4:** `_render_mission_tab_operator()` — FIELD NAME/VALUE for HIT %, HITS, Stability, Sample count, CEP50, 95% CI, CI width
- **Phase 5:** `_render_analysis_tab()`, `analysis.py::render()` — ANALYTICAL MODE, IMPACT COUNT
- **Phase 6:** `_start_simulation()`, `_on_simulation_done()` — CONFIG HASH, EVAL HASH, SNAPSHOT HASH
