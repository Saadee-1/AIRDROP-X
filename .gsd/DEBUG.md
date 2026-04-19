# Debug Session: Preview Browser Not Launching

## Symptom

The Streamlit web preview (`app.py`) does not launch when run.

**When:** On `streamlit run app.py`
**Expected:** Browser opens with SCYTHE web UI at localhost:8501
**Actual:** `ModuleNotFoundError: No module named 'streamlit'`

## Evidence

### E1: Streamlit Not Installed

```
ModuleNotFoundError: No module named 'streamlit'
```

Despite `requirements.txt` listing `streamlit>=1.28.0`, it was never `pip install`ed.

### E2: API Audit — All Clear

| API Call in `app.py` | Module | Match |
|---|---|---|
| `plots.plot_impact_dispersion(ax, pts, pos, rad, cep)` | `product.ui.plots` | ✓ |
| `decision_logic.evaluate_drop_decision(P, thresh)` | `src.decision_logic` | ✓ |
| `get_impact_points_and_metrics(ms, seed)` | `product.guidance.advisory_layer` | ✓ |
| `evaluate_advisory(ms, mode, seed=)` | `product.guidance.advisory_layer` | ✓ |
| `AdvisoryResult` attrs (7 checked) | `product.guidance.advisory_layer` | ✓ |

## Hypotheses

| # | Hypothesis | Likelihood | Status |
|---|------------|------------|--------|
| 1 | Streamlit not installed | 95% | **CONFIRMED** |
| 2 | API mismatch in app.py | 30% | ELIMINATED |
| 3 | Missing plots module | 20% | ELIMINATED |

## Resolution

**Root Cause:** `streamlit` package not installed in the Python environment despite being listed in `requirements.txt`.

**Fix:** `pip install streamlit>=1.28.0` → installed v1.54.0 with all dependencies.

**Verified:**

- `python -c "import streamlit"` → OK
- `streamlit run app.py --server.headless true` → Server started on port 8501
- Health endpoint `/_stcore/health` → `200 ok`
- No errors in server log
- All theme constants (`BG_MAIN`, `FONT_FAMILY`, `BUTTON_HOVER`) resolve correctly

**Regression Check:** Original `python main.py` (matplotlib desktop UI) unaffected.
