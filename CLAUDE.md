# SCYTHE — Claude Code Session Context
> Read this entire file before touching any code. Every architectural
> decision here was made deliberately. Do not refactor, rename, or
> restructure anything not explicitly in your current task.

---

## 1. Project Identity

**Full Name:** SCYTHE (Stochastic Computed Yield and Terminal Hit Estimator)
**Type:** Semi-autonomous probabilistic UAV payload drop guidance system
**Purpose:** Precision unpowered payload delivery under real-world
atmospheric uncertainty. Operator-aided, not fully autonomous.
**Hard Real-Time Target:** 50ms update constraint on guidance pipeline
**Qt Binding:** PySide6 ONLY — never PyQt5, never PyQt6
**Language:** Python 3.x
**UI Framework:** PySide6 (QGraphicsView-based tactical map)

---

## 2. Architecture Overview

```
product/
├── aircraft/
│   └── motion_predictor.py       # Physics + uncertainty engine
├── ui/
│   └── widgets/
│       ├── tactical_map_widget.py  # QGraphicsView subclass (TacticalMapWidget)
│       ├── status_banner.py        # StatusBannerWidget + DropStatus enum
│       └── hud_overlay.py          # HUD QWidget overlay (Wind/Drag/Release/Vehicle)
```

---

## 3. Coordinate System — LOCKED

- **ENU (East-North-Up)** coordinate system. Never change this.
- `MapTransform.pixels_per_meter` is the **single source of truth** for zoom.
  Do not introduce any other zoom variable anywhere.
- `MapTransform.apply_to_view` handles Y-flip (Qt Y-axis is inverted vs ENU).
- Do not touch `MapTransform` unless explicitly instructed.

---

## 4. Map Interaction — LOCKED

| Input | Action |
|---|---|
| Ctrl + Scroll | Zoom (cursor-anchored) |
| Scroll | Pan |
| follow_uav | True by default |

- UAV marker uses `ItemIgnoresTransformations` — fixed 15x14px pixel size.
- Do not change zoom or pan logic under any circumstances.

---

## 5. HUD Overlay — LOCKED

- HUD is a `QWidget` child of `QGraphicsView.viewport()` — NOT a scene item.
- Displays: Wind, Drag, Release, Vehicle bars + N↑ compass.
- Do not move HUD into the scene. Do not convert it to a scene item.
- Do not modify HUD in any task that does not explicitly target it.

---

## 6. Phase Completion Status

### Phase 0 — COMPLETE
- ENU coordinate system locked
- MapTransform.pixels_per_meter as single zoom source of truth
- Ctrl+scroll zoom (cursor anchored), scroll pan
- Dynamic grid spacing based on zoom level
- HUD overlay as QWidget on viewport (not scene)
- UAV marker with ItemIgnoresTransformations, fixed 15x14px
- follow_uav = True default
- 5 critical bugs fixed: print statement, hard-coded physics,
  MotionPredictor wired, envelope_dirty flag, _busy try/finally
- FPS counter removed
- Camera feed integration reverted (not yet implemented)

### Phase 1.1 — COMPLETE
- `StatusBannerWidget` implemented in `product/ui/widgets/status_banner.py`
- `DropStatus` IntEnum with 4 states: NO_DROP, APPROACH_CORRIDOR,
  IN_DROP_ZONE, DROP_NOW
- Banner integrated into `TacticalMapWidget` as viewport child widget
- `_reposition_status_banner()` method handles dynamic centering on resize
- Compile-verified. Runtime-verified. Banner displays correctly.

### Phase 1.1b — IN PROGRESS (next task)
See Section 8 below.

### Phase 1.2 — QUEUED
- Guidance arrow showing operator which direction to fly

### Phase 1.3 — QUEUED
- Impact ellipses live update (68% green, 95% amber)

### Phase 1.4 — QUEUED
- Opportunity corridor visualization on map

---

## 7. StatusBannerWidget — Critical Constraints

**File:** `product/ui/widgets/status_banner.py`

### Color Mapping
| State | Background | Text Color | Banner Text |
|---|---|---|---|
| NO_DROP | #CC2200 | white | "NO DROP" |
| APPROACH_CORRIDOR | #FF8C00 | white | "APPROACH CORRIDOR" |
| IN_DROP_ZONE | #00AA44 | white | "IN DROP ZONE" |
| DROP_NOW | #00FF66 | black | "DROP NOW" |

### Critical Implementation Notes
- `WA_TransparentForMouseEvents` was set initially but must be
  **removed** for Phase 1.1b clickable banner behavior.
- After removing it, scroll and pan events MUST be manually passed
  through via `mousePressEvent` override — only click events are
  consumed by the banner, all other events are forwarded to the
  map viewport. Failure to do this breaks pan/zoom.
- `set_status()` calls `self.update()` not `self.repaint()`
- Label stylesheet must include `background-color: transparent;`
- Do NOT use `text-transform: uppercase` in Qt stylesheets —
  it is silently ignored. Text strings are already uppercase.
- Alpha on background color is set to 200 (semi-transparent).

---

## 8. Phase 1.1b — Current Task Specification

### Goal
Make the status banner context-aware and interactive.

### Behavior Requirements

#### Sub-states for NO_DROP
`DropStatus.NO_DROP` must carry a reason code. Add a separate
`DropReason` enum:

```python
class DropReason(IntEnum):
    NONE = 0                    # Default, used for non-NO_DROP states
    MISSION_PARAMS_NOT_SET = 1  # Payload/mission params not configured
    UAV_TOO_FAR = 2             # UAV outside drop corridor
    WIND_EXCEEDED = 3           # Wind envelope exceeded
```

#### Advisory Text Layer
Below the main banner text, display a smaller advisory line:

| State + Reason | Banner Text | Advisory Text |
|---|---|---|
| NO_DROP + MISSION_PARAMS_NOT_SET | NO DROP | "Mission parameters not configured — tap to set up" |
| NO_DROP + UAV_TOO_FAR | NO DROP | "Outside drop corridor — adjust heading" |
| NO_DROP + WIND_EXCEEDED | NO DROP | "Wind envelope exceeded — hold position" |
| APPROACH_CORRIDOR | APPROACH CORRIDOR | "Intercept heading — maintain altitude" |
| IN_DROP_ZONE | IN DROP ZONE | "Confirm release conditions" |
| DROP_NOW | DROP NOW | "Release payload immediately" |

#### Clickable Banner (MISSION_PARAMS_NOT_SET only)
- When state is `NO_DROP + MISSION_PARAMS_NOT_SET`, banner is clickable.
- On click: emit a Qt signal `navigate_to_tab = Signal(int)` with the
  index of the Mission Configuration tab.
- Parent (main window) connects this signal to switch the active tab.
- On all other states: banner is NOT clickable (no signal emitted).
- `WA_TransparentForMouseEvents` must be removed.
- In `mousePressEvent`: if state is MISSION_PARAMS_NOT_SET, emit signal.
  For all other events (scroll, etc.), call `super()` to pass through.

#### Auto-redirect After Mission Commit
- When user commits mission parameters (saves/confirms in Mission
  Configuration tab), main window switches back to Tactical Map tab
  and calls `status_banner.set_status(DropStatus.NO_DROP,
  reason=DropReason.UAV_TOO_FAR)` or appropriate next state.
- This logic lives in the main window controller, NOT in the banner.

#### Visual Change for Clickable State
- When `MISSION_PARAMS_NOT_SET`: add a subtle pulsing border or
  underline to advisory text to hint at interactivity.
- Do NOT add full animation loops. A simple `QTimer` toggling
  border visibility at 1s interval is acceptable.

### API Change to `set_status()`
```python
def set_status(self, status: DropStatus,
               reason: DropReason = DropReason.NONE):
```
Both parameters required in the interface going forward.

---

## 9. Rules for Every Claude Code Task on SCYTHE

1. **Never mix Qt bindings.** PySide6 only. If you see a PyQt5 import
   anywhere, flag it and stop — do not proceed until resolved.

2. **Never add to QGraphicsScene** what belongs on the viewport.
   Banner, HUD, overlays = viewport children. Always.

3. **Never use `self.repaint()`** — always `self.update()`.

4. **Never hardcode physics values.** All physics parameters come
   from MotionPredictor. No magic numbers in UI code.

5. **Never change MapTransform, zoom logic, or pan logic** unless
   the task explicitly targets them.

6. **Always run `python -m py_compile`** on every modified file
   before reporting completion.

7. **Always output a change summary** listing every file modified
   and exactly what changed. Never just say "done."

8. **Never refactor what isn't broken.** SCYTHE has locked
   architecture. Your job is to add, not reorganize.

9. **Output complete files only when explicitly asked.** Otherwise
   output changed blocks only to keep the audit fast.

10. **One task at a time.** Do not implement Phase 1.2 while working
    on Phase 1.1b. Scope is defined per session.

---

## 10. Tabs / Navigation Reference

The main window has 5 tabs in this order (0-indexed):

| Index | Tab Name |
|---|---|
| 0 | Tactical Map |
| 1 | Telemetry |
| 2 | Mission Configuration |
| 3 | Analysis |
| 4 | System Status |

Mission Configuration tab index = **2**.
Tactical Map tab index = **0**.

---

## 11. What Is NOT Built Yet

- Camera feed integration (reverted in Phase 0, not scheduled yet)
- Guidance arrow (Phase 1.2)
- Impact ellipses (Phase 1.3)
- Opportunity corridor (Phase 1.4)
- Real drop status logic in `get_drop_status()` — currently returns
  `DropStatus.NO_DROP` as a placeholder stub with a TODO comment.
- Any networking or MAVLink integration
- Any data logging or export

Do not build any of the above unless explicitly instructed.

---

## 12. Contact / Supervision

This codebase is supervised. Every output from Claude Code is
audited before being run. Do not assume autonomy on architectural
decisions. When in doubt, stop and list your options with tradeoffs
rather than picking one and proceeding.
