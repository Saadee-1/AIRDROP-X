# SCYTHE — Claude Code Master Context File
# Stochastic Computed Yield and Terminal Hit Estimator
#
# READ THIS ENTIRE FILE BEFORE TOUCHING ANY CODE.
# Every decision here was made deliberately and is locked.
# Do not refactor, rename, or restructure anything not
# explicitly in your current task.
#
# AT THE END OF EVERY COMPLETED TASK: Update the Phase
# Completion Status section to reflect what was built.
# Mark completed items with CHECK. Add new decisions to
# the relevant section. Do not rewrite sections you did
# not touch.

---

## 1. Project Identity

Full Name: SCYTHE — Stochastic Computed Yield and Terminal Hit Estimator
Type: Semi-autonomous probabilistic UAV payload drop guidance system
Purpose: Precision unpowered payload delivery under real-world
atmospheric uncertainty. Operator-aided. NOT fully autonomous.
Hard Real-Time Target: 50ms end-to-end (target lock to advisory output)
Qt Binding: PySide6 ONLY. Never PyQt5. Never PyQt6.
Language: Python 3.x
UI Framework: PySide6 — QGraphicsView-based tactical map
Window Title: Must read "SCYTHE" not "AIRDROP-X"

---

## 2. Repository Structure

```
E:/AIRDROP-X/
|-- CLAUDE.md                          <- this file
|-- product/
|   |-- aircraft/
|   |   `-- motion_predictor.py        <- physics + uncertainty engine
|   |-- ui/
|   |   |-- widgets/
|   |   |   |-- tactical_map_widget.py <- QGraphicsView subclass (TacticalMapWidget)
|   |   |   |-- status_banner.py       <- StatusBannerWidget + DropStatus + DropReason
|   |   |   `-- hud_overlay.py         <- HUD QWidget overlay
|   |   |-- tabs/
|   |   |   `-- tactical_map_tab.py
|   |   |-- map_transform.py           <- MapTransform — DO NOT TOUCH
|   |   `-- tactical_map_controller.py
|   |-- runtime/
|   |   `-- runtime_loops.py           <- 4 runtime loops
|   `-- uncertainty/
|       `-- unscented_propagation.py   <- UT sigma point engine
|-- src/
|   `-- monte_carlo.py                 <- MC sampling engine
|-- qt_app/
|   |-- main_window.py                 <- main window, tab switching
|   `-- evaluation_worker.py           <- EvaluationWorker 6.6Hz
`-- configs/
    `-- mission_configs.py
```

---

## 3. Coordinate System — LOCKED. DO NOT TOUCH.

- ENU (East-North-Up) throughout. No internal conversions.
- MapTransform.pixels_per_meter is the ONLY zoom variable in the system.
  Do not introduce any other. Do not alias it. Do not copy it.
- MapTransform.apply_to_view handles Y-flip (Qt Y-axis inverted vs ENU).
- Do not modify MapTransform for any reason unless explicitly tasked.

---

## 4. Map Interaction — LOCKED. DO NOT TOUCH.

Input               | Action
--------------------|-------------------------
Ctrl + Scroll       | Zoom (cursor-anchored)
Scroll              | Pan
follow_uav          | True by default

- UAV marker: ItemIgnoresTransformations, fixed 15x14px
- Do not modify zoom or pan logic under any circumstances.

---

## 5. Runtime Architecture

Four runtime loops. Do not add loops. Do not change frequencies.

Loop                  | Frequency | Responsibility
----------------------|-----------|----------------------------------------
TelemetryLoop         | 50 Hz     | Sensor ingestion
GuidanceLoop          | 12 Hz     | Drop advisory, 50ms constraint
BackgroundPlannerLoop | 1 Hz      | Async MC planning (15-20s per cycle)
UIRenderLoop          | 30 Hz     | Display update

Workers:
- SimulationWorker — trajectory simulation
- EvaluationWorker — runs at 6.6Hz

State: Snapshot-driven immutable state.
SystemState uses threading.Lock. Never bypass this.

BackgroundPlannerLoop 15-20s per cycle is a known limitation.
Accepted as async advisory. Do not try to fix this unless tasked.

---

## 6. HUD Overlay — LOCKED

- HUD is a QWidget child of QGraphicsView.viewport() — NOT a scene item.
- Displays: Wind, Drag, Release, Vehicle bars + N compass
- Do not move HUD into the scene under any circumstances.
- Do not modify HUD in tasks that do not explicitly target it.

---

## 7. Complete Mathematics — SCYTHE Physics Engine

These are the implemented and validated math models.
Do not change physics logic without explicit instruction.

### 7.1 Vehicle Motion Model
Constant acceleration model used by MotionPredictor:
  x(t) = x0 + v0*t + 0.5*a*t^2
  v(t) = v0 + a*t
State vector: s = [x, y, z, vx, vy, vz]
Acceleration clamped: ||a|| <= 15 m/s^2

### 7.2 Relative Wind Velocity
  v_rel = v - v_wind
  v = ||v_rel||

### 7.3 Aerodynamic Drag
  F_d = -0.5 * rho * Cd * A * v * v_rel
  a_d = F_d / m
  rho = air density, Cd = drag coeff, A = reference area, v = speed

### 7.4 Gravity
  g = [0, 0, -9.81]
  a_total = g + a_d

### 7.5 Atmospheric Density — Exponential Model
  rho(z) = rho0 * exp(-z / H)
  rho0 = 1.225 kg/m^3
  H = 8500 m
  z_agl = z - z_ground

### 7.6 Rotational Payload Dynamics
  theta(t+1) = theta(t) + omega(t)*dt
  omega(t+1) = omega(t) - lambda*omega(t)*dt + sigma_w*sqrt(dt)*N(0,1)
  Cd_eff = Cd * (1 + k * |sin(theta)|)
  Cd_eff clamped: 0.5*Cd <= Cd_eff <= 2*Cd

### 7.7 Trajectory Integration — RK2
  k1 = f(x_n)
  k2 = f(x_n + k1*dt)
  x_{n+1} = x_n + (dt/2)*(k1 + k2)

### 7.8 Impact Detection — Ground Intersection
  alpha = (z_prev - z_ground) / (z_prev - z_curr)
  p_impact = p_prev + alpha*(p_curr - p_prev)

### 7.9 Unscented Transform — Sigma Point Propagation
  Sigma points: 2n+1
  x0 = mu
  x_i = mu + sqrt((n+lambda)*Sigma)_i
  x_{i+n} = mu - sqrt((n+lambda)*Sigma)_i
  Mean: mu = sum(W_i * x_i)
  Covariance: Sigma = sum(W_i * (x_i - mu)(x_i - mu)^T)

  Uncertainty vector (n=5, sigma points=11):
  u = [wind_x_bias, wind_y_bias, release_x, release_y, velocity_bias]

  Sigma point propagations currently sequential.
  Optimization (batch all 11): PENDING — do not implement until profiled.

### 7.10 Impact Distribution
  Modeled as Gaussian: mean=mu_impact, covariance=Sigma_impact

### 7.11 Hit Probability — Mahalanobis Distance
  d^2 = (mu - t)^T * Sigma^{-1} * (mu - t)
  P_hit = exp(-0.5 * d^2)

### 7.12 Finite Target Radius Correction
  sigma^2 = u^T * Sigma * u
  P_hit = exp(-0.5*d^2) * (1 - exp(-r^2 / (2*sigma^2)))

### 7.13 CEP50 — Circular Error Probable
  CEP50 = 0.5887 * sqrt(lambda1 + lambda2)
  lambda1, lambda2 = eigenvalues of impact covariance matrix

### 7.14 Monte Carlo Hit Probability
  P_hit = (1/N) * sum(I(||x_i - t|| < r))
  Wind: AR(1) correlated, Gaussian uncertainty
  CI: Wilson Confidence Interval
  Current N=1000. Adaptive MC upgrade pending (see Section 10).

---

## 8. Tab Navigation Reference

Index | Tab Name
------|--------------------
0     | Tactical Map
1     | Telemetry
2     | Mission Configuration
3     | Analysis
4     | System Status

Mission Configuration = index 2
Tactical Map = index 0

---

## 9. Phase Completion Status

### Phase 0 — COMPLETE
- [x] ENU coordinate system locked
- [x] MapTransform.pixels_per_meter as single zoom source of truth
- [x] Ctrl+scroll zoom cursor-anchored, scroll=pan
- [x] Dynamic grid spacing based on zoom level
- [x] HUD overlay as QWidget on viewport (not scene)
- [x] UAV marker ItemIgnoresTransformations, fixed 15x14px
- [x] follow_uav = True default
- [x] 5 critical bugs fixed: print statement, hard-coded physics,
      MotionPredictor wired, envelope_dirty flag, _busy try/finally
- [x] FPS counter removed
- [x] Camera feed reverted — not yet implemented

### Phase 1 — Operator Workflow — ACTIVE

#### Phase 1.1 — COMPLETE
- [x] StatusBannerWidget in product/ui/widgets/status_banner.py
- [x] DropStatus IntEnum: NO_DROP=0, APPROACH_CORRIDOR=1,
      IN_DROP_ZONE=2, DROP_NOW=3
- [x] Banner integrated into TacticalMapWidget as viewport child
- [x] _reposition_status_banner() for dynamic centering on resize
- [x] set_status() calls update() not repaint()
- [x] WA_TransparentForMouseEvents set (removed in 1.1b)
- [x] Label: background-color transparent, no text-transform
- [x] Compile-verified. Runtime-verified.

#### Phase 1.1b — CURRENT TASK
Context-aware and interactive status banner.

Requirements:
- Add DropReason IntEnum: NONE=0, MISSION_PARAMS_NOT_SET=1,
  UAV_TOO_FAR=2, WIND_EXCEEDED=3
- Advisory text line below main banner text (9pt, state-dependent)
- Banner height: 64px (was 44px)
- Remove WA_TransparentForMouseEvents
- mousePressEvent: emit navigate_to_tab(2) ONLY on
  MISSION_PARAMS_NOT_SET, do nothing on all other states
- wheelEvent: always call super() to pass scroll through
- Advisory blink on MISSION_PARAMS_NOT_SET: QTimer 1s toggle setVisible()
- Wire navigate_to_tab Signal to main window tab switcher
- Auto-redirect to Tactical Map after Commit Configuration button

Advisory text:
State + Reason               | Advisory Text                                   | Color
-----------------------------|------------------------------------------------|--------
NO_DROP+MISSION_PARAMS       | "Mission parameters not set — click to configure"| #FFDD00
NO_DROP+UAV_TOO_FAR          | "Outside drop corridor — adjust heading"         | white
NO_DROP+WIND_EXCEEDED        | "Wind envelope exceeded — hold position"         | white
NO_DROP+NONE                 | ""                                               | white
APPROACH_CORRIDOR            | "Intercept heading — maintain altitude"          | white
IN_DROP_ZONE                 | "Confirm release conditions"                     | white
DROP_NOW                     | "Release payload immediately"                    | black

#### Phase 1.2 — QUEUED
Guidance arrow — direction to fly toward drop zone.

#### Phase 1.3 — QUEUED
Impact ellipses live update — 68% green, 95% amber.
Pulls from UT covariance. CEP50 from eigenvalues.

#### Phase 1.4 — QUEUED
Opportunity corridor visualization on map.

### Phase 2 — Terrain — QUEUED
- 2.1 DEM flat placeholder georeferenced
- 2.2 SRTM tile loader
- 2.3 Physics engine ground_z from DEM

### Phase 3 — Sensor Layer — QUEUED
- 3.1 Camera feed background (Option A — raw overlay, no georef)
- 3.2 Opportunity Explorer visualization
- 3.3 Camera-to-world coordinate bridge

### Phase 4 — Polish — QUEUED
- 4.1 Wind indicator on map
- 4.2 CEP overlay
- 4.3 Moving target support
- 4.4 Terrain shading

---

## 10. Planned Physics Upgrades — Do Not Implement Until Explicitly Tasked

### Plan A — Priority Upgrades
1. Adaptive Monte Carlo: start N=200, increase while CI_width > threshold.
   Result: 200-400 samples vs 1000, ~3x faster. Implement BEFORE UT upgrade.

2. Gust Model: Dryden or Von Karman turbulence.
   Adds gust_x(t), gust_y(t), gust_z(t) to wind.

3. Rotational Coupling: full omega_x, omega_y, omega_z state,
   drag torque, orientation-dependent Cd.

4. Actuator Delay: t_delay = 0.2-0.5s. Significant impact shift at speed.

5. Release Corridor Solver: min/max range + optimal release point.

### Plan B — Advanced / Research Grade
6.  State-Dependent Uncertainty: sigma_wind(z) = a + b*z
7.  Gaussian Mixture Models: multi-modal wind distributions
8.  Particle Filter: sequential MC for live wind estimation
9.  Bayesian Hit Probability: Bayesian posterior replaces hits/N
10. Risk-Aware Advisory: expected loss over P_hit threshold
11. Moving Target Support: target velocity + future intercept distribution
12. Real-Time UAV Integration: PX4/ArduPilot MAVLink, GPS, IMU
13. Optimal Release Solver: argmax P_hit via gradient/CMA-ES
14. Multi-Payload Planning: coverage probability, multiple drops
15. DEM Terrain Interaction: SRTM elevation for impact prediction
16. Swarm Drop Planning: multi-UAV coordinated drop computation

---

## 11. Known Issues and Pending Fixes

- Window title reads "AIRDROP-X" — change to "SCYTHE" in main_window.py
- System Status ENGINE IDENTITY block says "AIRDROP-X v1.1" — update to "SCYTHE v1.1"
- get_drop_status() in motion_predictor.py returns DropStatus.NO_DROP placeholder.
  TODO comment marks the real logic insertion point.
- Sigma point propagations sequential — batch optimization pending profiling

---

## 12. Operator Workflow — How SCYTHE Is Used

1. UAV in flight or on ground, params not set
2. System shows NO_DROP + reason code
3. If MISSION_PARAMS_NOT_SET: operator clicks banner, routed to Mission Config tab
4. Operator sets: payload category, mass, Cd, area, target coords,
   evaluation depth, decision doctrine, simulation fidelity
5. Operator clicks "Commit Configuration"
6. System auto-redirects to Tactical Map tab
7. BackgroundPlannerLoop warms MC engine
8. Operator spots target on camera, clicks crosshair
9. Pixel converts to GPS coordinate
10. Auto-triggers simulation
11. Advisory output in <50ms hard target (70ms minimum acceptable)
12. Banner: NO_DROP -> APPROACH_CORRIDOR -> IN_DROP_ZONE -> DROP_NOW

Scope: static and slow-moving targets only.
NOT autonomous. Operator always in the loop.
Moving target support is next major upgrade after Phase 1.

---

## 13. Immutable Rules for Every Task

1.  PySide6 only. If you see PyQt5 anywhere, stop and flag it immediately.
2.  Never add to QGraphicsScene what belongs on the viewport.
3.  Never use self.repaint() — always self.update()
4.  Never hardcode physics values — all come from MotionPredictor
5.  Never change MapTransform, zoom logic, or pan logic unless explicitly tasked
6.  Run python -m py_compile on every modified file before reporting done
7.  Output change summary: every file modified, what changed, in which method
8.  Never refactor what is not broken
9.  Output complete files only when explicitly asked — changed blocks otherwise
10. One phase at a time — do not implement future phases speculatively
11. When in doubt: stop, list options with tradeoffs, do not proceed unilaterally
12. Never remove or alter math in Section 7 unless a physics upgrade was
    explicitly completed and verified by the supervising engineer

---

## 14. Self-Update Instructions

At the end of every completed task, update this file as follows:

1. Section 9 — Phase Completion Status:
   - Mark newly completed items [x]
   - Move CURRENT TASK label to next phase
   - Add architectural decisions made during the task

2. Section 11 — Known Issues:
   - Remove issues that were fixed
   - Add new issues discovered

3. Other sections:
   - Update facts that changed (class names, file paths, method names)
   - Do NOT rewrite sections you did not touch
   - Do NOT change Section 7 math unless a physics upgrade was completed

4. Commit message format: "chore: update CLAUDE.md post-[phase-name]"
