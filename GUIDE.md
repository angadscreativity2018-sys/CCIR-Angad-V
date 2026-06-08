# Orbit Simulator — User Guide

## Overview

Simulates a two-body gravitational system (e.g. a binary star) where one body undergoes a sudden velocity kick — modelling a supernova impulse. It tracks orbits before and after the kick, computes eccentricities, and produces plots.

**Files:**
- [main.py](main.py) — configuration, initial conditions, integration loop
- [step.py](step.py) — Velocity Verlet integrator (one timestep)
- [plot.py](plot.py) — all plotting functions

---

## Unit System

| Quantity  | 1 sim unit equals        |
|-----------|--------------------------|
| Mass      | 1 M☉ = 1.989 × 10³⁰ kg  |
| Length    | 10⁹ m                    |
| Time      | 1 year = 3.15 × 10⁷ s   |
| Velocity  | 31.75 m/s                |
| G         | 1.321 × 10⁸ (sim units)  |

---

## Configuration Variables (`main.py`)

### Display / Frame flags

| Variable | Default | Effect |
|---|---|---|
| `CENTER_OF_MASS_FRAME` | `True` | Shifts initial velocities so the centre of mass is stationary. Prevents the pair drifting off screen. |
| `PROJECT_TO_ORBITAL_PLANE` | `True` | Projects positions onto the pre-impulse orbital plane for a clean 2D view. Set `False` for raw 3D lab-frame view. |

> Both flags are most meaningful when running a **single simulation** (`ITERATIONS = 1`).

---

## Single Simulation

Set `ITERATIONS = 1`, `SHOW_PLOT = True`, `SHOW_MULTI_PLOT = False`, `SHOW_ECC_STATS = False`.

```python
SHOW_PLOT       = True
SHOW_MULTI_PLOT = False
SHOW_ECC_STATS  = False
ITERATIONS      = 1
```

### What you can tune

**Initial conditions** (inside `main()`):

| Variable | Default | Meaning |
|---|---|---|
| `mass1` | derived | Pre-kick mass of body 1; derived each run as `POST_IMPULSE_MASS1 + drawn ejecta` |
| `mass2` | `15` | Mass of body 2, in M☉ |
| `seperation` | from sweep | Initial separation (sim length units). Set via `INITIAL_SEPARATION` (or back-calculated from `INITIAL_PERIOD` when `USE_PERIOD = True`) |
| `e` | `0` | Pre-impulse eccentricity (0 = circular). Affects the initial relative velocity calculation |
| `dt` | `0.007` | Timestep size. Smaller = more accurate, slower |
| `total_time` | `20` | Total simulation duration in sim-years |
| `impulse_time` | `15` | Base time of the kick (sim-years). When `RANDOM_PHASE = True`, a random fraction of one period is subtracted so each run kicks at a different orbital phase |

**Impulse (supernova kick) parameters:**

| Variable | Default | Meaning |
|---|---|---|
| `APPLY_IMPULSE` | `True` | Turn on/off the kick entirely |
| `RANDOM_PHASE` | `True` | Subtract a random 0–1 orbital periods from the base impulse time each run |
| **Kick speed** | | |
| `IMPULSE_DIST` | `"normal"` | Distribution for kick speed: `"beta"` or `"normal"` |
| `IMPULSE_ALPHA` | `3.05` | **Beta only** — shape α |
| `IMPULSE_BETA` | `14.6` | **Beta only** — shape β |
| `IMPULSE_SCALE` | `17730` | **Beta only** — speed = `Beta(α,β) × SCALE` |
| `IMPULSE_MEAN` | `0` | **Normal only** — mean kick speed (sim velocity units) |
| `IMPULSE_SIGMA` | `8346` | **Normal only** — sigma; `abs()` applied so speed is always positive |
| **Mass** | | |
| `POST_IMPULSE_MASS1` | `1.4` | Remnant mass of body 1 after kick (M☉) — e.g. a neutron star |
| `MASS_LOSS_MEAN` | `2.6` | Mean ejecta mass (M☉). Initial mass = `POST_IMPULSE_MASS1 + drawn loss` |
| `MASS_LOSS_SIGMA` | `0.5` | Sigma of ejecta draw; set `0` for a fixed loss equal to `MASS_LOSS_MEAN` |

**What the kick does:** draws an ejecta mass from `|Normal(MASS_LOSS_MEAN, MASS_LOSS_SIGMA)|` and sets the pre-kick mass as `POST_IMPULSE_MASS1 + ejecta`. At the moment of the kick, a speed is sampled from the chosen distribution and applied in a random 3D direction; body 1's mass is then set to `POST_IMPULSE_MASS1`.

**Phase randomisation:** the kick is nominally at `t = 15` years. With `RANDOM_PHASE = True`, the actual kick time becomes `15 - U(0,1) × T` where `T` is the pre-impulse orbital period, giving uniform coverage of all orbital phases across runs.

### Single-run plot

The plot shows the orbital-plane (2D) or 3D lab-frame view with:
- **Blue dots** — body 1 pre-kick
- **Red dots** — body 2 pre-kick
- **Green dots** — body 1 post-kick
- **Orange dots** — body 2 post-kick
- **Yellow star** — location where the kick happened
- **Time slider** — scrub through the simulation to watch both bodies move


---

## Large Number of Simulations

Set `SHOW_PLOT = False` and increase `ITERATIONS`. Use `SHOW_MULTI_PLOT` and/or `SHOW_ECC_STATS` to get statistical output.

```python
SHOW_PLOT       = False
SHOW_MULTI_PLOT = True
SHOW_ECC_STATS  = True
ITERATIONS      = 200   # runs per separation group
```

### Additional variables for batch runs

| Variable | Default | Meaning |
|---|---|---|
| `ITERATIONS` | `200` | Number of independent runs **per group** |
| `SEPERATION_GROUPS` | `8` | Number of groups in the sweep |
| `MULTI_PLOT_WINDOW` | `500` | Steps shown either side of the impulse in the CoM trajectory plot |
| `USE_PERIOD` | `False` | `False` → sweep over separations; `True` → sweep over orbital periods (separation is back-calculated via Kepler's 3rd law) |
| `INITIAL_SEPARATION` | `600` | Starting separation for sweep (sim length units); used when `USE_PERIOD = False` |
| `SEPARATION_SCALE` | `1.5` | Geometric scale between separation groups: group j gets `INITIAL_SEPARATION × SCALE^j` |
| `LOG10_PERIOD_MEAN` | `0.0` | Mean of log₁₀(period) distribution (e.g. `0` → 1 yr) |
| `LOG10_PERIOD_SIGMA` | `0.5` | Sigma of log₁₀(period) distribution |

When `USE_PERIOD = True`, each run independently draws `log₁₀(T) ~ Normal(LOG10_PERIOD_MEAN, LOG10_PERIOD_SIGMA)`, converts to period via `T = 10^x`, then to separation via **Kepler's 3rd law**: `a = (G·M·T² / 4π²)^(1/3)`. `SEPERATION_GROUPS` is ignored — just `ITERATIONS` runs total. When `USE_PERIOD = False`, the separation sweep uses `INITIAL_SEPARATION × SEPARATION_SCALE^j` across `SEPERATION_GROUPS` groups.

### Plot 1 — Centre-of-mass trajectories (`SHOW_MULTI_PLOT`)

3D plot of every run's CoM path around the impulse moment. Pre-kick segment is grey (identical for all runs at a given separation). Post-kick colour encodes the resulting orbit type:

| Colour | Orbit type | Eccentricity |
|---|---|---|
| Royal blue | Circular | e < 0.1 |
| Gold | Elliptical | 0.1 ≤ e < 1 |
| Crimson | Hyperbolic / unbound | e ≥ 1 |

### Plot 2 — Orbit-type statistics (`SHOW_ECC_STATS`)

Stacked bar chart showing the percentage of circular / elliptical / hyperbolic outcomes for each initial separation value. Useful for seeing how separation affects the probability of the binary surviving the kick as a bound system.

---

## How the Integrator Works (`step.py`)

Uses the **Velocity Verlet** algorithm:

1. Compute gravitational force from current positions.
2. Half-update velocity, full-update positions.
3. Recompute force at new positions.
4. Complete velocity update with average of old and new accelerations.

This is symplectic and conserves energy much better than simple Euler integration over long runs.

---

## Quick-start recipes

**Watch one orbit with the kick:**
```python
SHOW_PLOT, SHOW_MULTI_PLOT, SHOW_ECC_STATS = True, False, False
ITERATIONS = 1
# then run: python main.py
```

**Statistical sweep over separations:**
```python
SHOW_PLOT, SHOW_MULTI_PLOT, SHOW_ECC_STATS = False, True, True
ITERATIONS      = 200
SEPERATION_GROUPS = 8
# then run: python main.py
```

**Pure orbital mechanics (no kick):**
```python
APPLY_IMPULSE = False
SHOW_PLOT     = True
ITERATIONS    = 1
```
