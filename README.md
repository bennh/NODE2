# NODE2_Project_1: Constrained Nonlinear Least Squares via Direct Multiple Shooting

## Overview

This project implements a full numerical optimization pipeline for solving parameter estimation problems governed by ordinary differential equations (ODEs), using the **Direct Multiple Shooting** method combined with **Constrained Nonlinear Least Squares (CNLLS)** solvers.

Two dynamic systems are used to demonstrate and validate the approach:

* **Pyridine Reaction System**: a real-world 7-state chemical kinetics model.
* **Notorious Test Example**: a synthetic benchmark from Bock et al. (2007), known for being numerically challenging.

The implementation supports both **IPOPT** (for general nonlinear programs with inequality constraints) and **Gauss-Newton with log-parameter transformation**, suitable for positive parameter estimation.

---

## File Structure

```text
NODE2_1/
├── multiple_shooting.py     # Direct multiple shooting problem formulation
├── cnlls_solver.py          # CNLLS solvers: IPOPT, Gauss-Newton, Gauss-Newton-logparam
├── pyridine.py              # Pyridine system dynamics, integrator, true values
├── notorious.py             # Notorious test example model and simulator
├── utils.py                 # Utility functions: noise, error metrics, visualization
├── demo.ipynb               # Full demonstration of parameter estimation on both systems
```

---

## Key Modules

### 1. `multiple_shooting.py`

Sets up the direct multiple shooting discretization:

* Shooting states `s_0, ..., s_{N-1}`
* Parameter vector `p`
* Matching constraints `F2`
* Optional inequality constraints `F3` (nonnegativity on states/params)

### 2. `cnlls_solver.py`

Implements three CNLLS solvers:

* `solve_cnlls_ipopt`: full CNLLS via IPOPT (supports `F1`, `F2`, `F3`)
* `solve_cnlls_gauss_newton`: GN with equality constraints only
* `solve_cnlls_gauss_newton_logparam`: GN with log-param reparametrization

### 3. `pyridine.py`

Defines:

* 7-variable ODE model for pyridine kinetics
* Initial condition and parameter ground truth
* CasADi integrator per time step

### 4. `notorious.py`

Implements:

* Bock's Section 6 test problem with known analytical solution
* Integrator setup using `cvodes`
* Measurement simulator with time-varying ODE

### 5. `utils.py`

Includes:

* `generate_measurements`, `add_noise`
* `evaluate_fit`: RMSE / MAE / R2 + plot
* `print_formatted_params`

### 6. `demo.ipynb`

Notebook demo:

* Parameter estimation for both systems
* Comparison of IPOPT vs GN-logparam solvers
* Visual + quantitative fit assessment

---

## How to Run

### Requirements

* Python 3.8+
* CasADi (tested with v3.6.4)
* NumPy, Matplotlib, scikit-learn

### Execute

1. Open `demo.ipynb` in JupyterLab or VS Code
2. Run all cells in sequence
3. Observe console output and plots

---

## Summary

* Demonstrated CNLLS + Multiple Shooting on real (Pyridine) and synthetic (Notorious) systems
* Verified two solvers: IPOPT (with `F3`) and GN-logparam (without explicit constraints)
* All modules are cleanly modularized and reusable
* Ideal baseline for dynamic system parameter estimation in chemical/biological modeling
