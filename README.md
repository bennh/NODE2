# NODE2 Project: Constrained Nonlinear Least Squares via Direct Multiple Shooting

## Overview

This project implements a full numerical optimization pipeline for solving parameter estimation problems governed by ordinary differential equations (ODEs), using the **Direct Multiple Shooting** method combined with **Constrained Nonlinear Least Squares (CNLLS)** solvers.

Two dynamic systems are used to demonstrate and validate the approach:

- **Pyridine Reaction System**: a real-world 7-state chemical kinetics model.
- **Notorious Test Example**: a synthetic benchmark from Bock et al. (2007), known for being numerically challenging.

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
