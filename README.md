# Parameter Estimation via Direct Multiple Shooting and CNLLS

This repository implements a parameter estimation framework using **Direct Multiple Shooting** and **Generalized Gauss-Newton methods** based on the work of Bock et al. (2007). It was developed for Programming Project 1 of the course _Numerical Optimization with Differential Equations II_ (Summer Semester 2025, Heidelberg University).

The framework is built using Python and CasADi and demonstrates its applicability through two systems:
- The **Notorious test example** (synthetic benchmark)
- The **Pyridine catalytic reaction system** (realistic chemical model)

---

## Project Structure

```bash
NODE2_1/
│
├── multiple_shooting.py        # Milestone 1: Direct multiple shooting method
├── cnlls_solver.py             # Milestone 2: CNLLS solvers (IPOPT, Gauss-Newton)
│
├── pyridine.py                 # Milestone 3: ODE model and data generation for Pyridine system
├── notorious.py                # Section 6 example: Notorious ODE benchmark
├── utils.py                    # Utilities for noise, plotting, measurement simulation
│
├── demo.ipynb                  # Notebook for full Pyridine parameter estimation and Notorious test case demo
│
