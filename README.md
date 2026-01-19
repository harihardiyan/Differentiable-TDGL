# Differentiable TDGL 2D Framework for Vortex Transport and Geometry Optimization
[![JAX](https://img.shields.io/badge/Framework-JAX-red.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

***This repository provides a differentiable implementation of the two‑dimensional Time‑Dependent Ginzburg–Landau (TDGL) equations using JAX.  
The framework is intended for exploratory studies of vortex transport and geometry‑based inverse design, with an emphasis on clarity and reproducibility.

The project should be viewed as a **computational methods resource** and a **proof‑of‑concept demonstration**, rather than a quantitative device study.

---

## 1. Overview

The code implements:

- Complex order parameter \( \psi \)
- Vector potential \( \mathbf{A} \)
- Electrochemical potential \( \mu \)
- Gauge‑covariant derivatives
- Magnetic field via \( \nabla \times \mathbf{A} \)
- Neumann boundary conditions
- Geometry‑parametric ratchet channels
- Vortex drift measurement
- Gradient‑based optimization using JAX autodiff

The framework enables end‑to‑end differentiation of TDGL dynamics with respect to geometry parameters.

---

## 2. Scope and Intended Use

This repository is suitable for:

- Researchers exploring differentiable physics simulations  
- Rapid prototyping of vortex‑based device concepts  
- Studying feasibility of geometry optimization  
- Educational demonstrations of TDGL dynamics  

The implementation prioritizes **clarity** over large‑scale performance.

---

## 3. Demonstrated Capabilities

The included example scripts show:

- Construction of a ratchet‑like channel geometry  
- TDGL evolution under \( +J_{\text{ext}} \) and \( -J_{\text{ext}} \)  
- Adaptive vortex‑core tracking  
- Measurement of rectified drift  
- Gradient‑based updates to geometry parameters  

These results illustrate the potential of differentiable TDGL for inverse design.

---

## 4. Non‑Claims and Limitations

This repository does **not** claim:

- Optimized or globally optimal diode geometries  
- Quantitative rectification efficiency  
- Multi‑vortex or high‑field behavior  
- Finite‑size scaling or convergence analysis  
- Benchmarking against established TDGL solvers  
- Robustness across noise seeds or parameter sweeps  

Simulations use a modest grid size (48×48) and a fixed‑iteration Jacobi μ‑solver.

---

## 5. Repository Structure

```
Differentiable-TDGL/
│
├── tdgl_core.py          # TDGL evolution, energy functional, μ-solver
├── geometry.py           # Ratchet geometry parametrization
├── optimize.py           # Drift loss and gradient-based optimization
├── run_demo.py           # Example demonstration script
├── plots/                # Optional output figures
└── README.md             # This document
```

---

## 6. Running the Demonstration

Install dependencies:

```bash
pip install jax jaxlib optax matplotlib
```

Run the example:

```bash
python run_demo.py
```

The script will:

1. Construct a ratchet geometry  
2. Evolve the TDGL system under opposite currents  
3. Track the vortex position  
4. Compute drift asymmetry  
5. Generate plots of geometry and trajectories  

---

## 7. Interpretation of Results

Typical outcomes include:

- A non‑zero difference in vortex drift under \( \pm J_{\text{ext}} \)  
- A ratchet‑like geometry emerging from optimization  
- Stable single‑vortex motion within the channel  

These results are **illustrative** and should not be interpreted as quantitative device predictions.

---

## 8. Citation

If you use this repository in academic work, please cite:

```
H. Hardiyan & Copilot,
"Differentiable TDGL 2D Framework for Vortex Transport and Geometry Optimization",
2025. GitHub: https://github.com/harihardiyan/Differentiable-TDGL
```

---

## 9. Contact

For questions or collaboration:

**Hari Hardiyan**  
Email: lorozloraz@gmail.com

---

## 10. License

MIT License.
