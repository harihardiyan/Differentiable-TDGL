# Differentiable TDGL: Inverse Design of Superconducting Vortex Ratchets

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/Framework-JAX-red.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This repository presents a **Differentiable Time-Dependent Ginzburg-Landau (TDGL)** framework implemented in JAX. The project explores the **inverse design** of superconducting fluxonic devices by utilizing automatic differentiation to optimize material landscapes. 

Specifically, the framework parameterizes the channel geometry and explicit pinning sites (via a vector $\theta$) and backpropagates through the TDGL solver to discover asymmetric configurations. The primary result is a **proof-of-concept vortex diode** that exhibits rectified single-vortex drift under alternating external current densities ($\pm J_{ext}$), effectively steering fluxons through gradient-based geometric evolution.

## Technical Architecture

The solver is engineered for high-performance differentiable physics:
*   **Grid Discretization:** 48×48 numerical domain using Neumann boundary conditions.
*   **State Dynamics:** Simultaneous evolution of the complex order parameter ($\psi$) and the vector potential ($A$).
*   **Electrostatic Integration:** A Jacobi-iterative $\mu$-solver computes the scalar potential for current injection.
*   **Differentiable Tracking:** An adaptive vortex core tracking algorithm that utilizes density percentiles and center-of-mass calculations, ensuring the vortex trajectory is a continuous and differentiable function of the design parameters.
*   **Optimization Pipeline:** Integration with `optax` for gradient-based updates of the geometry and pinning landscapes.

## Installation

The framework requires JAX and standard optimization libraries. It is recommended to use a hardware-accelerated environment (GPU/TPU) for faster backpropagation.

```bash
pip install --quiet jax jaxlib optax matplotlib
```

## Scientific Scope & Limitations

### ✅ Formal Claims (Proof-of-Concept)
*   **Differentiable TDGL Implementation:** Successful integration of TDGL dynamics within a gradient-based optimization loop.
*   **Inverse-Designed Geometry:** Discovery of asymmetric "ratchet" landscapes that influence fluxon dynamics.
*   **Rectified Drift Demonstration:** Qualitative observation of unidirectional vortex displacement under alternating drive polarities.

### ❌ Non-Claims & Constraints
*   **Not an "Optimized Diode":** The resulting structural configurations are candidates found within the parameter space, not global optima.
*   **Not a "Robust Design":** Sensitivity to initialization and noise is expected; the design is not yet characterized for environmental robustness.
*   **Not a "Converged Solution":** The optimization provides a demonstration of gradient flow rather than a guarantee of mathematical convergence to a unique minimum.
*   **Not "Quantitative Efficiency":** This work does not claim experimental-grade rectification ratios or high-frequency performance metrics.

## Usage

### Running the Optimization
To execute the inverse design process, run the main script. The system will perform 100 iterations of Adam optimization to shape the superconducting landscape.

```bash
python main.py
```

### Execution Flow
1.  **State Relaxation:** The system initializes $\psi$ and $A$ to reach a metastable state with a single vortex.
2.  **Trajectory Evaluation:** The script runs two simulations in parallel (for $+J_{ext}$ and $-J_{ext}$).
3.  **Loss Computation:** The objective function maximizes the absolute rectified drift: $|\Delta x_{pos} - \Delta x_{neg}|$.
4.  **Geometry Update:** The parameters $\theta$ are updated to refine the channel width, tilt, and pinning gradients.

## Results & Visualization

Upon completion, the framework generates two critical analytical plots:
*   **Vortex Drift Trajectories:** Time-evolution of the vortex center-of-mass ($x_{cm}$) for both current polarities, showing the asymmetry in transport.
*   **Optimized $a(x,y)$ Landscape:** A 2D map of the Ginzburg-Landau coefficient, revealing the discovered ratchet geometry and the spatial distribution of pinning sites.

## Authors

*   **Hari Hardiyan** (AI Orchestration) - [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)
*   **Copilot** (AI Pair Programmer)

## Citation

If this framework contributes to your research, please cite it as follows:

```bibtex
@software{Hardiyan_TDGL_JAX_2026,
  author = {Hardiyan, Hari and Copilot},
  title = {Differentiable TDGL 2D: Inverse Design of Vortex Ratchets},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/harihardiyan/differentiable-tdgl},
  note = {Proof-of-concept for gradient-based vortex diode design}
}
```

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for full details.
