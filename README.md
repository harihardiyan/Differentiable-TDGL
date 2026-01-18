

# Differentiable-TDGL: Inverse Design of Superconducting Vortex Ratchets

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/Framework-JAX-red.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
This repository presents a **Differentiable Time-Dependent Ginzburg-Landau (TDGL)** framework implemented in JAX for the inverse design of superconducting vortex ratchets. Traditionally, vortex guides and diodes are designed through intuition-led trial and error. By leveraging automatic differentiation (autodiff), this framework enables the calculation of gradients of vortex trajectories with respect to the underlying material landscape (the $a(x,y)$ coefficient) and channel geometry. As a proof-of-concept, we demonstrate an "inverse-designed" asymmetric channel that exhibits rectified single-vortex drift under alternating external current ($J_{ext}$).

## Key Features
- **Differentiable Physics:** Entire TDGL solver is written in JAX, allowing backpropagation through the time-evolution of the Order Parameter ($\psi$) and Vector Potential ($A$).
- **Adaptive Core Tracking:** Implements a differentiable center-of-mass tracking mechanism to locate vortex cores within the superconducting channel.
- **Geometry Parameterization:** A flexible ratchet mask and pinning landscape generator controlled by a learnable parameter vector $\theta$.
- **Jacobi $\mu$-Solver:** An integrated electrostatic potential solver for calculating current injection effects.

## Project Scope & Claims

### ✅ Scientific Claims (Proof-of-Concept)
1. **Differentiable Framework:** A functional pipeline that connects TDGL dynamics to gradient-based optimization.
2. **Inverse-Designed Geometry:** Successful demonstration of an automated process that alters a ratchet geometry to influence vortex motion.
3. **Rectified Drift:** Observation of asymmetric vortex displacement (rectification) under $\pm J_{ext}$ within the simulated environment.

### ❌ Non-Claims (Limitations)
To maintain academic integrity, this repository **does not** claim the following:
- **Optimized Diode:** The resulting geometries are "candidates" and not necessarily global optima for diode performance.
- **Robust Design:** The solution may be sensitive to initial conditions, noise, or specific $\kappa$ values.
- **Converged Solution:** The optimization path shown is a demonstration of the gradient flow, not a guarantee of mathematical convergence.
- **Quantitative Efficiency:** No claims are made regarding the experimental IV-rectification ratio or high-frequency performance.

## Installation

```bash
pip install jax jaxlib optax matplotlib
```

## Usage

The main script performs the following steps:
1. **Initialization:** Defines the GL parameters ($\kappa$, $\gamma$, $\sigma$) and the initial superconducting state.
2. **Forward Simulation:** Evolves the TDGL equations under a given external current.
3. **Loss Calculation:** Evaluates the "rectified drift" (the difference in vortex displacement between positive and negative current).
4. **Optimization:** Updates the geometry parameters $\theta$ using the Adam optimizer to maximize the drift asymmetry.

```python
# Run the demo
python tdgl_ratchet_opt.py
```

## Methodology

The framework minimizes/maximizes a loss function based on the vortex trajectory $x_{traj}$:
$$ \mathcal{L}(\theta) = - | \Delta x(+J_{ext}) - \Delta x(-J_{ext}) | + \lambda ||\theta||^2 $$
Where $\Delta x$ is the net displacement of the vortex core over $N$ time steps. By differentiating through the solver, we obtain $\nabla_{\theta} \mathcal{L}$, allowing the geometry to "evolve" toward a shape that favors unidirectional motion.

## Citation

If you use this framework in your research, please cite it as follows:

```bibtex
@software{Hardiyan_Differentiable_TDGL_2024,
  author = {Hardiyan, Hari and Copilot},
  title = {Differentiable TDGL: Inverse Design of Superconducting Vortex Ratchets},
  year = {2026},
  url = {https://github.com/your-username/differentiable-tdgl},
  note = {Proof-of-concept for gradient-based vortex diode design}
}
```

## Authors
- **Hari Hardiyan** - *AI Orchestration & Lead Developer* - [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)
- **Copilot** - *AI Pair Programmer*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Disclaimer: This repository is a research demo intended for computational physics exploration. It is provided "as-is" without guarantees of physical hardware reproducibility.*
