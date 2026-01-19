"""
run_demo.py
Example demonstration of differentiable TDGL vortex drift and geometry optimization.
"""

import jax.numpy as jnp
from optimize import optimize_theta, run_drift
from geometry import make_geometry
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Starting differentiable TDGL demonstration...")

    theta = jnp.zeros(1)
    theta_opt = optimize_theta(theta, n_steps=20)

    print("Optimized theta:", theta_opt)

    # Evaluate drift
    d_pos = run_drift(theta_opt, 0.2)
    d_neg = run_drift(theta_opt, -0.2)

    print("Drift +J:", float(d_pos))
    print("Drift -J:", float(d_neg))
    print("Rectified drift:", float(d_pos - d_neg))

    # Plot geometry
    a, mask = make_geometry(theta_opt, 48, 48)
    plt.imshow(a, origin="lower")
    plt.colorbar()
    plt.title("Geometry a(x,y)")
    plt.show()
