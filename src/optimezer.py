"""
optimize.py
Loss function based on rectified vortex drift and gradient-based optimization.
"""

import jax
import jax.numpy as jnp
import optax

from tdgl_core import State, Params, tdgl_step, to_complex
from geometry import make_geometry

def run_drift(theta, J_ext):
    a, mask = make_geometry(theta, 48, 48)
    params = Params(
        nx=48, ny=48, dx=0.4, dy=0.4,
        kappa=2.0, gamma_psi=1.0, gamma_A=1.0,
        sigma_n=1.0, J_ext=J_ext,
        a=a, H_ext=jnp.ones((48,48))*0.02, mask=mask
    )

    psi0 = to_complex(1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (48,48)))
    psi0 = jnp.stack([jnp.real(psi0), jnp.imag(psi0)], axis=-1)

    state = State(psi=psi0, A=jnp.zeros((48,48,2)), mu=jnp.zeros((48,48)))

    for _ in range(200):
        state = tdgl_step(state, params, dt=0.01, n_mu_iter=10)

    dens = jnp.abs(to_complex(state.psi))**2
    x_cm = jnp.sum(dens * jnp.arange(48)) / jnp.sum(dens)
    return x_cm


def loss_fn(theta):
    d_pos = run_drift(theta, 0.2)
    d_neg = run_drift(theta, -0.2)
    return -jnp.abs(d_pos - d_neg)


def optimize_theta(theta0, n_steps=20, lr=0.01):
    opt = optax.adam(lr)
    opt_state = opt.init(theta0)

    for i in range(n_steps):
        loss, grads = jax.value_and_grad(loss_fn)(theta0)
        updates, opt_state = opt.update(grads, opt_state)
        theta0 = optax.apply_updates(theta0, updates)
        print(f"Step {i}, Loss = {loss:.6f}")

    return theta0
