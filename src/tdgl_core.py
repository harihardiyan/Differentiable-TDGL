"""
tdgl_core.py
Differentiable TDGL 2D evolution, energy functional, and μ-solver.
This module provides the core physics components used by the framework.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax import tree_util

# ------------------------------------------------------------
# 1. Data structures
# ------------------------------------------------------------

@tree_util.register_pytree_node_class
@dataclass
class State:
    psi: jnp.ndarray   # complex order parameter (real-imag stacked)
    A: jnp.ndarray     # vector potential (Ax, Ay)
    mu: jnp.ndarray    # electrochemical potential

    def tree_flatten(self):
        return (self.psi, self.A, self.mu), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        psi, A, mu = children
        return cls(psi=psi, A=A, mu=mu)


@tree_util.register_pytree_node_class
@dataclass
class Params:
    nx: int; ny: int
    dx: float; dy: float
    kappa: float
    gamma_psi: float
    gamma_A: float
    sigma_n: float
    J_ext: float
    a: jnp.ndarray
    H_ext: jnp.ndarray
    mask: jnp.ndarray

    def tree_flatten(self):
        return (self.nx, self.ny, self.dx, self.dy, self.kappa,
                self.gamma_psi, self.gamma_A, self.sigma_n, self.J_ext,
                self.a, self.H_ext, self.mask), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------

def to_complex(phi):
    return phi[..., 0] + 1j * phi[..., 1]

def to_real(z):
    return jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)

def grad_x_neumann(f, dx):
    df = jnp.zeros_like(f)
    df = df.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dx))
    return df

def grad_y_neumann(f, dy):
    df = jnp.zeros_like(f)
    df = df.at[1:-1, :].set((f[2:, :] - f[:-2, :]) / (2 * dy))
    return df


# ------------------------------------------------------------
# 3. Energy functional
# ------------------------------------------------------------

def energy_density(state: State, params: Params):
    psi = to_complex(state.psi)
    abs2 = jnp.abs(psi)**2

    Dx = -1j * grad_x_neumann(psi, params.dx) - state.A[..., 0] * psi
    Dy = -1j * grad_y_neumann(psi, params.dy) - state.A[..., 1] * psi

    Bz = grad_x_neumann(state.A[..., 1], params.dx) - grad_y_neumann(state.A[..., 0], params.dy)

    return (
        params.a * abs2
        + 0.5 * abs2**2
        + jnp.abs(Dx)**2
        + jnp.abs(Dy)**2
        + params.kappa**2 * (Bz - params.H_ext)**2
    )


# ------------------------------------------------------------
# 4. μ-solver (Jacobi)
# ------------------------------------------------------------

def solve_mu(mu, params: Params, n_iter: int):
    dx, dy, sigma, J = params.dx, params.dy, params.sigma_n, params.J_ext

    for _ in range(n_iter):
        mu = mu.at[1:-1, 1:-1].set(
            (
                (mu[1:-1, 2:] + mu[1:-1, :-2]) / dx**2 +
                (mu[2:, 1:-1] + mu[:-2, 1:-1]) / dy**2
            ) / (2/dx**2 + 2/dy**2)
        )

        dmu = -J * dx / sigma
        mu = mu.at[:, 0].set(mu[:, 1] + dmu)
        mu = mu.at[:, -1].set(mu[:, -2] - dmu)

    return mu


# ------------------------------------------------------------
# 5. TDGL evolution step
# ------------------------------------------------------------

def tdgl_step(state: State, params: Params, dt: float, n_mu_iter: int):
    mu_new = solve_mu(state.mu, params, n_mu_iter)

    def total_energy(psi, A):
        s = State(psi=psi, A=A, mu=mu_new)
        return jnp.sum(energy_density(s, params))

    gpsi, gA = jax.grad(total_energy, argnums=(0, 1))(state.psi, state.A)

    grad_mu = jnp.stack([
        grad_x_neumann(mu_new, params.dx),
        grad_y_neumann(mu_new, params.dy)
    ], axis=-1)

    Jn = -params.sigma_n * grad_mu

    psi_new = state.psi - dt * params.gamma_psi * gpsi
    A_new = state.A - dt * params.gamma_A * (gA + Jn)

    return State(psi=psi_new, A=A_new, mu=mu_new)
