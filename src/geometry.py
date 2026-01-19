"""
geometry.py
Differentiable ratchet-channel geometry and GL coefficient construction.
"""

import jax
import jax.numpy as jnp

def make_geometry(theta, ny, nx):
    """
    Simple differentiable ratchet channel.
    theta[0] controls width variation.
    """
    x = jnp.arange(nx)[None, :]
    y = jnp.arange(ny)[:, None]

    width = (0.1 + 0.4 * jax.nn.sigmoid(theta[0])) * ny * (x / nx) + 0.1 * ny
    mask = (jnp.abs(y - ny/2) < width).astype(jnp.float32)

    a = -1.0 * mask + 4.0 * (1.0 - mask)
    return a, mask
