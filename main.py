# ============================================================
# TDGL 2D – Differentiable vortex diode in JAX
# 48×48, channel ratchet, explicit pinning, adaptive core tracking
# Paper-ready demo: single-vortex rectified drift under ±J_ext
# ============================================================

!pip install --quiet jax jaxlib optax matplotlib

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax import tree_util
import optax
import matplotlib.pyplot as plt

# ============================================================
# 1. PyTrees: State & Params
# ============================================================

@tree_util.register_pytree_node_class
@dataclass
class State:
    psi: jnp.ndarray  # (ny, nx, 2) real-imag
    A: jnp.ndarray    # (ny, nx, 2)
    mu: jnp.ndarray   # (ny, nx)

    def tree_flatten(self):
        return (self.psi, self.A, self.mu), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        psi, A, mu = children
        return cls(psi=psi, A=A, mu=mu)


@tree_util.register_pytree_node_class
@dataclass
class Params:
    nx: int
    ny: int
    dx: float
    dy: float
    kappa: float
    gamma_psi: float
    gamma_A: float
    sigma_n: float
    J_ext: float
    a: jnp.ndarray      # GL coefficient a(x,y)
    H_ext: jnp.ndarray  # external field Bz(x,y)
    mask: jnp.ndarray   # channel mask (ny, nx)

    def tree_flatten(self):
        children = (self.nx, self.ny, self.dx, self.dy,
                    self.kappa, self.gamma_psi, self.gamma_A,
                    self.sigma_n, self.J_ext, self.a, self.H_ext, self.mask)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        nx, ny, dx, dy, kappa, gamma_psi, gamma_A, sigma_n, J_ext, a, H_ext, mask = children
        return cls(nx=nx, ny=ny, dx=dx, dy=dy,
                   kappa=kappa, gamma_psi=gamma_psi, gamma_A=gamma_A,
                   sigma_n=sigma_n, J_ext=J_ext, a=a, H_ext=H_ext, mask=mask)


# ============================================================
# 2. Complex helpers
# ============================================================

def to_complex(phi):
    return phi[..., 0] + 1j * phi[..., 1]

def to_real(z):
    return jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)


# ============================================================
# 3. Differential operators (Neumann)
# ============================================================

def grad_x_neumann(f, dx):
    ny, nx = f.shape
    df = jnp.zeros_like(f)
    df = df.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2 * dx))
    return df

def grad_y_neumann(f, dy):
    ny, nx = f.shape
    df = jnp.zeros_like(f)
    df = df.at[1:-1, :].set((f[2:, :] - f[:-2, :]) / (2 * dy))
    return df

def curl_A_neumann(A, dx, dy):
    Ax = A[..., 0]
    Ay = A[..., 1]
    return grad_x_neumann(Ay, dx) - grad_y_neumann(Ax, dy)


# ============================================================
# 4. Energy functional
# ============================================================

def covariant_grad_psi(psi_c, A, dx, dy):
    Ax = A[..., 0]
    Ay = A[..., 1]
    dpsi_dx = grad_x_neumann(psi_c, dx)
    dpsi_dy = grad_y_neumann(psi_c, dy)
    Dx = -1j * dpsi_dx - Ax * psi_c
    Dy = -1j * dpsi_dy - Ay * psi_c
    return Dx, Dy

def energy_density(state: State, params: Params):
    psi_c = to_complex(state.psi)
    A = state.A
    abs2 = jnp.abs(psi_c)**2

    term_local = params.a * abs2 + 0.5 * abs2**2
    Dx, Dy = covariant_grad_psi(psi_c, A, params.dx, params.dy)
    term_kin = jnp.abs(Dx)**2 + jnp.abs(Dy)**2

    Bz = curl_A_neumann(A, params.dx, params.dy)
    term_mag = params.kappa**2 * (Bz - params.H_ext)**2

    return term_local + term_kin + term_mag

def total_energy(state: State, params: Params):
    return jnp.sum(energy_density(state, params)) * params.dx * params.dy


# ============================================================
# 5. TDGL evolution + μ-solver
# ============================================================

def tdgl_rhs(state: State, params: Params):
    def F(psi, A):
        return total_energy(State(psi=psi, A=A, mu=state.mu), params)

    gpsi, gA = jax.grad(F, argnums=(0, 1))(state.psi, state.A)

    grad_mu_x = grad_x_neumann(state.mu, params.dx)
    grad_mu_y = grad_y_neumann(state.mu, params.dy)
    Jn_x = -params.sigma_n * grad_mu_x
    Jn_y = -params.sigma_n * grad_mu_y

    dA_dt = -params.gamma_A * (gA + jnp.stack([Jn_x, Jn_y], axis=-1))
    dpsi_dt = -params.gamma_psi * gpsi

    return State(psi=dpsi_dt, A=dA_dt, mu=jnp.zeros_like(state.mu))


def solve_mu_jacobi(mu0, params: Params, n_iter: int):
    dx = params.dx
    dy = params.dy
    sigma = params.sigma_n
    Jext = params.J_ext

    def body(i, mu):
        ny, nx = mu.shape
        mu_new = mu

        mu_new = mu_new.at[1:-1, 1:-1].set(
            (
                (mu[1:-1, 2:] + mu[1:-1, :-2]) / dx**2 +
                (mu[2:, 1:-1] + mu[:-2, 1:-1]) / dy**2
            ) / (2/dx**2 + 2/dy**2)
        )

        dmu = -Jext * dx / sigma
        mu_new = mu_new.at[:, 0].set(mu_new[:, 1] + dmu)
        mu_new = mu_new.at[:, -1].set(mu_new[:, -2] - dmu)

        return mu_new

    return jax.lax.fori_loop(0, n_iter, body, mu0)


def tdgl_step(state: State, params: Params, dt, n_mu_iter: int):
    mu_new = solve_mu_jacobi(state.mu, params, n_iter=n_mu_iter)
    rhs = tdgl_rhs(State(state.psi, state.A, mu_new), params)
    return State(
        psi = state.psi + dt * rhs.psi,
        A   = state.A   + dt * rhs.A,
        mu  = mu_new
    )


# ============================================================
# 6. Vortex core tracking (adaptive, in-channel)
# ============================================================

def tdgl_evolve_track_centers(state0, params, dt, n_steps, n_mu_iter):
    mask = params.mask

    def body(carry, i):
        s = carry
        s_new = tdgl_step(s, params, dt, n_mu_iter)
        psi_c = to_complex(s_new.psi)
        dens = jnp.abs(psi_c)**2

        flat_dens = dens.reshape(-1)
        flat_mask = mask.reshape(-1)
        big = jnp.max(flat_dens) + 1.0
        dens_in = jnp.where(flat_mask > 0.5, flat_dens, big)
        thr = jnp.percentile(dens_in, 10.0)

        core = (dens * mask) < thr

        ny, nx = dens.shape
        x = jnp.arange(nx)[None, :].astype(jnp.float32)
        core_f = core.astype(jnp.float32)
        w = jnp.sum(core_f)
        x_cm = jnp.where(w > 0, jnp.sum(core_f * x) / (w + 1e-9), 0.5*(nx-1))
        return s_new, x_cm

    _, x_traj = jax.lax.scan(body, state0, jnp.arange(n_steps))
    return x_traj


# ============================================================
# 7. Geometry: ratchet channel + pinning from theta
# ============================================================

def make_ratchet_mask_param(ny, nx, theta):
    w_min_frac = 0.05 + 0.3 * jax.nn.sigmoid(theta[0])
    w_max_frac = 0.15 + 0.4 * jax.nn.sigmoid(theta[1])

    tilt_amp = 0.2 * jax.nn.tanh(theta[4])
    tilt_phase = theta[5]

    y = jnp.arange(ny)[:, None]
    x = jnp.arange(nx)[None, :]
    yc = ny / 2.0

    frac = x / (nx - 1 + 1e-9)
    width = w_min_frac*ny + (w_max_frac*ny - w_min_frac*ny) * frac

    center_tilt = tilt_amp * ny * jnp.sin(2 * jnp.pi * frac + tilt_phase)
    yc_eff = yc + center_tilt

    y_upper = yc_eff + width / 2.0
    y_lower = yc_eff - width / 2.0

    mask = (y >= y_lower) & (y <= y_upper)
    return mask.astype(jnp.float32)


def make_a_from_theta(theta, ny, nx):
    pin_strength = 0.5 * jax.nn.softplus(theta[2])
    pin_radius = 2.0
    pin_bias_x = jax.nn.sigmoid(theta[3])
    pin_bias_y = jax.nn.sigmoid(theta[6])
    pin_grad = 0.5 * jax.nn.tanh(theta[7])

    mask = make_ratchet_mask_param(ny, nx, theta)
    y = jnp.arange(ny)[:, None]
    x = jnp.arange(nx)[None, :]

    a_base = -1.0 * mask + 4.0 * (1.0 - mask)

    xs = jnp.linspace(nx*(0.3+0.4*pin_bias_x), nx-1, 3)
    ys = jnp.linspace(ny*(0.2+0.6*pin_bias_y), ny*(0.3+0.7*pin_bias_y), 2)

    a_mod = jnp.zeros((ny, nx))
    for j, yc in enumerate(ys):
        for i, xc in enumerate(xs):
            r2 = (y - yc)**2 + (x - xc)**2
            local_strength = pin_strength * (1.0 + pin_grad * (i / max(len(xs)-1, 1)))
            g = local_strength * jnp.exp(-r2 / (2 * pin_radius**2))
            a_mod = a_mod + g

    a_mod = a_mod / (a_mod.max() + 1e-9)
    a = a_base + a_mod * mask
    a = jnp.clip(a, -1.0, 6.0)
    return a, mask


# ============================================================
# 8. Init params & state
# ============================================================

def make_uniform_field(nx, ny, B0=0.02):
    return jnp.ones((ny, nx)) * B0

def init_params_from_theta(theta, nx=48, ny=48,
                           Lx=20.0, Ly=20.0,
                           kappa=2.0,
                           gamma_psi=1.0,
                           gamma_A=1.0,
                           sigma_n=1.0,
                           J_ext=0.20,
                           B0=0.02):
    dx = Lx / nx
    dy = Ly / ny
    a, mask = make_a_from_theta(theta, ny, nx)
    H_ext = make_uniform_field(nx, ny, B0=B0)
    return Params(nx=nx, ny=ny, dx=dx, dy=dy,
                  kappa=kappa, gamma_psi=gamma_psi, gamma_A=gamma_A,
                  sigma_n=sigma_n, J_ext=J_ext,
                  a=a, H_ext=H_ext, mask=mask)

def init_state(params, noise=0.1, key=jax.random.PRNGKey(0)):
    ny, nx = params.ny, params.nx
    psi0 = 1.0 + noise * jax.random.normal(key, (ny, nx))
    psi0 = to_real(psi0.astype(jnp.complex64))
    mu0 = jnp.zeros((ny, nx))
    A0 = jnp.zeros((ny, nx, 2))
    return State(psi=psi0, A=A0, mu=mu0)


# ============================================================
# 9. Drift runners
# ============================================================

def run_tdgl_vortex_drift_traj(theta, J_ext,
                               n_relax=300,
                               n_steps=700,
                               dt=0.01,
                               n_mu_iter=20,
                               key=jax.random.PRNGKey(0)):
    params = init_params_from_theta(theta, nx=48, ny=48, J_ext=J_ext)
    state0 = init_state(params, key=key)

    def body_relax(carry, i):
        return tdgl_step(carry, params, dt, n_mu_iter), None

    state_relaxed, _ = jax.lax.scan(body_relax, state0, jnp.arange(n_relax))
    x_traj = tdgl_evolve_track_centers(state_relaxed, params, dt, n_steps, n_mu_iter)
    return x_traj, params


def run_tdgl_vortex_drift(theta, J_ext,
                          n_relax=300,
                          n_steps=700,
                          dt=0.01,
                          n_mu_iter=20,
                          key=jax.random.PRNGKey(0)):
    x_traj, _ = run_tdgl_vortex_drift_traj(theta, J_ext, n_relax, n_steps, dt, n_mu_iter, key)
    return x_traj[-1] - x_traj[0]


# ============================================================
# 10. Drift-based loss – maximize |rect_drift|
# ============================================================

def drift_loss(theta):
    key_p = jax.random.PRNGKey(0)
    key_m = jax.random.PRNGKey(1)

    Jp = 0.20
    Jm = -0.20

    dp = run_tdgl_vortex_drift(theta, Jp, key=key_p)
    dm = run_tdgl_vortex_drift(theta, Jm, key=key_m)
    rect_drift = dp - dm

    reg = 5e-4 * jnp.mean(theta**2)
    return -jnp.abs(rect_drift) + reg, rect_drift


loss_and_grad = jax.value_and_grad(lambda th: drift_loss(th)[0])


# ============================================================
# 11. Optimization loop (hero run)
# ============================================================

theta0 = jnp.array([
    0.2,   # width min bias
   -0.2,   # width max bias
    0.3,   # pin strength initial
    0.2,   # pin bias x
    0.2,   # tilt amplitude
    0.0,   # tilt phase
    0.0,   # pin bias y
    0.2    # pin gradient
])

opt = optax.chain(
    optax.clip(0.1),
    optax.adam(2e-3)
)

opt_state = opt.init(theta0)
theta = theta0
n_opt = 100

print("=== Optimization ===")
for it in range(n_opt):
    loss_val, g = loss_and_grad(theta)
    updates, opt_state = opt.update(g, opt_state, theta)
    theta = optax.apply_updates(theta, updates)

    if it % 5 == 0:
        _, rect = drift_loss(theta)
        print(f"iter {it:03d}  loss={float(loss_val):.6e}  rect_drift={float(rect):.6e}")

print("Optimized theta:", theta)

# ============================================================
# 12. Final evaluation & plots
# ============================================================

print("\n=== Final evaluation ===")
x_traj_p, params_p = run_tdgl_vortex_drift_traj(theta, 0.20, key=jax.random.PRNGKey(123))
x_traj_m, params_m = run_tdgl_vortex_drift_traj(theta, -0.20, key=jax.random.PRNGKey(456))

dp = x_traj_p[-1] - x_traj_p[0]
dm = x_traj_m[-1] - x_traj_m[0]
rect = dp - dm

print("drift+, J_ext>0:", float(dp))
print("drift-, J_ext<0:", float(dm))
print("rect_drift      :", float(rect))

plt.figure(figsize=(6,4))
plt.plot(x_traj_p, label="J_ext > 0")
plt.plot(x_traj_m, label="J_ext < 0")
plt.xlabel("time step")
plt.ylabel("x_cm")
plt.legend()
plt.title("Vortex drift trajectories")
plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
plt.imshow(params_p.a, origin="lower")
plt.colorbar()
plt.title("a(x,y) optimized")
plt.tight_layout()
plt.show()
