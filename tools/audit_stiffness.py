"""Numerical Stability and Stiffness Audit for SWAT.

This tool computes the Jacobian of the drift equations across the state space
to determine the spectral radius and the maximum stable step size for
explicit integration (e.g., Euler-Maruyama).
"""

import os
import sys

# Force CPU for stability and Enable JAX X64
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import numpy as np

# Add model path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.swat._dynamics import drift_jax
from models.swat.simulation import DEFAULT_PARAMS

def audit_stiffness():
    print("Running Vectorized Stiffness Audit (CPU)...")
    p_dict = {k: jnp.float64(v) for k, v in DEFAULT_PARAMS.items()}
    
    # Jacobian of drift wrt state y
    jac_fn = jax.jacobian(drift_jax, argnums=0)
    # Vmapped Jacobian
    jac_vmap = jax.vmap(lambda y, u: jac_fn(y, p_dict, 0.5, u))
    
    # Grid setup
    n_samples = 6
    w_grid = jnp.linspace(0.01, 0.99, n_samples)
    z_grid = jnp.linspace(0.01, 0.99, n_samples)
    a_grid = jnp.linspace(0.01, 0.99, n_samples)
    t_grid = jnp.linspace(0.0, 2.0, n_samples)
    vh_vals = jnp.array([0.1, 1.0])
    vn_vals = jnp.array([0.0, 1.0])
    
    # Cartesian product
    W, Z, A, T, VH, VN = jnp.meshgrid(w_grid, z_grid, a_grid, t_grid, vh_vals, vn_vals, indexing='ij')
    
    y_flat = jnp.stack([W.flatten(), Z.flatten(), A.flatten(), T.flatten()], axis=1)
    u_flat = jnp.stack([VH.flatten(), VN.flatten(), jnp.zeros_like(VH.flatten())], axis=1)
    
    print(f"Computing Jacobians for {len(y_flat)} points...")
    Js = jac_vmap(y_flat, u_flat) # (N, 4, 4)
    
    print("Computing spectral radii...")
    # eigvals for 4x4 matrices
    eigvals = jnp.linalg.eigvals(Js) # (N, 4)
    rhos = jnp.max(jnp.abs(eigvals), axis=1) # (N,)
    
    max_idx = jnp.argmax(rhos)
    max_rho = rhos[max_idx]
    worst_y = y_flat[max_idx]
    worst_u = u_flat[max_idx]

    # Results
    h_max_days = 2.0 / max_rho
    h_max_mins = h_max_days * 24.0 * 60.0
    
    print(f"\nStiffness Audit Results:")
    print(f"  Max Spectral Radius (rho): {max_rho:.4f} day^-1")
    print(f"  Worst-case state: W={worst_y[0]:.3f}, Z={worst_y[1]:.3f}, a={worst_y[2]:.3f}, T={worst_y[3]:.3f}")
    print(f"  Worst-case controls: V_h={worst_u[0]:.1f}, V_n={worst_u[1]:.1f}")
    print(f"  Maximum stable step size (h_max): {h_max_mins:.2f} minutes")
    
    THRESHOLD_MINS = 15.0
    if h_max_mins >= THRESHOLD_MINS:
        print(f"  PASS: Model is stable for {THRESHOLD_MINS} min steps.")
        passed = True
        n_substeps = 1
    else:
        n_substeps = int(np.ceil(THRESHOLD_MINS / h_max_mins))
        print(f"  WARNING: Model is STIFF! Requires at least {n_substeps} sub-steps for 15 min bins.")
        passed = False

    return {
        'h_max_mins': float(h_max_mins),
        'max_rho_per_day': float(max_rho),
        'n_substeps_for_15min': n_substeps,
        'passed': passed,
    }

if __name__ == "__main__":
    audit_stiffness()
