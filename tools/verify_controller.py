"""Closed-loop Sandbox Verification (Controller Gradient Test).

Verifies that the MPC controller cost gradient correctly points toward
improved vitality and that the controller functional decreases the cost
over a single-stride planning horizon.
"""

import os
import sys

# Enable JAX X64
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import numpy as np

# Add model path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.swat.control import build_control_spec
from models.swat.simulation import DEFAULT_PARAMS, DEFAULT_INIT

def verify_controller():
    print("Running Controller Functional Verification...")
    
    # 1. Setup a pathological state (extremely low testosterone, high adenosine)
    y_bad = jnp.array([0.1, 0.1, 0.9, 0.001]) # [W, Z, a, T]

    # 2. Planning setup
    horizon_bins = 288 * 1 # 1 day
    dt_days = 1.0 / 288.0
    
    # Force deterministic rollout for gradient stability
    p_det = DEFAULT_PARAMS.copy()
    p_det['T_W'] = 0.0
    p_det['T_Z'] = 0.0
    p_det['T_a'] = 0.0
    p_det['T_T'] = 0.0
    
    # Build a ControlSpec
    spec = build_control_spec(
        n_steps=horizon_bins,
        dt=dt_days,
        n_anchors=8,
        n_inner=1, 
        init_state=y_bad,
        params=p_det,
        seed=42
    )

    # 3. Compute Cost and Gradient
    theta0 = jnp.zeros(spec.theta_dim)

    c0 = spec.cost_fn(theta0)
    print(f"  Initial Cost (7d plan): {c0:.4f}")

    grad_fn = jax.grad(spec.cost_fn)
    g = grad_fn(theta0)
    print(f"  Gradient range: [{jnp.min(g):.4e}, {jnp.max(g):.4e}]")

    mean_grad_vh = jnp.mean(g[0 : spec.theta_dim // 3])
    print(f"  Mean dCost/dTheta_Vh: {mean_grad_vh:.6f}")

    mean_grad_vn = jnp.mean(g[spec.theta_dim // 3 : 2 * spec.theta_dim // 3])
    print(f"  Mean dCost/dTheta_Vn: {mean_grad_vn:.6f}")

    # 5. Small step optimization
    learning_rate = 100.0 # Large step since gradients are tiny
    theta_new = theta0 - learning_rate * g
    c1 = spec.cost_fn(theta_new)
    print(f"\n[Step Test] Cost after large gradient step (LR=100): {c1:.6f}")
    assert c1 < c0

    print(f"\n[Step Test] Cost after one gradient step: {c1:.4f}")
    assert c1 < c0
    
    print("\nController Sandbox: PASS")

if __name__ == "__main__":
    verify_controller()
