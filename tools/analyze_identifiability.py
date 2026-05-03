"""Fisher Information Matrix (FIM) Analysis for SWAT Identifiability.

This tool computes the Fisher Information Matrix for the SWAT model
parameters to identify which subsets are mathematically identifiable
from the 4-channel observation model.
"""

import os
import sys

# Enable JAX X64
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import numpy as np
from collections import OrderedDict

# Add model path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.swat._dynamics import drift_jax, diffusion_state_dep
from models.swat.estimation import PARAM_PRIOR_CONFIG, FROZEN_PARAMS, _PI
from models.swat.simulation import DEFAULT_INIT, DEFAULT_PARAMS, scenario_presets

def compute_fim(scenario_name="A_healthy", n_days=7):
    """Computes the FIM using the sensitivity of the mean trajectory."""
    
    p_truth_dict = DEFAULT_PARAMS.copy()
    
    # 11-parameter subset
    estimable_names = [
        'E_crit', 'alpha_T', 'V_n_scale', 
        'delta_HR', 'sigma_HR', 
        'c_tilde', 'delta_c', 
        'mu_step0', 'sigma_step', 
        'delta_s', 'sigma_s'
    ]
    # Filter to only those present in DEFAULT_PARAMS or that we can map
    active_names = []
    for n in estimable_names:
        if n in p_truth_dict:
            active_names.append(n)
        elif n == 'delta_HR': active_names.append('HR_base')
        elif n == 'delta_s': active_names.append('s_base')
        elif n == 'c_tilde': active_names.append('c_tilde')
        elif n == 'delta_c': active_names.append('delta_c')

    active_names = sorted(list(set(active_names)))
    p_truth_vec = jnp.array([p_truth_dict[name] for name in active_names])
    
    presets = scenario_presets(n_days)
    preset = presets[scenario_name]
    init_state = jnp.array(preset['init'])
    dt = 1.0 / 288.0
    n_bins = int(n_days / dt)
    t_eval = jnp.arange(n_bins) * dt
    V_h = jnp.array(np.repeat(preset['v_h_daily'], 288))
    V_n = jnp.array(np.repeat(preset['v_n_daily'], 288))
    V_c = jnp.array(np.repeat(preset['v_c_daily'], 288))
    u_traj = jnp.stack([V_h, V_n, V_c], axis=1)

    print(f"Computing FIM for {len(active_names)} parameters in {scenario_name}...")

    def get_mean_obs(theta_vec):
        p_dict = {name: theta_vec[i] for i, name in enumerate(active_names)}
        for fname, fval in FROZEN_PARAMS.items():
            if fname not in p_dict:
                p_dict[fname] = jnp.float64(fval)
        
        # Trajectory
        def body(y, k):
            f = drift_jax(y, p_dict, t_eval[k], u_traj[k])
            y_next = y + f * dt
            return y_next, y_next
        _, y_traj = jax.lax.scan(body, init_state, jnp.arange(n_bins))
        y_traj = jnp.concatenate([init_state[None, :], y_traj[:-1]], axis=0)
        
        W = y_traj[:, 0]
        # Observations (mean values)
        hr = (p_dict.get('HR_base', 50.0)) + p_dict.get('alpha_HR', 25.0) * W
        steps = p_dict['mu_step0'] + p_dict.get('beta_W_steps', 0.8) * W
        stress = (p_dict.get('s_base', 30.0)) + p_dict.get('alpha_s', 40.0) * W + p_dict.get('beta_s', 10.0) * V_n
        
        return jnp.stack([hr, steps, stress], axis=1)

    # Jacobian of the mean observations wrt parameters
    jacobian_fn = jax.jacobian(get_mean_obs)
    J = jacobian_fn(p_truth_vec) # (N_bins, 3, N_params)
    
    # Noise scales (sigmas)
    sigma_hr = p_truth_dict['sigma_HR']
    sigma_step = p_truth_dict['sigma_step']
    sigma_s = p_truth_dict['sigma_s']
    inv_cov = jnp.diag(jnp.array([1.0/sigma_hr**2, 1.0/sigma_step**2, 1.0/sigma_s**2]))
    
    # FIM = sum_k J_k^T * InvCov * J_k
    # J is (N_bins, 3, N_params)
    # We want (N_params, N_params)
    
    # J has shape (N_bins, 3, N_params)
    # inv_cov has shape (3, 3)
    # Indices: b=bins, i/k=obs, j/l=params
    FIM = jnp.einsum('bij,ik,bkl->jl', J, inv_cov, J)
    
    # Add info for sigmas themselves (Analytical FIM for sigma is 2/sigma^2 per bin)
    fim_full = np.array(FIM)
    for i, name in enumerate(active_names):
        if name == 'sigma_HR': fim_full[i, i] += 2.0 * n_bins / sigma_hr**2
        if name == 'sigma_step': fim_full[i, i] += 2.0 * n_bins / sigma_step**2
        if name == 'sigma_s': fim_full[i, i] += 2.0 * n_bins / sigma_s**2
        
    return fim_full, active_names

if __name__ == "__main__":
    fim_a, names = compute_fim("A_healthy", n_days=7)
    fim_c, _ = compute_fim("C_recovery", n_days=7)
    
    fim = fim_a + fim_c
    
    print("\nFIM Identifiability Results (Combined A+C):")
    diag = np.diag(fim)
    for i, name in enumerate(names):
        print(f"  {name:<15}: {diag[i]:.4e}")
        
    evals, evecs = np.linalg.eigh(fim)
    cond = evals[-1] / (jnp.maximum(evals[0], 1e-18))
    
    print(f"\nCondition Number: {cond:.4e}")
    print(f"Eigenvalues: {evals}")
    
    if evals[0] < 1e-6:
        print("\nDegeneracy Detected! Smallest Eigenvector components:")
        min_vec = evecs[:, 0]
        for i, name in enumerate(names):
            if abs(min_vec[i]) > 0.1:
                print(f"  {name:<15}: {min_vec[i]:.4f}")
