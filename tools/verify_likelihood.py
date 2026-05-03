"""Likelihood Sanity and Sensitivity Test for SWAT.

Verifies that the log-likelihoods for each observation channel are peaked 
at the truth and that "illegal" or extreme states result in correctly 
lowered likelihoods.
"""

import os
import sys
import math

# Enable JAX X64
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import numpy as np

# Add model path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.swat.estimation import SWAT_ESTIMATION, PARAM_PRIOR_CONFIG
from models.swat.simulation import DEFAULT_PARAMS, DEFAULT_INIT

def verify_likelihood():
    print("Running Likelihood Sanity Tests...")
    em = SWAT_ESTIMATION
    p_dict = DEFAULT_PARAMS.copy()
    
    # Map p_dict to the vector expected by the estimator
    theta_list = []
    for name in PARAM_PRIOR_CONFIG.keys():
        if name.startswith('delta_') and name != 'delta_c':
            theta_list.append(0.0) # Truth has no offset relative to BASE
        else:
            theta_list.append(p_dict[name])
    p_vec = jnp.array(theta_list)
    
    # Debug: print mapped params
    print("\n[Debug] Estimator Parameter Vector:")
    for i, name in enumerate(PARAM_PRIOR_CONFIG.keys()):
        print(f"  {name:<12}: {float(p_vec[i]):.3f}")
    
    # 1. Setup Truth State and Observation
    y_truth = jnp.array([0.5, 0.583, 0.5, 0.5]) # [W, Z, a, T]
    W_truth = float(y_truth[0])
    Z_truth = float(y_truth[1])
    
    # Generate 'truth' observations for this state
    hr_obs = (p_dict['HR_base']) + p_dict['alpha_HR'] * W_truth
    sleep_label = 1 
    steps_obs = p_dict['mu_step0'] + p_dict['beta_W_steps'] * W_truth
    stress_obs = (p_dict['s_base']) + p_dict['alpha_s'] * W_truth + p_dict['beta_s'] * 0.2
    
    grid_obs = {
        'hr_value': jnp.array([hr_obs]),
        'hr_present': jnp.array([1.0]),
        'sleep_label': jnp.array([sleep_label]),
        'sleep_present': jnp.array([1.0]),
        'log_steps_value': jnp.array([steps_obs]),
        'steps_present': jnp.array([1.0]),
        'stress_value': jnp.array([stress_obs]),
        'stress_present': jnp.array([1.0]),
        'V_n': jnp.array([0.2])
    }

    # 2. Sensitivity wrt Wakefulness (W)
    print("\n[Per-Channel] Sensitivity to Wakefulness (W):")
    w_grid = np.linspace(0.0, 1.0, 100)
    
    hr_lws = []
    step_lws = []
    stress_lws = []
    
    def hr_lik(w, p):
        hr_mean = (p['HR_base']) + p['alpha_HR'] * w
        return -0.5 * math.log(2.0 * math.pi) - math.log(p['sigma_HR']) - 0.5 * ((hr_obs - hr_mean)/p['sigma_HR'])**2
    
    def step_lik(w, p):
        step_mean = p['mu_step0'] + p['beta_W_steps'] * w
        return -0.5 * math.log(2.0 * math.pi) - math.log(p['sigma_step']) - 0.5 * ((steps_obs - step_mean)/p['sigma_step'])**2

    def stress_lik(w, p):
        stress_mean = (p['s_base']) + p['alpha_s'] * w + p['beta_s'] * 0.2
        return -0.5 * math.log(2.0 * math.pi) - math.log(p['sigma_s']) - 0.5 * ((stress_obs - stress_mean)/p['sigma_s'])**2

    for w in w_grid:
        hr_lws.append(hr_lik(w, p_dict))
        step_lws.append(step_lik(w, p_dict))
        stress_lws.append(stress_lik(w, p_dict))
    
    print(f"  HR peaks at W = {w_grid[np.argmax(hr_lws)]:.3f} (Truth = {W_truth:.3f})")
    print(f"  Step peaks at W = {w_grid[np.argmax(step_lws)]:.3f} (Truth = {W_truth:.3f})")
    print(f"  Stress peaks at W = {w_grid[np.argmax(stress_lws)]:.3f} (Truth = {W_truth:.3f})")
    
    # Re-verify the EM call directly
    em_lws = []
    for w in w_grid:
        y = y_truth.at[0].set(w)
        lw = em.obs_log_weight_fn(y, grid_obs, 0, p_vec)
        em_lws.append(float(lw))
    peak_em_w = w_grid[np.argmax(em_lws)]
    print(f"  EM call peaks at W = {peak_em_w:.3f}")
    
    assert abs(peak_em_w - W_truth) < 0.02

    # 3. Sensitivity wrt Sleep Depth (Z)
    print("\n[Channel 2] Sensitivity to Sleep Depth (Z):")
    z_grid = np.linspace(0.0, 1.0, 50)
    lws_z = []
    for z in z_grid:
        y = y_truth.at[1].set(z)
        lw = em.obs_log_weight_fn(y, grid_obs, 0, p_vec)
        lws_z.append(float(lw))
        
    peak_z = z_grid[np.argmax(lws_z)]
    print(f"  Likelihood peaks at Z = {peak_z:.3f} (Truth ~ {Z_truth:.3f})")
    assert lws_z[np.argmin(np.abs(z_grid - Z_truth))] > np.max(lws_z) - 0.1

    # 4. Boundary Sanity
    print("\n[Boundary] Testing illegal states:")
    y_bad = jnp.array([-10.0, 20.0, -5.0, -1.0])
    lw_bad = em.obs_log_weight_fn(y_bad, grid_obs, 0, p_vec)
    print(f"  Log-likelihood for illegal state: {lw_bad:.2f}")
    assert lw_bad < em_lws[np.argmax(em_lws)] - 100.0

    # 5. Prior/Truth Alignment
    print("\n[Pillar IV] Prior vs Truth Alignment Check:")
    for name, (dist, params) in PARAM_PRIOR_CONFIG.items():
        if name in p_dict:
            truth = p_dict[name]
            if dist == 'normal':
                mu, sigma = params
                z = (truth - mu) / sigma
                print(f"  {name:<12}: Truth={truth:>7.2f}, Prior=N({mu:>5.1f}, {sigma:>4.1f}), Z={z:>5.2f}")
            elif dist == 'lognormal':
                log_mu, sigma = params
                mu_val = np.exp(log_mu)
                z = (np.log(truth) - log_mu) / sigma
                print(f"  {name:<12}: Truth={truth:>7.2f}, Prior=LogN(ln {mu_val:>5.1f}, {sigma:>4.1f}), Z={z:>5.2f}")

    print("\nLikelihood Sanity: PASS")

if __name__ == "__main__":
    verify_likelihood()
