"""Sim ↔ Estimator observation-channel consistency.

The existing reconciliation test ([test_reconciliation.py]) only checks
that the plant and estimator agree on the **latent SDE dynamics**. It
does not exercise the four observation channels, so a sim/est mismatch
in the channel mean (or in the sleep marginal-probability formula) goes
undetected.

That gap is exactly how bugs D1 and D2 in
[bug_reports/2026-05-03_latex_vs_code_reconciliation.md] hid for a
while: the simulator's `gen_obs_hr` and `gen_obs_stress` were silently
omitting the subject-specific offsets `delta_HR` and `delta_s`, while
the estimator's likelihood included them. Any sandbox check that drove
those parameters to non-zero truth values would have failed silently.

This file fills that gap. For each of the four observation channels, it
asserts that the sim's noise-free predicted mean (or, for sleep, the
per-label marginal probability) matches the estimator's likelihood
prediction at the same state, parameters, and exogenous control. The
authority is `LaTex_docs/swat_model.tex` §3 — the formulas in the test
are the LaTeX formulas, written out in code.

If the sim or the estimator silently drifts away from the LaTeX, the
test fails.
"""
import math
import os
import sys

# Enable JAX X64 so the float comparisons are tight.
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Add model path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.swat.simulation import (
    DEFAULT_PARAMS,
    gen_obs_hr,
    gen_obs_sleep,
    gen_obs_steps,
    gen_obs_stress,
)
from models.swat.estimation import (
    FROZEN_PARAMS,
    PARAM_PRIOR_CONFIG,
    SWAT_ESTIMATION,
    _PI,
    obs_log_weight_fn,
)


# ─────────────────────────────────────────────────────────────────────
# Test fixtures — a single non-trivial scenario
# ─────────────────────────────────────────────────────────────────────
#
# Using non-zero subject offsets (delta_HR = +5 bpm, delta_s = +7 pts)
# is critical: with the D1/D2 bug, the sim ignored those offsets, so any
# test using delta_HR=delta_s=0 would not have caught it.

W = 0.4               # wakefulness
Z = 0.6               # sleep depth
a_state = 0.5
T_state = 0.4
V_n = 1.2

DELTA_HR_TRUTH = 5.0   # subject runs 5 bpm above population baseline
DELTA_S_TRUTH = 7.0    # subject reports 7 pts above population baseline


def _build_params() -> dict:
    """Truth params dict with non-zero subject offsets.

    Starts from DEFAULT_PARAMS (which now includes delta_HR=0 and
    delta_s=0 after the D1/D2 fix), overrides the offsets so the test
    actually exercises them.
    """
    p = dict(DEFAULT_PARAMS)
    p['delta_HR'] = DELTA_HR_TRUTH
    p['delta_s'] = DELTA_S_TRUTH
    return p


def _build_estimator_params_vec(p_dict: dict) -> jnp.ndarray:
    """Pack a flat params vector indexed by `_PI` for `obs_log_weight_fn`."""
    return jnp.array([p_dict[name] for name in _PI], dtype=jnp.float64)


def _est_log_lik_at(channel: str, obs_value, x_state, p_dict, V_n_at_bin) -> float:
    """Single-bin estimator log-weight, with only the named channel present.

    Builds a one-bin grid_obs that flips on exactly one channel's
    presence mask so the returned log-weight is the channel's likelihood
    in isolation. Lets us isolate one channel at a time.
    """
    grid_obs = {
        'V_h': jnp.array([1.0]),
        'V_n': jnp.array([V_n_at_bin]),
        'V_c': jnp.array([0.0]),
        'hr_value': jnp.array([0.0]),       'hr_present': jnp.array([0.0]),
        'stress_value': jnp.array([0.0]),   'stress_present': jnp.array([0.0]),
        'log_steps_value': jnp.array([0.0]),'steps_present': jnp.array([0.0]),
        'sleep_label': jnp.array([0]),       'sleep_present': jnp.array([0.0]),
    }
    if channel == 'hr':
        grid_obs['hr_value'] = jnp.array([float(obs_value)])
        grid_obs['hr_present'] = jnp.array([1.0])
    elif channel == 'steps':
        grid_obs['log_steps_value'] = jnp.array([float(obs_value)])
        grid_obs['steps_present'] = jnp.array([1.0])
    elif channel == 'stress':
        grid_obs['stress_value'] = jnp.array([float(obs_value)])
        grid_obs['stress_present'] = jnp.array([1.0])
    elif channel == 'sleep':
        grid_obs['sleep_label'] = jnp.array([int(obs_value)])
        grid_obs['sleep_present'] = jnp.array([1.0])
    else:
        raise ValueError(f"unknown channel {channel!r}")

    p_vec = _build_estimator_params_vec(p_dict)
    log_w = obs_log_weight_fn(jnp.asarray(x_state, dtype=jnp.float64),
                                grid_obs, k=0, params=p_vec)
    return float(log_w)


def _max_gauss_log_density(sigma: float) -> float:
    """log N(x | x, sigma^2) — the value of the log-density at its peak.

    Equals -log(sigma·sqrt(2π)).
    """
    return -math.log(sigma * math.sqrt(2.0 * math.pi))


# ─────────────────────────────────────────────────────────────────────
# Channel tests
# ─────────────────────────────────────────────────────────────────────
#
# Pattern for each Gaussian channel (HR, steps, stress):
#   1. Compute the LaTeX mean directly in the test (single source of truth).
#   2. Assert the simulator's noise-free output (sigma → 0) equals the
#      LaTeX mean.
#   3. Assert the estimator's log-likelihood, evaluated at obs = LaTeX
#      mean and at the truth state, equals the maximum possible value
#      -log(sigma·sqrt(2π)). That can only happen if the estimator's
#      predicted mean equals the LaTeX mean too.
#
# If either side drifts away from the LaTeX, exactly one of (2) or (3)
# fails, pinpointing which side broke.


def test_hr_channel_consistency():
    """HR channel: sim and estimator both produce LaTeX §3.1 mean."""
    p = _build_params()
    state = np.array([W, Z, a_state, T_state], dtype=np.float64)
    trajectory = state[None, :]                # one bin
    t_grid = np.array([0.0])

    # 1. LaTeX §3.1 mean
    expected_mean = (p['HR_base'] + p['delta_HR']
                     + p['alpha_HR'] * W)

    # 2. Simulator side — set sigma to ~0 so the sample IS the mean
    p_no_noise = dict(p, sigma_HR=0.0)
    out = gen_obs_hr(trajectory, t_grid, p_no_noise, seed=0)
    sim_mean = float(out['obs_value'][0])
    assert math.isclose(sim_mean, expected_mean, abs_tol=1e-12), (
        f"sim HR mean {sim_mean} != LaTeX mean {expected_mean}; "
        f"check `gen_obs_hr` formula against LaTeX §3.1 "
        f"(possibly missing delta_HR — bug D1)")

    # 3. Estimator side — log-likelihood at obs = expected_mean must
    #    hit the Gaussian peak -log(sigma·sqrt(2π)).
    est_lp = _est_log_lik_at('hr', expected_mean,
                              x_state=state, p_dict=p, V_n_at_bin=V_n)
    expected_peak = _max_gauss_log_density(p['sigma_HR'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-10), (
        f"estimator HR mean does not match LaTeX mean: log-lik at obs={expected_mean} "
        f"is {est_lp}, peak should be {expected_peak} "
        f"(residual = sqrt(-2·sigma²·(est_lp - peak)) = "
        f"{math.sqrt(max(0.0, -2.0 * p['sigma_HR']**2 * (est_lp - expected_peak)))})")


def test_steps_channel_consistency():
    """Steps channel: sim and estimator both produce LaTeX §3.3 mean."""
    p = _build_params()
    state = np.array([W, Z, a_state, T_state], dtype=np.float64)
    trajectory = state[None, :]
    t_grid = np.array([0.0])

    # 1. LaTeX §3.3 mean — log(steps + 1) ~ N(mu_step0 + beta_W_steps · W, sigma_step²)
    expected_mean = p['mu_step0'] + p['beta_W_steps'] * W

    # 2. Simulator side. gen_obs_steps requires a sleep_label vector
    #    (it only flips the present_mask based on it; the log_value is
    #    sampled regardless). Pass a wake bin so present_mask=1.
    p_no_noise = dict(p, sigma_step=0.0)
    out = gen_obs_steps(trajectory, t_grid, p_no_noise,
                          sleep_label=np.array([0], dtype=np.int32), seed=0)
    sim_mean = float(out['log_value'][0])
    assert math.isclose(sim_mean, expected_mean, abs_tol=1e-12), (
        f"sim steps mean {sim_mean} != LaTeX mean {expected_mean}; "
        f"check `gen_obs_steps` formula against LaTeX §3.3")

    # 3. Estimator side
    est_lp = _est_log_lik_at('steps', expected_mean,
                              x_state=state, p_dict=p, V_n_at_bin=V_n)
    expected_peak = _max_gauss_log_density(p['sigma_step'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-10), (
        f"estimator steps mean does not match LaTeX mean: log-lik {est_lp} "
        f"!= peak {expected_peak}")


def test_stress_channel_consistency():
    """Stress channel: sim and estimator both produce LaTeX §3.4 mean."""
    p = _build_params()
    state = np.array([W, Z, a_state, T_state], dtype=np.float64)
    trajectory = state[None, :]
    t_grid = np.array([0.0])

    # 1. LaTeX §3.4 mean
    expected_mean = (p['s_base'] + p['delta_s']
                     + p['alpha_s'] * W
                     + p['beta_s'] * V_n)

    # 2. Simulator side — guard the [0, 100] clip by picking a state
    #    that yields a mean comfortably inside the box (W=0.4, V_n=1.2:
    #    30+7+40·0.4+10·1.2 = 65, well inside).
    p_no_noise = dict(p, sigma_s=0.0)
    out = gen_obs_stress(trajectory, t_grid, p_no_noise,
                          V_n_per_bin=np.array([V_n], dtype=np.float64),
                          seed=0)
    sim_mean = float(out['obs_value'][0])
    assert math.isclose(sim_mean, expected_mean, abs_tol=1e-12), (
        f"sim stress mean {sim_mean} != LaTeX mean {expected_mean}; "
        f"check `gen_obs_stress` formula against LaTeX §3.4 "
        f"(possibly missing delta_s — bug D2)")

    # 3. Estimator side
    est_lp = _est_log_lik_at('stress', expected_mean,
                              x_state=state, p_dict=p, V_n_at_bin=V_n)
    expected_peak = _max_gauss_log_density(p['sigma_s'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-10), (
        f"estimator stress mean does not match LaTeX mean: log-lik {est_lp} "
        f"!= peak {expected_peak}")


def test_sleep_channel_marginals():
    """Sleep channel: sim's per-label marginals match estimator's at the first bin.

    LaTeX §3.2:
        c1 = c_tilde,  c2 = c_tilde + delta_c
        P(0|Z) = 1 - sigmoid(sharp · (Z - c1))
        P(1|Z) = sigmoid(sharp · (Z - c1)) - sigmoid(sharp · (Z - c2))
        P(2|Z) = sigmoid(sharp · (Z - c2))

    The sticky HMM only kicks in from bin 1 onward, so checking bin 0
    (where _ordinal_log_lik returns log(p_marg[label])) directly tests
    the marginal formula.
    """
    p = _build_params()
    state = np.array([W, Z, a_state, T_state], dtype=np.float64)

    # 1. LaTeX marginals — written out here as a third source of truth.
    sharp = float(FROZEN_PARAMS['sleep_sharpness'])
    c1 = p['c_tilde']
    c2 = c1 + p['delta_c']
    s1 = 1.0 / (1.0 + math.exp(-sharp * (Z - c1)))
    s2 = 1.0 / (1.0 + math.exp(-sharp * (Z - c2)))
    p_marg_expected = np.array([1.0 - s1, s1 - s2, s2], dtype=np.float64)

    # Sanity: marginals sum to 1.
    assert math.isclose(p_marg_expected.sum(), 1.0, abs_tol=1e-12)

    # 2. Estimator side — for k=0 (first bin) the sticky-HMM falls back
    #    to p_marg, so log-lik at label k equals log(p_marg[k]).
    for label in (0, 1, 2):
        est_lp = _est_log_lik_at('sleep', label,
                                  x_state=state, p_dict=p, V_n_at_bin=V_n)
        expected_lp = math.log(p_marg_expected[label])
        assert math.isclose(est_lp, expected_lp, abs_tol=1e-12), (
            f"estimator sleep marginal disagrees with LaTeX for label={label}: "
            f"got log-prob {est_lp}, expected {expected_lp} "
            f"(sim P_marg = {p_marg_expected[label]:.6f})")

    # 3. Simulator side — the sim's per-bin marginal P_marg formula is
    #    inside `gen_obs_sleep`. Test it indirectly: with the
    #    sticky-HMM disabled (tau_persist tiny → P_stay≈0), labels are
    #    drawn iid from p_marg per bin. Empirical frequencies over many
    #    bins must match p_marg.
    n_bins = 20000
    p_iid = dict(p, tau_sleep_persist_h=1e-6)   # disable stickiness
    trajectory = np.tile(state[None, :], (n_bins, 1))
    t_grid = np.arange(n_bins, dtype=np.float64) / 96.0   # 15-min bins
    out = gen_obs_sleep(trajectory, t_grid, p_iid, seed=12345)
    counts = np.bincount(out['obs_label'], minlength=3).astype(np.float64)
    p_marg_emp = counts / counts.sum()
    # Multinomial 3-sigma bound on a single-cell estimate at this n is
    # roughly 3·sqrt(p(1-p)/n) ≤ 3·sqrt(0.25/20000) ≈ 0.011.
    assert np.allclose(p_marg_emp, p_marg_expected, atol=0.015), (
        f"sim sleep marginals disagree with LaTeX: empirical "
        f"{p_marg_emp} vs expected {p_marg_expected}")


# ─────────────────────────────────────────────────────────────────────
# Subject-offset specificity test
# ─────────────────────────────────────────────────────────────────────
#
# Belt-and-braces: explicitly verify that bumping `delta_HR` and
# `delta_s` shifts the simulated channels by exactly that amount.
# This is the test that would have failed loudest under D1 / D2.

def test_subject_offsets_actually_shift_the_signal():
    """Sweep delta_HR and delta_s, check observed mean shifts 1:1."""
    p_zero = dict(DEFAULT_PARAMS)
    p_zero['delta_HR'] = 0.0
    p_zero['delta_s'] = 0.0

    state = np.array([W, Z, a_state, T_state], dtype=np.float64)
    trajectory = state[None, :]
    t_grid = np.array([0.0])

    for shift in (-7.0, -2.0, 3.0, 11.0):
        # HR
        p = dict(p_zero, delta_HR=shift, sigma_HR=0.0)
        hr_shifted = float(
            gen_obs_hr(trajectory, t_grid, p, seed=0)['obs_value'][0])
        hr_baseline = float(
            gen_obs_hr(trajectory, t_grid, dict(p_zero, sigma_HR=0.0),
                        seed=0)['obs_value'][0])
        assert math.isclose(hr_shifted - hr_baseline, shift, abs_tol=1e-12), (
            f"delta_HR={shift} bpm should shift HR mean by exactly {shift} bpm; "
            f"got shift {hr_shifted - hr_baseline} (regression of bug D1)")

        # Stress
        p = dict(p_zero, delta_s=shift, sigma_s=0.0)
        s_shifted = float(gen_obs_stress(
            trajectory, t_grid, p,
            V_n_per_bin=np.array([V_n]), seed=0)['obs_value'][0])
        s_baseline = float(gen_obs_stress(
            trajectory, t_grid, dict(p_zero, sigma_s=0.0),
            V_n_per_bin=np.array([V_n]), seed=0)['obs_value'][0])
        assert math.isclose(s_shifted - s_baseline, shift, abs_tol=1e-12), (
            f"delta_s={shift} should shift stress mean by exactly {shift}; "
            f"got shift {s_shifted - s_baseline} (regression of bug D2)")


if __name__ == "__main__":
    # Allow running with `python tests/test_obs_consistency.py`.
    test_hr_channel_consistency()
    test_steps_channel_consistency()
    test_stress_channel_consistency()
    test_sleep_channel_marginals()
    test_subject_offsets_actually_shift_the_signal()
    print("All sim/est observation-channel consistency tests PASSED.")
