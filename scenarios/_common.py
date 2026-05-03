"""End-to-end 14-day scenario runner — exercises `StepwisePlant`.

Each per-set script in this folder calls ``run_swat_scenario('A_healthy')``
(or 'B_amplitude_collapse', 'C_recovery', 'D_phase_shift',
'E_overtrained', 'F_sedentary'). The runner:

1. Reads the truth control schedule + initial state from
   `models.swat.simulation.scenario_presets()` (the in-repo source of
   truth — these match the canonical PARAM_SET_*/INIT_STATE_* values).
2. Builds a `StepwisePlant` with `DEFAULT_PARAMS` and the scenario init.
3. Drives the plant for `n_days * BINS_PER_DAY` bins under the daily
   piecewise-constant V_h / V_n / V_c schedule.
4. Applies per-bin Bernoulli dropout to HR + stress (sleep / steps
   preserved) via the small ``_apply_dropout`` helper below — pure
   numpy, no external deps.
5. Writes a packaged-style artefact to
   `swat_model_factory/outputs/swat/<scenario_key>/`:
   - `trajectory.npz` — (n_bins, 4) latents + per-bin V_h/V_n/V_c
   - `obs_HR.npz`, `obs_sleep.npz`, `obs_steps.npz`, `obs_stress.npz`
6. Prints a one-screen summary including end-of-trial T vs the spec's
   expected end-of-trial T (a quick sanity check that the plant's 14-day
   trajectory still lands in the right basin).

The dev repo is fully self-contained — it depends only on `smc2fc`
(declared in `pyproject.toml`) plus the standard JAX/numpy stack. The
plant + obs samplers + scenario_presets here are the same code path the
MPC's plant exercises in production; a regression here means a
regression in the closed-loop bench downstream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


# Anchor every path to the dev-repo root so the script works no matter
# what cwd it's invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Per-channel Bernoulli dropout — pure numpy, no external deps.
# Tiny helper (formerly imported from psim.scenarios.missing_data; inlined
# here so the dev repo is self-contained on `smc2fc` + JAX/numpy alone).
def _apply_dropout(obs_data: dict, channels, rate: float = 0.05, seed: int = 42):
    """Drop a fraction of observations per channel, in place.

    Parameters
    ----------
    obs_data : dict
        Per-channel dict ``{channel_name: {'t_idx': arr, …}}``. Other
        bin-aligned fields (e.g. ``obs_value``, ``obs_label``,
        ``log_value``, ``present_mask``) are sliced in step with
        ``t_idx`` so the per-bin records stay aligned.
    channels : iterable of str
        Channels to apply dropout to. Other channels untouched.
    rate : float in [0, 1)
        Per-bin dropout probability.
    seed : int
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    for ch in channels:
        if ch not in obs_data:
            continue
        d = obs_data[ch]
        if 't_idx' not in d or len(d['t_idx']) == 0:
            continue
        idx = d['t_idx']
        keep = rng.random(len(idx)) > rate
        d['t_idx'] = idx[keep]
        for key in list(d.keys()):
            if key == 't_idx':
                continue
            v = d[key]
            if hasattr(v, '__len__') and len(v) == len(keep):
                d[key] = np.asarray(v)[keep]
    return obs_data


# Expected end-of-trial T BASIN per the LaTeX spec.
#
# These are deliberately qualitative ('high' / 'low' / 'rising' / etc.)
# rather than specific numerical targets — the actual numbers drift
# with parameter tuning (the LaTeX's old "T* ≈ 0.55 / ≈ 1.0" expected
# values were calibrated for an older mu(E) form pre the V_h-anabolic
# structural fix; current healthy T* ≈ 0.84 with the dev repo's
# parameters). The basin classification, however, is invariant: a
# regression that flips a scenario from 'healthy' to 'collapsed'
# (or vice versa) means the dynamics or controls are broken.
#
# (basin_label, threshold_check, explanation)
#   - basin_label is what gets printed
#   - threshold_check(T_end) returns True iff the basin is correct
#   - explanation is the qualitative spec
EXPECTED_END_T = {
    'A_healthy':            ('high (>0.5)',  lambda T: T > 0.5,
                              'healthy basin equilibrium'),
    'B_amplitude_collapse': ('collapsed (<0.2)', lambda T: T < 0.2,
                              'low-V_h + high-V_n collapses E below E_crit'),
    'C_recovery':           ('rising past 0.5', lambda T: T > 0.5,
                              'rises from T_0=0.05 toward healthy T*'),
    'D_phase_shift':        ('collapsed (<0.2)', lambda T: T < 0.2,
                              '6-h jet-lag drives phase(V_c)=0, E below E_crit'),
    'E_overtrained':        ('reduced (<0.7)', lambda T: T < 0.7,
                              'high V_n damping flattens the rhythm'),
    'F_sedentary':          ('collapsed (<0.3)', lambda T: T < 0.3,
                              'low V_h kills the rhythm via gate term'),
}


def run_swat_scenario(
    scenario_key: str,
    *,
    n_days: int = 14,
    dropout_rate: float = 0.05,
    seed: int = 42,
) -> int:
    """Run one 14-day SWAT scenario end-to-end via StepwisePlant.

    Returns the conventional Unix exit code (0 = success).
    """
    # Lazy imports so this module is itself import-clean.
    from models.swat.simulation import scenario_presets, DEFAULT_PARAMS
    from models.swat._plant import StepwisePlant
    from models.swat._v_schedule import BINS_PER_DAY, DT_BIN_DAYS

    presets = scenario_presets(n_days)
    if scenario_key not in presets:
        raise KeyError(
            f"Unknown scenario {scenario_key!r}; "
            f"choices: {sorted(presets.keys())}")
    preset = presets[scenario_key]

    stride_bins = n_days * BINS_PER_DAY
    expected_basin, basin_check, expected_meaning = EXPECTED_END_T.get(
        scenario_key, ('?', lambda T: True, '(no spec)'))

    print(f"=== SWAT / {scenario_key} ===")
    print(f"  bins:    {stride_bins}  ({n_days}d × {BINS_PER_DAY}/d)")
    print(f"  dt:      {DT_BIN_DAYS * 24:.3f} h = {DT_BIN_DAYS * 24 * 60:.0f} min")
    print(f"  init:    W={preset['init'][0]:.3f}, Z={preset['init'][1]:.3f}, "
          f"a={preset['init'][2]:.3f}, T={preset['init'][3]:.3f}")
    print(f"  control: V_h={preset['v_h_daily'][0]:.2f}, "
          f"V_n={preset['v_n_daily'][0]:.2f}, "
          f"V_c={preset['v_c_daily'][0]:.2f} h  (constant across {n_days} days)")
    print(f"  expected basin: {expected_basin}  ({expected_meaning})")

    # ── 1. Build plant + run forward sim ─────────────────────────────
    print("\n[1/3] Forward-simulate plant + sample 4 channels via StepwisePlant.advance()")
    plant = StepwisePlant(
        truth_params=dict(DEFAULT_PARAMS),
        state=preset['init'].copy(),
        seed_offset=seed,
        dt=DT_BIN_DAYS,
    )
    out = plant.advance(
        stride_bins=stride_bins,
        V_h_daily=preset['v_h_daily'],
        V_n_daily=preset['v_n_daily'],
        V_c_daily=preset['v_c_daily'],
    )
    traj = out['trajectory']
    print(f"   trajectory: shape {traj.shape}")
    print(f"   W:  range [{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}], mean {traj[:, 0].mean():.3f}")
    print(f"   Z:  range [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}], mean {traj[:, 1].mean():.3f}")
    print(f"   a:  range [{traj[:, 2].min():.3f}, {traj[:, 2].max():.3f}], mean {traj[:, 2].mean():.3f}")
    print(f"   T:  start {traj[0, 3]:.3f}, end {traj[-1, 3]:.3f}, "
          f"mean {traj[:, 3].mean():.3f}")
    print(f"   obs samples: HR={len(out['obs_HR']['t_idx'])}, "
          f"sleep={len(out['obs_sleep']['t_idx'])}, "
          f"steps={len(out['obs_steps']['t_idx'])}, "
          f"stress={len(out['obs_stress']['t_idx'])}")

    # ── 2. Apply dropout to HR + stress ──────────────────────────────
    print(f"\n[2/3] Apply {dropout_rate * 100:.0f}% dropout on hr/stress (sleep, steps preserved)")
    obs_for_dropout = {
        'hr':     out['obs_HR'],
        'stress': out['obs_stress'],
    }
    _apply_dropout(obs_for_dropout,
                    channels=['hr', 'stress'],
                    rate=dropout_rate,
                    seed=seed + 200)
    print(f"   after dropout: HR={len(out['obs_HR']['t_idx'])}, "
          f"stress={len(out['obs_stress']['t_idx'])}")

    # ── 3. Save artefact ─────────────────────────────────────────────
    out_dir = _REPO_ROOT / 'outputs' / 'swat' / scenario_key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/3] Save artefact to {out_dir}")

    np.savez(
        out_dir / 'trajectory.npz',
        trajectory=traj,
        V_h_per_bin=out['V_h']['value'],
        V_n_per_bin=out['V_n']['value'],
        V_c_per_bin=out['V_c']['value'],
    )

    obs_dir = out_dir / 'obs'
    obs_dir.mkdir(exist_ok=True)
    np.savez(obs_dir / 'obs_HR.npz', **out['obs_HR'])
    np.savez(obs_dir / 'obs_sleep.npz', **out['obs_sleep'])
    np.savez(obs_dir / 'obs_steps.npz', **out['obs_steps'])
    np.savez(obs_dir / 'obs_stress.npz', **out['obs_stress'])

    print(f"\nDone. Artefact at: {out_dir}")
    end_T = float(traj[-1, 3])
    basin_ok = basin_check(end_T)
    verdict = "BASIN OK" if basin_ok else "BASIN MISMATCH — investigate plant/dynamics"
    print(f"  Achieved end-of-trial T: {end_T:.3f}  "
          f"(expected basin: {expected_basin})  →  {verdict}")
    return 0 if basin_ok else 1
