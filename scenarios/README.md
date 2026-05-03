# SWAT 14-day reference scenarios

End-to-end plant + observation runs for the six canonical SWAT
scenarios. Each script drives `StepwisePlant.advance()` for 14 days
under a fixed daily V_h / V_n / V_c control schedule, samples all four
observation channels, applies 5% Bernoulli dropout to HR + stress,
and writes a packaged artefact.

These are the **plant regression tests** — if anything in `_plant.py`,
`_dynamics.py`, or the obs samplers in `simulation.py` quietly drifts,
at least one scenario's end-of-trial T will land in the wrong basin
(e.g. `A_healthy` collapsing to zero, or `B_amplitude_collapse`
sustaining like a healthy basin).

## Running

From inside this folder, with the env active (`pip install -e .`):

```bash
python 14d_set_A_healthy.py
python 14d_set_B_amplitude.py
python 14d_set_C_recovery.py
python 14d_set_D_phase_shift.py
python 14d_set_E_overtrained.py
python 14d_set_F_sedentary.py
```

Each writes its artefact to `outputs/swat/<scenario_key>/`:

- `trajectory.npz` — `(n_bins, 4)` latent trajectory + per-bin
  `V_h_per_bin`, `V_n_per_bin`, `V_c_per_bin` arrays
- `obs/obs_HR.npz` — `t_idx`, `obs_value` (after dropout)
- `obs/obs_sleep.npz` — `t_idx`, `obs_label` (3-level ordinal)
- `obs/obs_steps.npz` — `t_idx`, `log_value`, `present_mask`
- `obs/obs_stress.npz` — `t_idx`, `obs_value` (after dropout)

At the end the script prints a basin verdict (`BASIN OK` /
`BASIN MISMATCH`) and exits 0 if the basin matches, 1 if it doesn't.

## The six scenarios

Per `models/swat/simulation.py:scenario_presets()`. All run for 14 days
at 15-min resolution (`BINS_PER_DAY = 96` → 1344 bins).

| Set | V_h | V_n | V_c | T_0 | Expected basin | Behaviour |
|-----|-----|-----|-----|-----|----------------|-----------|
| **A_healthy** | 1.0 | 0.2 | 0.0 h | 0.5 | high (>0.5)        | healthy basin equilibrium |
| **B_amplitude_collapse** | 0.2 | 1.0 | 0.0 h | 0.5 | collapsed (<0.2)   | low V_h + chronic load → E < E_crit |
| **C_recovery** | 1.0 | 0.2 | 0.0 h | 0.05 | rising past 0.5 | recovery from flatline → healthy basin |
| **D_phase_shift** | 1.0 | 0.2 | 6.0 h | 0.5 | collapsed (<0.2) | 6-h jet-lag drives phase(V_c)=0 |
| **E_overtrained** | 1.0 | 1.0 | 0.0 h | 0.5 | reduced (<0.7) | high V_n damping flattens the rhythm |
| **F_sedentary** | 0.2 | 0.2 | 0.0 h | 0.5 | collapsed (<0.3) | low V_h kills the rhythm via gate |

Sets B and D test two **independent** failure modes — amplitude (gate
collapse) vs phase (entrainment phase mis-alignment) — that exercise
the Stuart-Landau bifurcation in `T` from different angles. Set C is
the time-reversed Set B (recovery from flatline). Sets E and F test
two more pathological corners: over-training (high V_n damping) and
sedentary (low V_h shrinks the gate).

## Mixed-likelihood obs model

SWAT's four channels exercise the full mixed-likelihood discipline:

| Channel | Likelihood | Time grid | Drives |
|---------|------------|-----------|--------|
| `hr` | Gaussian | dense (every 15 min) | W |
| `sleep` | 3-level ordinal {0=wake, 1=light+REM, 2=deep} | dense | Z |
| `steps` | log-Gaussian, wake-gated | sparse (only when `sleep_label == 0`) | W |
| `stress` | Gaussian | dense | W, V_n |

The per-channel sim ↔ estimator consistency is asserted in
[../tests/test_obs_consistency.py](../tests/test_obs_consistency.py) —
that's the per-bin spec contract. The 14-day end-to-end basin check is
the regression net here.

## Running as part of the test suite

The same six scenarios are also run automatically by
[../tests/test_plant_regression_scenarios.py](../tests/test_plant_regression_scenarios.py)
(parametrized over the six keys). So `pytest tests/ -v` will catch
plant regressions even if you don't run the scripts manually.
