# LaTeX-vs-code reconciliation — SWAT model factory

**Generated:** 2026-05-03 21:37:42 BST
**Authority document:** [LaTex_docs/swat_model.tex](../LaTex_docs/swat_model.tex)
**Scope:** every concrete claim in the LaTeX checked against the code in `swat_model_factory/models/swat/`
**Method:** read each LaTeX section end-to-end; read each code file end-to-end; compare line-by-line; flag — do not fix.

## Triage status (as of 2026-05-03 22:02 BST)

| ID | Status | Resolution |
|----|--------|------------|
| D1 | **FIXED in code** | `gen_obs_hr` now includes `delta_HR` (CHANGELOG 21:45 BST) |
| D2 | **FIXED in code** | `gen_obs_stress` now includes `delta_s` (CHANGELOG 21:45 BST) |
| D3 | **FIXED in LaTeX** | code is correct (no V_c regularization); LaTeX §8.2 updated (CHANGELOG 22:02 BST) |
| D4 | **FIXED in LaTeX** | code is correct (sticky-HMM IS in estimator, fully-observed regime); LaTeX §3.2 / §6.3 updated |
| D5 | **FIXED in LaTeX** | code is correct (GK-DPF IS implemented); LaTeX §6.1 / §6.3 updated |
| D6 | **FIXED in LaTeX** | code is correct (V_h ∈ [0,4], V_n ∈ [0,5]); LaTeX §1.2 updated |
| D7 | **FIXED in LaTeX** | code is correct (MPC uses n_substeps=4 for speed); LaTeX §6.1 updated |
| D8 | **FIXED in LaTeX** | V_c_max = 3 h documented in §1.2, §2.2, §5 |
| D9 | **FIXED in LaTeX** | φ_0 = -π/3 documented in §2.1 and §5 |
| D10 | **OPEN** | stale docstring in `control.py` and `_dynamics.py` referring to obsolete `μ(E) = μ_0 + μ_E·E` form; not blocking |
| D11 | **OPEN** | silent stress clip to [0, 100] not in spec; not blocking at typical operating points |
| `scenarios/_common.py` broken `SWAT_MODEL` import | **FIXED** | Rewrote the runner to use `StepwisePlant.advance()` directly via the dev repo's leaner API instead of psim's `SDEModel`/`synthesise_scenario`. All 6 scenarios now run end-to-end as plant regression tests. (CHANGELOG 2026-05-03 22:32 BST) |
| D12 | **FIXED in LaTeX** | default truth values for free params now documented in §5 subsection |

For each fixed item the original write-up below is preserved unchanged for historical reference.

The original report flagged each issue as if the code might be wrong. Senior decision (Ajay, 2026-05-03 21:55 BST): for D3/D4/D5/D6/D7/D8/D9/D12 the code is correct and the LaTeX was the side that needed updating.

---

# Discrepancies between LaTeX spec and code

## High-severity — affect numerical results

### D1. Sim vs estimator asymmetry: HR mean missing `δ_HR` in the simulator

- **LaTeX §3.1**: `HR ~ N(HR_base + δ_HR + α_HR · W, σ_HR²)`
- **Estimator** ([estimation.py:355](../models/swat/estimation.py#L355)): `hr_mean = (p['HR_base'] + p['delta_HR']) + p['alpha_HR'] * W` ✓
- **Simulator** ([simulation.py:185](../models/swat/simulation.py#L185)): `hr_mean = params['HR_base'] + params['alpha_HR'] * W` — **`delta_HR` is missing**

The simulator generates HR observations without the subject-specific offset, but the estimator's likelihood includes it. This means the truth value of `δ_HR` is effectively 0 in any synthesized data, so identifiability tests on `δ_HR` will look funny.

### D2. Same problem for stress: `δ_s` missing in the simulator

- **LaTeX §3.4**: `S ~ N(s_base + δ_s + α_s · W + β_s · V_n, σ_s²)`
- **Estimator** ([estimation.py:382](../models/swat/estimation.py#L382)): includes `delta_s` ✓
- **Simulator** ([simulation.py:314-316](../models/swat/simulation.py#L314-L316)): `delta_s` missing

Same shape of bug as D1.

### D3. The cost functional is missing the V_c regularization term

- **LaTeX §8.2**: `J(θ) = E[ -∫(T + λ_E · E) dt + λ_c ∫(V_c/12)² dt ]`
- **Code** ([control.py:210](../models/swat/control.py#L210)): `return -(T_acc + lambda_E_jax * E_acc)`

The `λ_c ∫(V_c/12)² dt` quadratic penalty on circadian disruption is not in the code at all. There is no `lambda_c` parameter in `build_control_spec`. The MPC will currently propose larger phase shifts than the LaTeX's design intends.

### D4. The estimator IS using the sticky-HMM, but the LaTeX says it isn't

- **LaTeX §3.2** "Note on Inference Asymmetry": *"While the generative model uses this sticky-HMM persistence for physiological realism, the SMC² estimator treats the labels as conditionally independent given Z (using only P_marg)"*
- **LaTeX §6.3** "Likelihood Evaluation": same claim — *"intentionally dropping the generative sticky-HMM persistence for computational tractability"*
- **LaTeX §6.3 (Required Upgrades)**: lists "Exact Sticky-HMM Sleep Inference" as a TODO
- **Code** ([estimation.py:317-337](../models/swat/estimation.py#L317-L337)): `_ordinal_log_lik` actually does `p_sticky = P_stay * is_same + (1 - P_stay) * p_marg` using the previous bin's sleep label

The code is doing the sticky HMM (specifically the "Fully Observed Regime" from LaTeX §6.3, where `k_prev` is read from the previous bin's data). The "asymmetry" the LaTeX advertises is not what's in the code. Either (a) the code is ahead of the LaTeX (the upgrade got done, the LaTeX wasn't updated), or (b) the upgrade was done in the wrong place. This needs senior judgement.

### D5. The propagate_fn is GK-DPF (guided), but the LaTeX says it's a pure bootstrap filter

- **LaTeX §6.1**: *"No Guided Proposal: Unlike the FSA-v2 implementation which uses a Guided Kalman-Density Particle Filter (GK-DPF) to fuse Gaussian channels during propagation, the SWAT implementation relies on a pure bootstrap filter."*
- **LaTeX §6.1**: *"Because the proposal matches the prior dynamics, the predictive log-weight is zero."*
- **LaTeX §6.3 (Required Upgrades)**: lists GK-DPF as a TODO
- **Code** ([estimation.py:185-286](../models/swat/estimation.py#L185-L286)): a full GK-DPF — Kalman fusion of HR / Stress / Steps to build a guided proposal, with `pred_lw = log_pred_total - obs_ll_new` (line 284, **not** zero). The function's own docstring (line 187) says *"GK-DPF for SWAT 4-state."*

Same flavour as D4. The code is ahead of the LaTeX, or the LaTeX section is out of date. It's a substantive contradiction either way.

---

## Medium-severity — internal inconsistencies in the LaTeX itself

### D6. LaTeX contradicts itself on `V_h` and `V_n` ranges

- **LaTeX §1.2** (state space): `V_h ∈ [0, 1]`, `V_n ≥ 0` (no upper bound)
- **LaTeX §8.1** (MPC): `V_h ∈ [0, 4]`, `V_n ∈ [0, 5]`
- **Code** ([_v_schedule.py:43-45](../models/swat/_v_schedule.py#L43-L45)): `V_H_BOUNDS = (0, 4)`, `V_N_BOUNDS = (0, 5)`

The code matches §8.1, not §1.2. §1.2 needs to be updated to `V_h ∈ [0, 4]` and `V_n ∈ [0, 5]` for internal consistency.

### D7. MPC cost rollout uses `n_substeps = 4`, not 10

- **LaTeX §6.1 / §7.2**: `n_substeps = 10` (for stiff Z dynamics)
- **Filter** ([estimation.py:194](../models/swat/estimation.py#L194)): 10 ✓
- **Plant** ([_plant.py:78](../models/swat/_plant.py#L78)): 10 ✓
- **MPC controller** ([control.py:240](../models/swat/control.py#L240)): default `n_substeps = 4` for the cost-rollout integrator

The cost rollouts inside the MPC are integrated more coarsely than the plant and filter. Could be a deliberate speed/accuracy trade-off (10× scan over 14-day horizons × N_inner trials gets expensive), but it's not noted anywhere and the LaTeX says 10.

---

## Low-severity — undocumented constants / stale comments

### D8. `V_c_max = 3.0 hours` in code but not specified in the LaTeX

- **LaTeX §2.2**: `phase(V_c) = cos(π · min(|V_c|, V_c_max) / (2 V_c_max))` — leaves `V_c_max` unspecified
- **Code** ([_dynamics.py:81](../models/swat/_dynamics.py#L81)): `V_C_MAX_HOURS = 3.0`

Combined with §1.2 / §8.1 (`V_c ∈ [-12, 12]`), this means **any phase shift past ±3 hours produces `phase = 0`, fully killing the entrainment quality term `E`**. That's a load-bearing modelling choice — it should be in the LaTeX.

### D9. `φ_0 = -π/3` in code but not specified in the LaTeX

- **LaTeX §2.1**: `C_eff(t, V_c) = sin(2π(t - V_c/24) + φ_0)` — leaves `φ_0` unspecified
- **Code** ([_dynamics.py:80](../models/swat/_dynamics.py#L80)): `PHI_0_FROZEN = -π/3` (= morning chronotype)

Should be added to the LaTeX's frozen-parameters list (§5).

### D10. Stale docstring in `control.py` uses obsolete `μ(E) = μ_0 + μ_E · E` form

- **LaTeX §2.1**: `μ(E) = μ_E · (E - E_crit)`
- **Actual code** ([_dynamics.py:252](../models/swat/_dynamics.py#L252)): `mu = mu_E * (E_dyn - E_crit)` ✓ matches LaTeX
- **Stale docstring** in [control.py:16](../models/swat/control.py#L16): says `μ(E) = μ_0 + μ_E · E` and references `E_crit = -μ_0/μ_E = 0.5`

Same stale form also appears in [_dynamics.py:57](../models/swat/_dynamics.py#L57) (the module docstring).

The actual code is correct — it's only the explanatory text that's out of date.

### D11. Stress channel is silently clipped to [0, 100]

- **LaTeX §3.4**: `S ~ N(...)` — no clipping mentioned
- **Code** ([simulation.py:318](../models/swat/simulation.py#L318)): `stress = np.clip(stress, 0.0, 100.0)`

If the truth is anywhere near the boundary, the clipping makes the observed channel non-Gaussian, but the estimator (line 382) still uses an exact Gaussian likelihood. Probably harmless at typical operating points (mean ≈ 30 + 40·0.5 + 10·V_n ≈ 50, σ_s ≈ 15 — clipping only kicks in 3σ out) but not in the spec.

### D12. `DEFAULT_PARAMS` includes a default for `α_T` and `V_n_scale`, which the LaTeX classifies as free parameters

- **LaTeX §4.1**: `α_T` and `V_n_scale` are estimable
- **Code** ([_dynamics.py:115, 126](../models/swat/_dynamics.py#L115)): `TRUTH_PARAMS` sets `alpha_T=0.3` and `V_n_scale=2.0`

These are defaults used as ground truth for synthetic data generation — they don't override the priors, since the estimator explicitly excludes anything in `PARAM_PRIOR_CONFIG` from `FROZEN_PARAMS`. So this is consistent (the truth values just have to live somewhere for synthesis), but it's worth noting that the LaTeX doesn't tell the reader where the simulator gets these values from. A "default truth values for synthetic-data scenarios" subsection could clarify.

---

# What matched cleanly (everything not above)

To be explicit about what was checked and found consistent:

- **All 4 drift equations** (§2.1) — match [_dynamics.py:246-253](../models/swat/_dynamics.py#L246-L253)
- **All entrainment-quality components** (§2.2) — match [_dynamics.py:172-190](../models/swat/_dynamics.py#L172-L190)
- **Diffusion structure** (§2.3) — Jacobi for W/Z/a, additive for T — matches [_dynamics.py:284-289](../models/swat/_dynamics.py#L284-L289)
- **All 11 prior distributions** (§4) — exact match in [estimation.py:111-134](../models/swat/estimation.py#L111-L134)
- **All 12 frozen-parameter values** (§5) — exact match across [_dynamics.py:94-127](../models/swat/_dynamics.py#L94-L127) and [estimation.py:68-102](../models/swat/estimation.py#L68-L102)
- **State domain** (W, Z, a ∈ [0,1], T ≥ 0) — matches `state_clip` in [_dynamics.py:294-316](../models/swat/_dynamics.py#L294-L316)
- **Plant architecture** (§7) — sub-stepping, stochastic injection per sub-step, state_clip per sub-step, scan compilation, causal obs ordering HR → sleep → steps → stress, deterministic seed offsets — all match [_plant.py](../models/swat/_plant.py)
- **MPC action space and RBF parameterization** (§8.1) — bounds match, n_anchors=8 matches, all three squashing functions (4·σ, 5·σ, 12·tanh) match exactly

---

# What was NOT verified

Per the junior-engineer rule — flagging what's outside scope:

- **§6.4 SF-bridge warm-start** — `shard_init_fn` returns `global_init` (i.e. cold-start every window). The actual SF-bridge logic presumably lives in `smc2fc/core/sf_bridge.py` per the project CLAUDE.md, but that file was not read to confirm whether it gets called for SWAT.
- **§8.3 Tempered control posterior + adaptive tempering + HMC mutation** — framework-level (`smc2fc.control...`), not opened.
- **§6.3 "Sinkhorn OT resampling" and "Liu-West shrinkage"** — same — framework-level, not checked.

If verifying these is wanted, it'd mean reading into the `smc2fc/` package outside this folder.

---

# Suggested next steps

The discrepancies cluster into three groups:

1. **Code bugs** (D1, D2, D3) — the simulator and the controller are missing terms that the LaTeX requires. These should be code changes.
2. **The LaTeX is out of date** (D4, D5, possibly D7) — the code has implemented things the LaTeX still describes as TODOs.
3. **LaTeX is incomplete** (D6, D8, D9, D10, D11, D12) — internal contradictions or missing details that should be added to the LaTeX.

Senior decision required before any code or LaTeX edits.
