# SWAT Model Factory

## What this folder is for

This is a **safe place to develop and test the SWAT model on its own**, before any of its files get pulled into the bigger `smc2fc` framework.

The idea: bugs in a model are much easier to find when the model is sitting by itself, with nothing else going on, than when it's wired up inside the full filtering-and-control pipeline. So:

1. You change something in the SWAT model here.
2. You run the checks in this folder (do the numbers add up? are the parameters identifiable? is the simulator stable? do the plant and the estimator agree bit-for-bit?).
3. Only once everything passes do those files get copied into `smc2fc` (or its sibling folders like `version_2/models/swat/`) for the real benchmark runs.

Think of it as the workshop where the model gets built and inspected, not the factory floor where it gets used in production.

## What gets checked here

Four kinds of check. Each one catches a different class of bug:

1. **Are the parameters identifiable?** (`tools/analyze_identifiability.py`) — computes the Fisher information from the four observation channels and looks at its rank. If the rank is too low, it means some parameters can't be told apart from data, no matter how much data you have. Fix the model before going further.

2. **Is the simulator numerically stable?** (`tools/audit_stiffness.py`) — works out how small the time step has to be so the integrator doesn't blow up. The answer is written into the export manifest so the framework knows what step size it can safely use.

3. **Do the plant and the estimator agree?** (`tests/test_reconciliation.py`) — the "plant" (what generates the data) and the "estimator" (what tries to recover the state from the data) share one dynamics file. This test runs both and checks that they produce the exact same numbers. If they drift apart, the filter is silently estimating the wrong thing.

4. **Does the likelihood / controller still work?** (`tools/verify_likelihood.py`, `tools/verify_controller.py`) — quick sanity tests on the bits that the SMC² filter and the MPC controller will call.

## What lives where

- `models/swat/` — the actual model files (dynamics, simulation, estimation, plant, control). These are the files that eventually get copied out to the framework.
- `tools/` — the four checks above, plus `export_to_framework.py` which runs every check and only then bundles the model files into `exports/`.
- `tests/` — the plant-vs-estimator agreement test.
- `scenarios/` — six reference scenarios (healthy, amplitude collapse, recovery, phase shift, overtrained, sedentary). Used to see whether new model changes still produce the expected biological behaviour.
- `exports/` — verified model bundles, with a `MANIFEST.json` recording what passed and the safe step size.

## Installation

This repo declares `smc2fc` (the framework) as its only project-level dependency — it's a personal project on github with its own `pyproject.toml`. Pip resolves it from there.

```bash
# Inside whichever Python env you want (conda, venv, …):
pip install -e ".[test]"
```

That single command:
1. Clones and installs `smc2fc` from `github.com/ajaytalati/python-smc2-filtering-control` (master branch).
2. Pulls in jax / numpy / matplotlib / scipy / blackjax / diffrax via `smc2fc`'s own dependency list.
3. Adds pytest for running the test suite.

Bug fixes in `smc2fc` are picked up by `pip install -e . --upgrade` — no vendored copy to keep in sync.

If you're working on `smc2fc` locally and want this dev repo to see your in-progress edits, install it in editable mode first and pip will respect that:
```bash
pip install -e /path/to/python-smc2-filtering-control
pip install -e ".[test]"   # picks up the local version
```

## How to run things

From inside this folder, with the env active:

```bash
# Run the plant/estimator and sim/estimator agreement tests (most important quick check):
JAX_ENABLE_X64=True pytest tests/ -v

# Run all five pillars and write a verified bundle to exports/:
python tools/export_to_framework.py

# Inspect identifiability or stiffness on their own:
python tools/analyze_identifiability.py
python tools/audit_stiffness.py
```

The `PYTHONPATH=.` prefix used in older versions of this README is no longer required after `pip install -e .` — the four subpackages (`models`, `tools`, `tests`, `scenarios`) are made importable by the install. If you'd rather not install at all, the `PYTHONPATH=.` workflow still works.
