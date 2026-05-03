"""Plant regression smoke test — drive `StepwisePlant` for 14 days under
each of the six canonical scenarios and assert the end-of-trial T lands
in the expected qualitative basin (high / collapsed / rising / etc.).

Why this test exists
--------------------
`_plant.py` (the simulator-as-plant) is what the closed-loop MPC drives
at each control-decision step. If `_plant.py`, `_dynamics.py`, or any of
the obs samplers in `simulation.py` quietly drift, the closed-loop bench
downstream will produce wrong results. This smoke test catches dramatic
drift early — before the bug ships.

The basin classifier (defined in `scenarios/_common.py:EXPECTED_END_T`)
is deliberately qualitative, not a fixed numerical target. Specific
numbers like "T* ≈ 0.55" come from older parameterizations and would
either silently hide model evolution or fail spuriously on every tune.
A basin classification ('high' vs 'collapsed' vs 'rising') is invariant
to those drifts and still loud if the dynamics genuinely invert.

Cost
----
Each scenario advances the plant for 14 days × 96 bins/day = 1344 bins
with 10× sub-stepping; on CPU each takes ~1.5 s, so the full 6-scenario
sweep adds ~10 s to the pytest run. Acceptable for a regression net
that protects the load-bearing plant.

The scenarios also write artefact files to `outputs/swat/<scenario>/`
as a side effect — this is intentional, the artefacts let you visually
inspect what the plant produced if a basin check fails.
"""
import os
import sys
from pathlib import Path

# Enable JAX X64 for tight float comparisons.
os.environ['JAX_ENABLE_X64'] = 'True'

# Force CPU. The test is fast enough on CPU and doesn't need a GPU
# context (which can add cuSolver-init noise in CI).
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import pytest

# Make the dev-repo root importable so `from scenarios._common import …`
# works even when pytest is invoked from outside the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scenarios._common import EXPECTED_END_T, run_swat_scenario


@pytest.mark.parametrize("scenario_key", sorted(EXPECTED_END_T.keys()))
def test_scenario_lands_in_expected_basin(scenario_key):
    """Each canonical scenario's end-of-trial T must land in the right basin.

    `run_swat_scenario` returns 0 if the basin matches, 1 otherwise.
    """
    rc = run_swat_scenario(scenario_key)
    assert rc == 0, (
        f"Plant regression: scenario {scenario_key!r} ended in the wrong "
        f"basin. Inspect outputs/swat/{scenario_key}/trajectory.npz to see "
        f"the 14-day trajectory; check whether _plant.py, _dynamics.py, "
        f"or the obs samplers in simulation.py have drifted.")
