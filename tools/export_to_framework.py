"""SWAT Port-Package Exporter.

Automates the generation of a production-grade SWAT model package.
Validates the model against all factory pillars before bundling:
1. Identifiability (FIM)
2. Stability (Stiffness)
3. Reconciliation (Mirror)
4. Likelihood Sanity
5. Controller Sandbox
"""

import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np

# Enable JAX X64
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

# Anchor every path to the dev-repo root (the parent of `tools/`) so the
# script works no matter what cwd it's invoked from. Without this, an AI
# agent or a CI runner calling `python tools/export_to_framework.py` from
# any other directory would FileNotFoundError on the model-bundle copy.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Add model path to sys.path
sys.path.append(str(_REPO_ROOT))

# Import validation tools (using their main logic)
from tools.analyze_identifiability import compute_fim
from tools.audit_stiffness import audit_stiffness
from tools.verify_likelihood import verify_likelihood
from tools.verify_controller import verify_controller
from tests.test_reconciliation import test_mirror_reconciliation

FIM_DIAG_THRESHOLD = 1e-3  # parameters with FIM diag below this are not identifiable
                            # *from the channels the FIM tool currently covers*

# Parameters known to be identifiable from channels the FIM tool does NOT cover.
# The FIM tool's get_mean_obs only includes hr / steps / stress — the sleep
# channel is missing. c_tilde and delta_c are the sleep ordinal cutoffs and are
# therefore identifiable from sleep data even though their FIM diagonal here
# is zero by construction. Add them by hand until the FIM tool covers sleep.
SLEEP_CHANNEL_IDENTIFIABLE = ["c_tilde", "delta_c"]


def export_package(output_name="swat_v2_verified"):
    print(f"=== SWAT Port-Package Exporter: {output_name} ===\n")

    export_dir = _REPO_ROOT / "exports" / output_name
    export_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Pillar 1: Identifiability
    print("Checking Pillar I: Identifiability...")
    fim_a, names = compute_fim("A_healthy", n_days=1)
    fim_c, _ = compute_fim("C_recovery", n_days=1)
    fim_combined = fim_a + fim_c
    evals = np.linalg.eigvalsh(fim_combined)
    diag = np.diag(fim_combined)
    fim_identifiable = [n for n, d in zip(names, diag) if d > FIM_DIAG_THRESHOLD]
    identifiable_subset = sorted(set(fim_identifiable) | set(SLEEP_CHANNEL_IDENTIFIABLE))
    rank = int(np.sum(evals > 1e-8))
    results['identifiability'] = {
        'rank': rank,
        'cond': float(evals[-1] / (evals[0] + 1e-12)),
        'identifiable_from_fim': fim_identifiable,
        'identifiable_subset': identifiable_subset,
        'sleep_channel_added_manually': SLEEP_CHANNEL_IDENTIFIABLE,
        'n_identifiable': len(identifiable_subset),
        'fim_tool_coverage_note': "FIM tool covers hr/steps/stress only; sleep params added manually",
        'passed': bool(rank >= 5),
    }
    print(f"  Rank: {rank}, Cond: {results['identifiability']['cond']:.2e}")
    print(f"  Identifiable from FIM: {fim_identifiable}")
    print(f"  Plus manual (sleep channel): {SLEEP_CHANNEL_IDENTIFIABLE}")
    print(f"  Combined subset:       {identifiable_subset}")

    # Pillar 2: Stability
    print("\nChecking Pillar II: Stability...")
    stiff = audit_stiffness()
    results['stability'] = stiff

    # Pillar 3: Reconciliation
    print("\nChecking Pillar III: Reconciliation...")
    try:
        test_mirror_reconciliation()
        results['reconciliation'] = {'passed': True}
    except Exception as e:
        print(f"  FAILED: {e}")
        results['reconciliation'] = {'passed': False, 'error': str(e)}

    # Pillar 4: Likelihood
    print("\nChecking Pillar IV: Likelihood...")
    try:
        verify_likelihood()
        results['likelihood'] = {'passed': True}
    except Exception as e:
        print(f"  FAILED: {e}")
        results['likelihood'] = {'passed': False, 'error': str(e)}

    # Pillar 5: Sandbox
    print("\nChecking Pillar V: Sandbox...")
    try:
        verify_controller()
        results['controller'] = {'passed': True}
    except Exception as e:
        print(f"  FAILED: {e}")
        results['controller'] = {'passed': False, 'error': str(e)}

    # Final Gate
    all_passed = all(v['passed'] for v in results.values())
    
    if all_passed:
        print("\n=== ALL GATES PASSED. Bundling package... ===")
        
        # Bundle files
        model_src = _REPO_ROOT / "models" / "swat"
        dest_src = export_dir / "model"
        dest_src.mkdir(parents=True, exist_ok=True)

        for f in ["_dynamics.py", "_plant.py", "_v_schedule.py", "control.py", "estimation.py", "simulation.py", "__init__.py"]:
            shutil.copy(model_src / f, dest_src / f)
            
        # Manifest — values pulled from the actual checks above, not hard-coded
        manifest = {
            "model": "SWAT",
            "version": "2.0.0-verified",
            "validation": results,
            "export_config": {
                "h_max_mins": results['stability']['h_max_mins'],
                "n_substeps": results['stability']['n_substeps_for_15min'],
                "identifiable_subset": results['identifiability']['identifiable_subset'],
            }
        }
        
        with open(export_dir / "MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=4)
            
        print(f"\nPackage exported successfully to: {export_dir}")
    else:
        print("\n!!! EXPORT FAILED: One or more validation gates failed. !!!")
        print(json.dumps(results, indent=4))
        sys.exit(1)

if __name__ == "__main__":
    export_package()
