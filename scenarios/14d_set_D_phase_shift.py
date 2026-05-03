#!/usr/bin/env python3
"""SWAT Set D — phase-shift pathology (chronic shift work / jet lag).

Truth: V_h=1.0, V_n=0.2, V_c=6.0 h, T_0=0.5.
Expected: healthy potentials but rhythm 6 h delayed → phase(V_c)
saturates to 0 (since |V_c| > V_c_max=3 h) → entrainment quality E
drops below E_crit → T decays to ~0.11 over 14 days. The fourth
failure mode that (V_h, V_n) alone cannot produce.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402


def main():
    return run_swat_scenario('D_phase_shift')


if __name__ == "__main__":
    sys.exit(main())
