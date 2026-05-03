#!/usr/bin/env python3
"""SWAT Set B — amplitude collapse: 14-day plant + obs scenario.

Truth: V_h=0.2, V_n=1.0, V_c=0.0 h, T_0=0.5.
Expected: low V_h + chronic load drives gate ≈ 0.121, E below E_crit,
T decays from 0.5 toward ~0.13 (hypogonadal-flatline basin).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402


def main():
    return run_swat_scenario('B_amplitude_collapse')


if __name__ == "__main__":
    sys.exit(main())
