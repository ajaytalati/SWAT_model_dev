#!/usr/bin/env python3
"""SWAT Set F — sedentary: 14-day plant + obs scenario.

Truth: V_h=0.2, V_n=0.2, V_c=0.0 h, T_0=0.5. Low fitness, no chronic
load. The gate term V_h · exp(-V_n/V_n_scale) ≈ 0.181 strongly
suppresses the W and Z amplitudes, so the entrainment quality stays
low and T collapses toward the flatline.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402


def main():
    return run_swat_scenario('F_sedentary')


if __name__ == "__main__":
    sys.exit(main())
