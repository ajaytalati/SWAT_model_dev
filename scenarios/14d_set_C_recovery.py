#!/usr/bin/env python3
"""SWAT Set C — recovery scenario: 14-day plant + obs.

Truth: V_h=1.0, V_n=0.2, V_c=0.0 h, T_0=0.05 (start near flatline).
Expected: with healthy controls and pathological initial T, T should
rise from 0.05 toward T* ≈ 1.0 over 14 days.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402


def main():
    return run_swat_scenario('C_recovery')


if __name__ == "__main__":
    sys.exit(main())
