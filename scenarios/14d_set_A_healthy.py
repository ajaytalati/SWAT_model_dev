#!/usr/bin/env python3
"""SWAT Set A — healthy basin: 14-day plant + obs scenario.

Truth: V_h=1.0, V_n=0.2, V_c=0.0 h, T_0=0.5.
Expected: T equilibrates near T* ≈ 0.55 over 14 days.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402


def main():
    return run_swat_scenario('A_healthy')


if __name__ == "__main__":
    sys.exit(main())
