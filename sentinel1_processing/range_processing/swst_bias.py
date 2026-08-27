"""SWST Bias Correction from DAD Section 6.1.3."""

import numpy as np


def correct(range_start_time_s, bias_s):
    """Add the configured SWST bias to range start time."""
    return np.asarray(range_start_time_s) + float(bias_s)


__all__ = ["correct"]
