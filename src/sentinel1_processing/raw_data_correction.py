"""Raw-data corrections from Sentinel-1 L1 DAD Section 9.2."""

import numpy as np


def estimate_iq_bias(radar_data):
    """Estimate the constant I/Q mean used by DAD Section 9.2.1."""
    data = np.asarray(radar_data)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("radar_data must be a non-empty 2-D array.")
    return complex(np.mean(data, dtype=np.complex128))


__all__ = ["estimate_iq_bias"]
