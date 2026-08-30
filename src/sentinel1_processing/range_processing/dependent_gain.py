"""Range Dependent Gain Correction from DAD Section 6.1.2."""

import numpy as np


def apply(range_compressed, gain):
    """Multiply each range sample by its gain value."""
    gain = np.asarray(gain)
    if gain.ndim != 1 or gain.size != range_compressed.shape[1]:
        raise ValueError("gain must contain one value per range sample.")
    range_compressed *= gain[None, :]
    return range_compressed


__all__ = ["apply"]
