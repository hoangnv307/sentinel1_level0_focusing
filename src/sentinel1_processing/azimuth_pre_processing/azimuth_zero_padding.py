"""Azimuth Zero-Padding from DAD Section 6.2.1."""

import numpy as np


def apply(block, fft_length):
    """Zero-pad an azimuth block to the forward FFT length."""
    data = np.asarray(block)
    if data.ndim != 2 or data.shape[0] > fft_length:
        raise ValueError("block must be 2-D and no longer than fft_length.")
    padded = np.zeros((fft_length, data.shape[1]), dtype=np.complex64)
    padded[:data.shape[0]] = data
    return padded


__all__ = ["apply"]
