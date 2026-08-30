"""Azimuth Forward FFT from DAD Section 6.2.3."""

from scipy.fft import fft, fftfreq, fftshift


def apply(block, azimuth_sample_period_s):
    """Transform an azimuth block to the range-Doppler domain."""
    if azimuth_sample_period_s <= 0:
        raise ValueError("azimuth_sample_period_s must be positive.")
    frequency_hz = fftshift(
        fftfreq(block.shape[0], d=azimuth_sample_period_s)
    )
    range_doppler = fftshift(fft(block, axis=0), axes=0).astype("complex64")
    return frequency_hz, range_doppler


__all__ = ["apply"]
