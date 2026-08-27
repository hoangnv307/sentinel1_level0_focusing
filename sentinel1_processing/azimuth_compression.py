"""Azimuth Compression from Sentinel-1 L1 DAD Section 6.3.4."""

import numpy as np
from scipy.fft import ifft, ifftshift


def fm_rate_magnitude(range_m, velocity_mps, doppler_centroid_hz, wavelength_m):
    """Calculate the azimuth FM-rate magnitude from DAD Eq. 9-22."""
    d = np.sqrt(np.maximum(
        1.0 - (
            wavelength_m * doppler_centroid_hz / (2.0 * velocity_mps)
        ) ** 2,
        0.0,
    ))
    return 2.0 * velocity_mps**2 * d**3 / (wavelength_m * range_m)


def calculate_matched_filter(
    azimuth_frequency_hz,
    slant_ranges_m,
    effective_velocity_mps,
    wavelength_m,
):
    """Calculate the azimuth matched filter from DAD §6.3.4.1."""
    d = np.sqrt(np.maximum(
        1.0 - (
            wavelength_m
            * azimuth_frequency_hz
            / (2.0 * effective_velocity_mps[None, :])
        ) ** 2,
        0.0,
    ))
    return np.exp(
        4.0j * np.pi * slant_ranges_m[None, :] * d / wavelength_m
    )


def calculate_time_correction_filter(
    azimuth_frequency_hz,
    azimuth_time_correction_s,
):
    """Calculate the azimuth time correction filter from DAD Eq. 6-40."""
    correction = np.asarray(azimuth_time_correction_s, dtype=np.float64)
    if correction.ndim == 0:
        correction = np.full(azimuth_frequency_hz.shape[1], float(correction))
    if correction.shape != (azimuth_frequency_hz.shape[1],):
        raise ValueError(
            "azimuth_time_correction_s must be scalar or one value per range sample."
        )
    return np.exp(2.0j * np.pi * azimuth_frequency_hz * correction[None, :])


def apply_filters(range_doppler, matched_filter, time_correction_filter):
    """Apply the two azimuth filters from DAD §6.3.4."""
    range_doppler *= matched_filter * time_correction_filter
    return range_doppler


def inverse_fft(range_doppler):
    """Transform focused data back to azimuth time (DAD §6.3.4)."""
    return ifft(ifftshift(range_doppler, axes=0), axis=0).astype(np.complex64)


def compress(
    range_doppler,
    azimuth_baseband_hz,
    doppler_centroid_hz,
    effective_velocity_mps,
    *,
    wavelength_m,
    slant_ranges_m,
    azimuth_time_correction_s=0.0,
    range_chunk=256,
):
    """Apply matched filtering, time correction, and inverse azimuth FFT."""
    for r0 in range(0, slant_ranges_m.size, range_chunk):
        r1 = min(r0 + range_chunk, slant_ranges_m.size)
        azimuth_frequency_hz = (
            azimuth_baseband_hz[:, None]
            + doppler_centroid_hz[None, r0:r1]
        )
        matched_filter = calculate_matched_filter(
            azimuth_frequency_hz,
            slant_ranges_m[r0:r1],
            effective_velocity_mps[r0:r1],
            wavelength_m,
        )
        time_filter = calculate_time_correction_filter(
            azimuth_frequency_hz,
            np.asarray(azimuth_time_correction_s)[r0:r1]
            if np.ndim(azimuth_time_correction_s) > 0
            else azimuth_time_correction_s,
        )
        apply_filters(range_doppler[:, r0:r1], matched_filter, time_filter)

    return inverse_fft(range_doppler)


__all__ = [
    "fm_rate_magnitude",
    "calculate_matched_filter",
    "calculate_time_correction_filter",
    "apply_filters",
    "inverse_fft",
    "compress",
]
