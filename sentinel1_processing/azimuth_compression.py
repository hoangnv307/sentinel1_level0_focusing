"""Azimuth matched filtering of one Range-Doppler block."""

import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift

from .rcmc import correct as correct_range_cell_migration
from .src import apply as apply_secondary_range_compression


def fm_rate_magnitude(range_m, velocity_mps, fdc_hz, wavelength_m):
    """Magnitude of Sentinel-1 L1 DAD Eq. 9-22."""
    d = np.sqrt(np.maximum(
        1.0 - (wavelength_m * fdc_hz / (2.0 * velocity_mps)) ** 2,
        0.0,
    ))
    return 2.0 * velocity_mps**2 * d**3 / (wavelength_m * range_m)


def compress(
    block,
    doppler_centroid_hz,
    effective_velocity_mps,
    *,
    azimuth_sample_period_s,
    range_sample_period_s,
    range_sample_frequency_hz,
    speed_of_light_mps,
    wavelength_m,
    slant_ranges_m,
    src_segment_samples=1024,
    rcmc_sinc_table=None,
    range_chunk=256,
):
    """Apply SRC, RCMC and the azimuth matched filter to one block."""
    azimuth_baseband_hz = fftshift(
        fftfreq(block.shape[0], d=azimuth_sample_period_s)
    )
    range_doppler = fftshift(fft(block, axis=0), axes=0).astype(np.complex64)

    apply_secondary_range_compression(
        range_doppler,
        azimuth_baseband_hz,
        doppler_centroid_hz,
        effective_velocity_mps,
        speed_of_light_mps=speed_of_light_mps,
        wavelength_m=wavelength_m,
        range_sample_period_s=range_sample_period_s,
        slant_ranges_m=slant_ranges_m,
        segment_samples=src_segment_samples,
    )
    corrected = correct_range_cell_migration(
        range_doppler,
        azimuth_baseband_hz,
        doppler_centroid_hz,
        effective_velocity_mps,
        speed_of_light_mps=speed_of_light_mps,
        wavelength_m=wavelength_m,
        range_sample_frequency_hz=range_sample_frequency_hz,
        slant_ranges_m=slant_ranges_m,
        sinc_table=rcmc_sinc_table,
    )

    for r0 in range(0, slant_ranges_m.size, range_chunk):
        r1 = min(r0 + range_chunk, slant_ranges_m.size)
        azimuth_frequency_hz = (
            azimuth_baseband_hz[:, None]
            + doppler_centroid_hz[None, r0:r1]
        )
        d = np.sqrt(np.maximum(
            1.0 - (
                wavelength_m
                * azimuth_frequency_hz
                / (2.0 * effective_velocity_mps[None, r0:r1])
            ) ** 2,
            0.0,
        ))
        corrected[:, r0:r1] *= np.exp(
            4.0j
            * np.pi
            * slant_ranges_m[None, r0:r1]
            * d
            / wavelength_m
        )

    return ifft(ifftshift(corrected, axes=0), axis=0).astype(np.complex64)


azimuth_fm_rate_magnitude = fm_rate_magnitude
compress_azimuth_block = compress

__all__ = ["fm_rate_magnitude", "compress"]
