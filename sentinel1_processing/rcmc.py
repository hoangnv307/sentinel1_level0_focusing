"""Sinc-interpolated Range Cell Migration Correction (RCMC)."""

import numpy as np


def build_interpolation_table(length=16, phases=64):
    """Build the windowed-sinc fractional-delay lookup table."""
    left = length // 2 - 1
    offsets = np.arange(-left, length - left)
    window = np.hanning(length)
    table = np.empty((phases, length))

    for phase in range(phases):
        kernel = np.sinc(offsets - phase / phases) * window
        table[phase] = kernel / np.sum(kernel)

    return offsets, table


def correct(
    range_doppler,
    azimuth_baseband_hz,
    doppler_centroid_hz,
    effective_velocity_mps,
    *,
    speed_of_light_mps,
    wavelength_m,
    range_sample_frequency_hz,
    slant_ranges_m,
    sinc_table=None,
):
    """Correct range migration using a quantized windowed-sinc kernel."""
    offsets, table = sinc_table or build_interpolation_table()
    phases = table.shape[0]
    n_azimuth, n_range = range_doppler.shape
    corrected = np.zeros_like(range_doppler)
    range_spacing_m = speed_of_light_mps / (2.0 * range_sample_frequency_hz)
    first_range_m = slant_ranges_m[0]

    for azimuth_index in range(n_azimuth):
        azimuth_frequency_hz = (
            azimuth_baseband_hz[azimuth_index] + doppler_centroid_hz
        )
        d = np.sqrt(np.maximum(
            1.0 - (
                wavelength_m
                * azimuth_frequency_hz
                / (2.0 * effective_velocity_mps)
            ) ** 2,
            1e-15,
        ))
        source_position = (
            slant_ranges_m / d - first_range_m
        ) / range_spacing_m

        base = np.floor(source_position).astype(np.int64)
        phase = np.floor(
            (source_position - base) * phases + 0.5
        ).astype(np.int64)
        carry = phase == phases
        base[carry] += 1
        phase[carry] = 0

        valid = (
            (base + offsets[0] >= 0)
            & (base + offsets[-1] < n_range)
        )
        columns = np.nonzero(valid)[0]
        indices = base[columns, None] + offsets[None, :]
        corrected[azimuth_index, columns] = np.sum(
            range_doppler[azimuth_index, indices] * table[phase[columns]],
            axis=1,
        )

    return corrected


build_sinc_table = build_interpolation_table
apply_rcmc = correct

__all__ = ["build_interpolation_table", "correct"]
