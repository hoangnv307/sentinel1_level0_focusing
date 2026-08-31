"""Stripmap block processing for DAD Sections 6.2 and 6.3."""

from dataclasses import dataclass

import numpy as np

from ..azimuth_pre_processing import azimuth_forward_fft, azimuth_zero_padding
from . import (
    azimuth_compression,
    range_cell_migration_correction,
    secondary_range_compression,
)


@dataclass(frozen=True)
class ProcessingBlockLayout:
    """Azimuth block support, overlap, and step."""

    matched_filter_support_samples: int
    overlap_samples: int
    step_samples: int
    support_probe_indices: np.ndarray
    support_probe_samples: tuple[int, ...]


def calculate_layout(
    num_azimuth_lines,
    slant_ranges_m,
    packet_azimuth_times_s,
    doppler_centroid_for_line,
    velocity_estimator,
    *,
    wavelength_m,
    azimuth_sample_frequency_hz,
    processing_bandwidth_hz,
    fft_length=4096,
    extra_overlap_samples=50,
):
    """Calculate Stripmap block overlap from DAD Sections 9.12 and 9.13."""
    probes = np.unique(np.round(
        np.linspace(0, num_azimuth_lines - 1, 5)
    ).astype(np.int64))
    supports = []

    for index in probes:
        doppler_centroid_hz = doppler_centroid_for_line(index)
        velocity_mps = velocity_estimator.evaluate_block(
            block_center_time_s=packet_azimuth_times_s[index],
            slant_range_m=slant_ranges_m,
            fdc_hz=doppler_centroid_hz,
            azimuth_bandwidth_hz=processing_bandwidth_hz,
            n_control_points=9,
            range_polynomial_degree=2,
        )
        far_rate = azimuth_compression.fm_rate_magnitude(
            slant_ranges_m[-1],
            velocity_mps[-1],
            doppler_centroid_hz[-1],
            wavelength_m,
        )
        if not np.isfinite(far_rate) or far_rate <= 0.0:
            raise ValueError("Invalid far-range azimuth FM rate.")
        supports.append(int(np.ceil(
            processing_bandwidth_hz / far_rate * azimuth_sample_frequency_hz
        )))

    support = max(supports)
    overlap = support + extra_overlap_samples
    if overlap >= fft_length:
        raise ValueError("Azimuth overlap must be smaller than fft_length.")

    return ProcessingBlockLayout(
        matched_filter_support_samples=support,
        overlap_samples=overlap,
        step_samples=fft_length - overlap,
        support_probe_indices=probes,
        support_probe_samples=tuple(supports),
    )


def focus_block(
    block,
    doppler_centroid_hz,
    effective_velocity_mps,
    *,
    fft_length,
    azimuth_sample_period_s,
    range_sample_period_s,
    range_sample_frequency_hz,
    speed_of_light_mps,
    wavelength_m,
    slant_ranges_m,
    azimuth_time_correction_s=0.0,
    src_segment_samples=1024,
    rcmc_sinc_table=None,
):
    """Run Azimuth Pre-Processing and Azimuth Processing for one block."""
    padded = azimuth_zero_padding.apply(block, fft_length)
    azimuth_baseband_hz, range_doppler = azimuth_forward_fft.apply(
        padded, azimuth_sample_period_s
    )
    secondary_range_compression.apply(
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
    corrected = range_cell_migration_correction.apply(
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
    return azimuth_compression.compress(
        corrected,
        azimuth_baseband_hz,
        doppler_centroid_hz,
        effective_velocity_mps,
        wavelength_m=wavelength_m,
        slant_ranges_m=slant_ranges_m,
        azimuth_time_correction_s=azimuth_time_correction_s,
    )


def focus_slc(
    range_compressed,
    slant_ranges_m,
    packet_azimuth_times_s,
    doppler_centroid_for_line,
    velocity_estimator,
    layout,
    *,
    wavelength_m,
    speed_of_light_mps,
    azimuth_sample_period_s,
    range_sample_period_s,
    range_sample_frequency_hz,
    processing_bandwidth_hz,
    fft_length=4096,
    azimuth_time_correction_s=0.0,
    src_segment_samples=1024,
    rcmc_kernel_length=16,
    rcmc_phases=64,
    output=None,
):
    """Focus Stripmap blocks and assemble the SLC."""
    if output is None:
        focused_image = np.zeros_like(range_compressed, dtype=np.complex64)
    else:
        if output.shape != range_compressed.shape:
            raise ValueError(
                "output shape must match the range-compressed input shape."
            )
        if not np.issubdtype(output.dtype, np.complexfloating):
            raise ValueError("output must have a complex dtype.")
        focused_image = output
        focused_image[...] = 0
    left_throw = layout.overlap_samples // 2
    right_throw = layout.overlap_samples - left_throw
    sinc_table = range_cell_migration_correction.build_interpolation_table(
        rcmc_kernel_length, rcmc_phases
    )

    for start in range(0, range_compressed.shape[0], layout.step_samples):
        real_length = min(fft_length, range_compressed.shape[0] - start)
        center = start + (real_length - 1) // 2
        doppler_centroid_hz = doppler_centroid_for_line(center)
        velocity_mps = velocity_estimator.evaluate_block(
            block_center_time_s=packet_azimuth_times_s[center],
            slant_range_m=slant_ranges_m,
            fdc_hz=doppler_centroid_hz,
            azimuth_bandwidth_hz=processing_bandwidth_hz,
            n_control_points=9,
            range_polynomial_degree=2,
        )
        focused_block = focus_block(
            range_compressed[start:start + real_length],
            doppler_centroid_hz,
            velocity_mps,
            fft_length=fft_length,
            azimuth_sample_period_s=azimuth_sample_period_s,
            range_sample_period_s=range_sample_period_s,
            range_sample_frequency_hz=range_sample_frequency_hz,
            speed_of_light_mps=speed_of_light_mps,
            wavelength_m=wavelength_m,
            slant_ranges_m=slant_ranges_m,
            azimuth_time_correction_s=azimuth_time_correction_s,
            src_segment_samples=src_segment_samples,
            rcmc_sinc_table=sinc_table,
        )

        first = start == 0
        last = start + real_length >= range_compressed.shape[0]
        keep0 = 0 if first else left_throw
        keep1 = real_length if last else fft_length - right_throw
        focused_image[start + keep0:start + keep1] = focused_block[keep0:keep1]
        if last:
            break

    return focused_image


__all__ = [
    "ProcessingBlockLayout",
    "calculate_layout",
    "focus_block",
    "focus_slc",
]
