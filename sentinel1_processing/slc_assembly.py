"""Azimuth-block sizing, focusing and SLC assembly."""

from dataclasses import dataclass

import numpy as np

from .azimuth_compression import (
    compress as compress_azimuth,
    fm_rate_magnitude,
)
from .rcmc import build_interpolation_table


@dataclass(frozen=True)
class Layout:
    matched_filter_support_samples: int
    overlap_samples: int
    step_samples: int
    support_probe_indices: np.ndarray
    support_probe_samples: tuple[int, ...]


def estimate_layout(
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
    """Estimate the fixed beam-wide azimuth support and block overlap."""
    probes = np.unique(np.round(
        np.linspace(0, num_azimuth_lines - 1, 5)
    ).astype(np.int64))
    supports = []

    for index in probes:
        fdc_hz = doppler_centroid_for_line(index)
        velocity_mps = velocity_estimator.evaluate_block(
            block_center_time_s=packet_azimuth_times_s[index],
            slant_range_m=slant_ranges_m,
            fdc_hz=fdc_hz,
            azimuth_bandwidth_hz=processing_bandwidth_hz,
            n_control_points=9,
            range_polynomial_degree=2,
        )
        far_rate = fm_rate_magnitude(
            slant_ranges_m[-1], velocity_mps[-1], fdc_hz[-1], wavelength_m
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

    return Layout(
        matched_filter_support_samples=support,
        overlap_samples=overlap,
        step_samples=fft_length - overlap,
        support_probe_indices=probes,
        support_probe_samples=tuple(supports),
    )


def assemble(
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
    src_segment_samples=1024,
    rcmc_kernel_length=16,
    rcmc_phases=64,
):
    """Focus overlapping azimuth blocks and stitch their valid samples."""
    focused_image = np.zeros_like(range_compressed, dtype=np.complex64)
    left_throw = layout.overlap_samples // 2
    right_throw = layout.overlap_samples - left_throw
    sinc_table = build_interpolation_table(rcmc_kernel_length, rcmc_phases)

    for start in range(0, range_compressed.shape[0], layout.step_samples):
        real_length = min(fft_length, range_compressed.shape[0] - start)
        block = np.zeros(
            (fft_length, range_compressed.shape[1]), dtype=np.complex64
        )
        block[:real_length] = range_compressed[start:start + real_length]
        center = start + (real_length - 1) // 2
        fdc_hz = doppler_centroid_for_line(center)
        velocity_mps = velocity_estimator.evaluate_block(
            block_center_time_s=packet_azimuth_times_s[center],
            slant_range_m=slant_ranges_m,
            fdc_hz=fdc_hz,
            azimuth_bandwidth_hz=processing_bandwidth_hz,
            n_control_points=9,
            range_polynomial_degree=2,
        )
        focused_block = compress_azimuth(
            block,
            fdc_hz,
            velocity_mps,
            azimuth_sample_period_s=azimuth_sample_period_s,
            range_sample_period_s=range_sample_period_s,
            range_sample_frequency_hz=range_sample_frequency_hz,
            speed_of_light_mps=speed_of_light_mps,
            wavelength_m=wavelength_m,
            slant_ranges_m=slant_ranges_m,
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


SlcLayout = Layout
estimate_slc_layout = estimate_layout
assemble_slc = assemble

__all__ = ["Layout", "estimate_layout", "assemble"]
