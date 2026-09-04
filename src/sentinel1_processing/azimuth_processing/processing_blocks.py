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


@dataclass(frozen=True)
class L1OutputGeometry:
    """Valid Stripmap SLC extent on the input PRI/range grid."""

    azimuth_start_line: int
    azimuth_stop_line: int
    range_start_sample: int
    range_stop_sample: int
    first_zero_doppler_time_s: float
    last_zero_doppler_time_s: float

    @property
    def shape(self):
        return (
            self.azimuth_stop_line - self.azimuth_start_line,
            self.range_stop_sample - self.range_start_sample,
        )

    @classmethod
    def from_focus_support(
        cls,
        packet_azimuth_times_s,
        num_range_samples,
        layout,
        *,
        azimuth_sample_period_s,
        nominal_dc_time_offset_s=0.0,
        required_first_time_s=None,
        required_last_time_s=None,
    ):
        """Build the valid output timeline after azimuth-filter throwaway.

        ``required_first_time_s``/``required_last_time_s`` (zero-Doppler output
        times, DAD §8.3.1) bound the slice on the PRI grid and are snapped to
        integer lines. When omitted the support is the symmetric azimuth-filter
        margin, so any caller that knows the annotation output extent should pass
        them to reproduce the ESA SLC slice (e.g. this scene: first/last line
        UTC from imageAnnotation).
        """
        times = np.asarray(packet_azimuth_times_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("packet_azimuth_times_s must be strictly increasing.")
        pri = float(azimuth_sample_period_s)
        if pri <= 0:
            raise ValueError("azimuth_sample_period_s must be positive.")
        # ponytail: one nominal offset; use range-extreme geometry offsets when
        # the downlink-quaternion convention is available.
        half_overlap = layout.overlap_samples / 2
        margin = half_overlap + nominal_dc_time_offset_s / pri

        lower = max(0, int(np.ceil(margin)))
        upper = min(
            times.size,
            int(np.floor(times.size - half_overlap + nominal_dc_time_offset_s / pri)),
        )

        def snap_to_line(time_s, cell):
            # Indices are integer PRI samples; snap any sub-half-sample goal to
            # the nearest grid line so the anchor never splits a PRI.
            return int(np.rint((float(time_s) - float(times[0])) / pri))

        if required_first_time_s is not None:
            lower = max(lower, snap_to_line(required_first_time_s, lower))
        if required_last_time_s is not None:
            upper = min(upper, snap_to_line(required_last_time_s, upper) + 1)
        if upper <= lower:
            raise ValueError(
                "Zero-Doppler output window lies outside the az focus support."
            )
        return cls(
            lower,
            upper,
            0,
            int(num_range_samples),
            float(times[0] + lower * pri),
            float(times[0] + (upper - 1) * pri),
        )


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
    output_geometry=None,
    output=None,
):
    """Focus Stripmap blocks and assemble the SLC."""
    geometry = output_geometry or L1OutputGeometry(
        0,
        range_compressed.shape[0],
        0,
        range_compressed.shape[1],
        float(packet_azimuth_times_s[0]),
        float(packet_azimuth_times_s[-1]),
    )
    valid_azimuth = (
        0
        <= geometry.azimuth_start_line
        < geometry.azimuth_stop_line
        <= range_compressed.shape[0]
    )
    valid_range = (
        0
        <= geometry.range_start_sample
        < geometry.range_stop_sample
        <= range_compressed.shape[1]
    )
    if not (valid_azimuth and valid_range):
        raise ValueError("output_geometry lies outside the range-compressed input.")
    if output is None:
        focused_image = np.zeros(geometry.shape, dtype=np.complex64)
    else:
        if output.shape != geometry.shape:
            raise ValueError(
                f"output shape must match output_geometry shape {geometry.shape}."
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
        global0 = max(start + keep0, geometry.azimuth_start_line)
        global1 = min(start + keep1, geometry.azimuth_stop_line)
        if global1 > global0:
            r0 = geometry.range_start_sample
            r1 = geometry.range_stop_sample
            focused_image[
                global0 - geometry.azimuth_start_line:
                global1 - geometry.azimuth_start_line
            ] = focused_block[global0 - start:global1 - start, r0:r1]
        if last:
            break

    return focused_image


__all__ = [
    "ProcessingBlockLayout",
    "L1OutputGeometry",
    "calculate_layout",
    "focus_block",
    "focus_slc",
]
