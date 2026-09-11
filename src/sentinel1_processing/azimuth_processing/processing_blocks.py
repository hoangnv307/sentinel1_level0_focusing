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
    """Stripmap output timeline and its valid input-grid stitching support."""

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
        dc_time_offsets_s=None,
        slice_overlap_s=0.0,
        required_first_time_s=None,
        required_last_time_s=None,
    ):
        """Build the valid output timeline after azimuth-filter throwaway.

        The default first line follows DAD Eq. 8-15. When range-dependent DC
        offsets are supplied, the first/last support follows Eq. 8-18/8-19.
        An explicit first time is quantized to the next PRI; an explicit
        last-line time is quantized down.
        """
        times = np.asarray(packet_azimuth_times_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("packet_azimuth_times_s must be strictly increasing.")
        pri = float(azimuth_sample_period_s)
        if pri <= 0:
            raise ValueError("azimuth_sample_period_s must be positive.")
        extra_overlap = (
            layout.overlap_samples - layout.matched_filter_support_samples
        )
        if extra_overlap < 0:
            raise ValueError("overlap_samples must include matched-filter support.")
        if slice_overlap_s < 0:
            raise ValueError("slice_overlap_s must be non-negative.")

        base_margin = (
            0.5 * float(slice_overlap_s) / pri
            + 0.5 * layout.matched_filter_support_samples
            + extra_overlap
        )
        if dc_time_offsets_s is None:
            dc_min = dc_max = float(nominal_dc_time_offset_s) / pri
        else:
            dc_offsets = np.asarray(dc_time_offsets_s, dtype=np.float64)
            if dc_offsets.size == 0 or not np.all(np.isfinite(dc_offsets)):
                raise ValueError("dc_time_offsets_s must contain finite values.")
            dc_min = float(np.min(dc_offsets)) / pri
            dc_max = float(np.max(dc_offsets)) / pri

        # DAD Eq. 8-18/8-19: the leading and trailing supports are asymmetric.
        lower = max(0, int(np.ceil(base_margin + dc_max)))
        upper = min(
            times.size,
            int(np.floor(times.size - (base_margin - dc_min))),
        )

        if required_first_time_s is not None:
            first = (float(required_first_time_s) - float(times[0])) / pri
            lower = max(lower, int(np.ceil(first)))
        if required_last_time_s is not None:
            last = (float(required_last_time_s) - float(times[0])) / pri
            upper = min(upper, int(np.floor(last)) + 1)
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
    velocity_estimator,
    *,
    wavelength_m,
    azimuth_sample_frequency_hz,
    processing_bandwidth_hz,
    max_doppler_centroid_hz,
    fft_length=4096,
    extra_overlap_samples=50,
):
    """Calculate the Stripmap block grid from L0 range/orbit and AUX_PP1."""
    ranges = np.asarray(slant_ranges_m, dtype=np.float64)
    times = np.asarray(packet_azimuth_times_s, dtype=np.float64)
    if (
        ranges.ndim != 1
        or ranges.size < 2
        or np.any(np.diff(ranges) <= 0)
    ):
        raise ValueError("slant_ranges_m must be a strictly increasing vector.")
    if (
        times.shape != (int(num_azimuth_lines),)
        or times.size < fft_length
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("packet_azimuth_times_s must cover one complete block.")

    def support_at(center_line):
        center_time = np.interp(center_line, np.arange(times.size), times)
        velocity = velocity_estimator.evaluate_block(
            block_center_time_s=center_time,
            slant_range_m=ranges,
            fdc_hz=max_doppler_centroid_hz,
            azimuth_bandwidth_hz=processing_bandwidth_hz,
            n_control_points=min(9, ranges.size),
            range_polynomial_degree=2,
        )
        far_rate = azimuth_compression.fm_rate_magnitude(
            ranges[-1], velocity[-1], max_doppler_centroid_hz, wavelength_m
        )
        if not np.isfinite(far_rate) or far_rate <= 0.0:
            raise ValueError("Invalid far-range azimuth FM rate.")
        return int(np.ceil(
            processing_bandwidth_hz / far_rate * azimuth_sample_frequency_hz
        ))

    support = support_at((fft_length - 1) / 2.0)
    for _ in range(10):
        overlap = support + extra_overlap_samples
        if overlap >= fft_length:
            raise ValueError("Azimuth overlap must be smaller than fft_length.")
        step = fft_length - overlap
        starts = np.arange(0, times.size - fft_length + 1, step)
        centers = starts + (fft_length - 1) / 2.0
        supports = tuple(map(support_at, centers))
        updated = max(support, *supports)
        if updated == support:
            return ProcessingBlockLayout(
                matched_filter_support_samples=support,
                overlap_samples=overlap,
                step_samples=step,
                support_probe_indices=centers,
                support_probe_samples=supports,
            )
        support = updated
    raise RuntimeError("Azimuth block layout did not converge.")


def derive_output_geometry(
    slant_ranges_m,
    packet_azimuth_times_s,
    doppler_centroid_for_line,
    geometry_doppler_for_line,
    velocity_estimator,
    layout,
    *,
    wavelength_m,
    speed_of_light_mps,
    azimuth_sample_period_s,
    range_sample_frequency_hz,
    processing_bandwidth_hz,
    fft_length=4096,
    rcmc_kernel_length=16,
    rcmc_phases=64,
    apply_coarse_bistatic_delay_correction=False,
    support_slant_ranges_m=None,
    support_doppler_centroid_for_line=None,
):
    """Derive the valid Stripmap SLC support from DAD §6.3.2 and §8.3.1."""
    ranges = np.asarray(slant_ranges_m, dtype=np.float64)
    if (support_slant_ranges_m is None) != (
        support_doppler_centroid_for_line is None
    ):
        raise ValueError("Support ranges and Doppler provider must be supplied together.")
    complete_support = support_slant_ranges_m is not None
    support_ranges = np.asarray(
        ranges if support_slant_ranges_m is None else support_slant_ranges_m,
        dtype=np.float64,
    )
    support_dc_provider = (
        doppler_centroid_for_line
        if support_doppler_centroid_for_line is None
        else support_doppler_centroid_for_line
    )
    times = np.asarray(packet_azimuth_times_s, dtype=np.float64)
    starts = list(range(0, times.size - fft_length + 1, layout.step_samples))
    if not starts:
        raise ValueError("Input must contain one complete azimuth FFT block.")

    pri = float(azimuth_sample_period_s)
    half_support = 0.5 * layout.matched_filter_support_samples
    extra_overlap = layout.overlap_samples - layout.matched_filter_support_samples

    def state(line, dc_provider=doppler_centroid_for_line, state_ranges=ranges):
        fdc = dc_provider(line)
        velocity = velocity_estimator.evaluate_block(
            block_center_time_s=times[line],
            slant_range_m=state_ranges,
            fdc_hz=fdc,
            azimuth_bandwidth_hz=processing_bandwidth_hz,
            n_control_points=9,
            range_polynomial_degree=2,
        )
        rate = azimuth_compression.fm_rate_magnitude(
            state_ranges, velocity, fdc, wavelength_m
        )
        return fdc, velocity, rate, -fdc / rate

    _, _, _, first_focus_dc_time = state(0)
    support_offset_lines = (
        half_support + extra_overlap + np.max(first_focus_dc_time) / pri
    )
    azimuth_start = int(np.floor(support_offset_lines))

    # DAD Eq. 8-15 uses nominal geometry DC at segment start, independently
    # from the Fine DCE used by the focusing and support calculations.
    geometry_fdc, _, geometry_rate, geometry_dc_time = state(
        0, geometry_doppler_for_line
    )
    nominal_index = int(np.argmax(geometry_fdc))
    anchor_offset_lines = (
        0.5
        * processing_bandwidth_hz
        / geometry_rate[-1]
        / pri
        # N overlap samples span N - 1 PRI intervals on the time grid.
        + max(extra_overlap - 1, 0)
        + geometry_dc_time[nominal_index] / pri
    )
    first_output_time = float(times[0] + np.ceil(anchor_offset_lines) * pri)

    last_start = starts[-1]
    if complete_support:
        last_center = last_start + (fft_length - 1) // 2
        _, _, last_rate, last_dc_time = state(
            last_center, support_dc_provider, support_ranges
        )
        # DAD Eq. 8-19: use the final block's own state and complete L0 range.
        trailing_throwaway = int(np.ceil(np.max(
            0.5 * processing_bandwidth_hz / last_rate / pri
            + extra_overlap
            + last_dc_time / pri
        )))
    else:
        _, _, last_rate, last_dc_time = state(times.size - 1)
        trailing_throwaway = int(np.floor(np.max(
            0.5 * processing_bandwidth_hz / last_rate / pri
            - last_dc_time / pri
        )))
    azimuth_stop = last_start + fft_length - trailing_throwaway

    offsets, _ = range_cell_migration_correction.build_interpolation_table(
        rcmc_kernel_length, rcmc_phases
    )
    range_start = int(-offsets[0])
    range_stop = ranges.size
    baseband = np.fft.fftshift(np.fft.fftfreq(fft_length, d=pri))
    spacing = speed_of_light_mps / (2.0 * range_sample_frequency_hz)
    for start in starts:
        center = start + (fft_length - 1) // 2
        fdc, velocity, _, _ = state(center)
        frequency = np.maximum(
            np.abs(baseband[0] + fdc),
            np.abs(baseband[-1] + fdc),
        )
        d = np.sqrt(np.maximum(
            1.0 - (wavelength_m * frequency / (2.0 * velocity)) ** 2,
            1e-15,
        ))
        source = (ranges / d - ranges[0]) / spacing
        base = np.floor(source).astype(np.int64)
        phase = np.floor((source - base) * rcmc_phases + 0.5).astype(np.int64)
        base += phase == rcmc_phases
        valid = np.flatnonzero(base + offsets[-1] < ranges.size)
        if valid.size == 0:
            raise ValueError("RCMC leaves no valid range samples.")
        range_stop = min(range_stop, int(valid[-1]) + 1)

    output_lines = azimuth_stop - azimuth_start
    if apply_coarse_bistatic_delay_correction:
        range_start_time = 2.0 * ranges[range_start] / speed_of_light_mps
        range_interval = 1.0 / range_sample_frequency_hz
        range_samples = range_stop - range_start
        first_output_time -= 0.5 * (
            range_start_time + 0.5 * range_samples * range_interval
        )
    return L1OutputGeometry(
        azimuth_start,
        azimuth_stop,
        range_start,
        range_stop,
        first_output_time,
        float(first_output_time + (output_lines - 1) * pri),
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
    """Run one Stripmap block in the order of DAD Figure 6-1."""
    # DAD §6.2.1: Azimuth Zero-Padding.
    azimuth_padded = azimuth_zero_padding.apply(block, fft_length)

    # DAD §6.2.3: Azimuth Forward FFT -> range-Doppler domain.
    azimuth_baseband_hz, range_doppler = azimuth_forward_fft.apply(
        azimuth_padded, azimuth_sample_period_s
    )

    # DAD §6.3.1: Secondary Range Compression.
    range_doppler = secondary_range_compression.apply(
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
    # DAD §6.3.2: Range Cell Migration Correction.
    range_doppler = range_cell_migration_correction.apply(
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
    # DAD §6.3.4: Azimuth Compression -> focused azimuth-time domain.
    return azimuth_compression.compress(
        range_doppler,
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
    """Focus Stripmap blocks and assemble the valid DAD §8.3.1 SLC support."""
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

    block_starts = range(
        0,
        range_compressed.shape[0] - fft_length + 1,
        layout.step_samples,
    )
    if not block_starts:
        raise ValueError("Input must contain one complete azimuth FFT block.")
    last_start = block_starts[-1]
    for start in block_starts:
        real_length = fft_length
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
        last = start == last_start
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
    "derive_output_geometry",
    "focus_block",
    "focus_slc",
]
