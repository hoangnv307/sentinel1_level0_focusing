"""Joblib-cached processing stages used by the demonstration notebook."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from joblib import Memory
import numpy as np

import sentinel1_processing.azimuth_pre_processing as azimuth_pre_processing
import sentinel1_processing.azimuth_processing as azimuth_processing
import sentinel1_processing.raw_data_correction as raw_data_correction
from utils.checkpoint import CachePolicy, cached_call


def _decode_chunk(source_identity, selected_chunk, pipeline_version, level0_file):
    return level0_file.get_acquisition_chunk_data(selected_chunk)


def _estimate_iq_bias(decode_token, radar_data):
    return raw_data_correction.estimate_iq_bias(radar_data)


def _range_compression(
    range_token,
    radar_data,
    raw_slant_range_times_s,
    range_reference_function,
    range_sample_frequency_hz,
    pulse_start_frequency_hz,
    pulse_ramp_rate_hz_per_s,
    pulse_length_s,
    iq_bias,
):
    return azimuth_pre_processing.range.compression.compress(
        radar_data,
        raw_slant_range_times_s,
        sample_rate_hz=range_sample_frequency_hz,
        pulse_start_frequency_hz=pulse_start_frequency_hz,
        pulse_ramp_rate_hz_per_s=pulse_ramp_rate_hz_per_s,
        pulse_length_s=pulse_length_s,
        iq_bias=iq_bias,
        range_reference_function=range_reference_function,
    )


def _estimate_doppler(
    doppler_token,
    estimator,
    segments,
    t0_s,
    slice_start_times_s,
    last_slice_stop_time_s,
    product_start_time_s,
    product_stop_time_s,
    zero_doppler_offset_s,
):
    estimates, scene = estimator.estimate_segments(
        segments,
        t0_s=t0_s,
        slice_start_times_s=slice_start_times_s,
        last_slice_stop_time_s=last_slice_stop_time_s,
        product_start_time_s=product_start_time_s,
        product_stop_time_s=product_stop_time_s,
        zero_dop_minus_acq_time_s=zero_doppler_offset_s,
        return_prepared_scene=True,
    )
    return estimates, scene.alignment_summary()


def _focus_slc(
    focus_token,
    range_compressed,
    slant_ranges_m,
    packet_azimuth_times_s,
    doppler_centroid_for_block,
    effective_velocity_estimator,
    azimuth_block_layout,
    wavelength_m,
    speed_of_light_mps,
    azimuth_sample_period_s,
    range_sample_period_s,
    range_sample_frequency_hz,
    processing_bandwidth_hz,
    fft_length,
    azimuth_time_correction_s,
    src_segment_samples,
    rcmc_kernel_length,
    rcmc_phases,
):
    return azimuth_processing.processing_blocks.focus_slc(
        range_compressed,
        slant_ranges_m,
        packet_azimuth_times_s,
        doppler_centroid_for_block,
        effective_velocity_estimator,
        azimuth_block_layout,
        wavelength_m=wavelength_m,
        speed_of_light_mps=speed_of_light_mps,
        azimuth_sample_period_s=azimuth_sample_period_s,
        range_sample_period_s=range_sample_period_s,
        range_sample_frequency_hz=range_sample_frequency_hz,
        processing_bandwidth_hz=processing_bandwidth_hz,
        fft_length=fft_length,
        azimuth_time_correction_s=azimuth_time_correction_s,
        src_segment_samples=src_segment_samples,
        rcmc_kernel_length=rcmc_kernel_length,
        rcmc_phases=rcmc_phases,
    )


class NotebookCheckpoints:
    def __init__(
        self,
        cache_root: str | Path,
        *,
        policy: CachePolicy = "reuse",
        tag: str = "default",
        pipeline_version: int = 1,
        verbose: bool = True,
    ) -> None:
        self.policy = policy
        self.pipeline_version = pipeline_version
        self.memory = Memory(
            None if policy == "off" else Path(cache_root) / tag,
            mmap_mode=None if policy == "off" else "r",
            verbose=int(verbose),
        )
        self._decode = self.memory.cache(_decode_chunk, ignore=["level0_file"])
        self._iq_bias = self.memory.cache(_estimate_iq_bias, ignore=["radar_data"])
        self._range = self.memory.cache(
            _range_compression,
            ignore=[
                "radar_data",
                "raw_slant_range_times_s",
                "range_reference_function",
            ],
        )
        self._doppler = self.memory.cache(
            _estimate_doppler,
            ignore=["estimator", "segments"],
        )
        self._focus = self.memory.cache(
            _focus_slc,
            ignore=[
                "range_compressed",
                "slant_ranges_m",
                "packet_azimuth_times_s",
                "doppler_centroid_for_block",
                "effective_velocity_estimator",
                "azimuth_block_layout",
            ],
        )
        self._range_tokens: dict[int, tuple] = {}

    def decode(
        self,
        *,
        input_path: str | Path,
        selected_chunk: int,
        level0_file: Any,
    ) -> np.ndarray:
        return cached_call(
            self._decode,
            _decode_chunk,
            self.policy,
            self._source_identity(input_path),
            selected_chunk,
            self.pipeline_version,
            level0_file,
        )

    def estimate_iq_bias(
        self,
        *,
        input_path: str | Path,
        selected_chunk: int,
        radar_data: np.ndarray,
    ) -> complex:
        return cached_call(
            self._iq_bias,
            _estimate_iq_bias,
            self.policy,
            self._decode_token(input_path, selected_chunk),
            radar_data,
        )

    def range_compression(
        self,
        *,
        input_path: str | Path,
        selected_chunk: int,
        level0_file: Any,
        raw_slant_range_times_s: np.ndarray,
        range_sample_frequency_hz: float,
        pulse_start_frequency_hz: float,
        pulse_ramp_rate_hz_per_s: float,
        pulse_length_s: float,
        swst_bias_s: float,
        range_reference_function: np.ndarray | None = None,
        radar_data: np.ndarray | None = None,
        iq_bias: complex | None = None,
    ) -> tuple[np.ndarray, np.ndarray, complex]:
        if radar_data is None:
            radar_data = self.decode(
                input_path=input_path,
                selected_chunk=selected_chunk,
                level0_file=level0_file,
            )
        if iq_bias is None:
            iq_bias = self.estimate_iq_bias(
                input_path=input_path,
                selected_chunk=selected_chunk,
                radar_data=radar_data,
            )

        range_token = (
            self._decode_token(input_path, selected_chunk),
            raw_slant_range_times_s.size,
            float(raw_slant_range_times_s[0]),
            float(raw_slant_range_times_s[-1]),
            float(range_sample_frequency_hz),
            float(pulse_start_frequency_hz),
            float(pulse_ramp_rate_hz_per_s),
            float(pulse_length_s),
            float(swst_bias_s),
            complex(iq_bias),
            0 if range_reference_function is None else range_reference_function.size,
            self.pipeline_version,
        )
        self._range_tokens[selected_chunk] = range_token
        compressed, range_times = cached_call(
            self._range,
            _range_compression,
            self.policy,
            range_token,
            radar_data,
            raw_slant_range_times_s,
            range_reference_function,
            range_sample_frequency_hz,
            pulse_start_frequency_hz,
            pulse_ramp_rate_hz_per_s,
            pulse_length_s,
            iq_bias,
        )
        return compressed, range_times, iq_bias

    def doppler_estimation(
        self,
        *,
        selected_chunks: Sequence[int],
        estimator: Any,
        segments: Sequence[Any],
        t0_s: float,
        slice_start_times_s: Sequence[float],
        last_slice_stop_time_s: float,
        product_start_time_s: float,
        product_stop_time_s: float,
        zero_doppler_offset_s: float,
    ) -> tuple[Sequence[Any], list[dict]]:
        missing = [chunk for chunk in selected_chunks if chunk not in self._range_tokens]
        if missing:
            raise RuntimeError(f"Range compression must run first for chunks {missing}")
        doppler_token = (
            tuple(self._range_tokens[chunk] for chunk in selected_chunks),
            asdict(estimator.config),
            float(estimator.prf_hz),
            self.pipeline_version,
        )
        return cached_call(
            self._doppler,
            _estimate_doppler,
            self.policy,
            doppler_token,
            estimator,
            segments,
            t0_s,
            tuple(slice_start_times_s),
            last_slice_stop_time_s,
            product_start_time_s,
            product_stop_time_s,
            zero_doppler_offset_s,
        )

    def focus_slc(
        self,
        *,
        input_path: str | Path,
        selected_chunk: int,
        doppler_centroid_estimates: Sequence[Any],
        range_compressed: np.ndarray,
        slant_ranges_m: np.ndarray,
        packet_azimuth_times_s: np.ndarray,
        doppler_centroid_for_block: Callable[[int], np.ndarray],
        effective_velocity_estimator: Any,
        azimuth_block_layout: Any,
        wavelength_m: float,
        speed_of_light_mps: float,
        azimuth_sample_period_s: float,
        range_sample_period_s: float,
        range_sample_frequency_hz: float,
        processing_bandwidth_hz: float,
        fft_length: int,
        extra_overlap_samples: int,
        azimuth_time_correction_s: float,
        src_segment_samples: int,
        rcmc_kernel_length: int,
        rcmc_phases: int,
    ) -> np.ndarray:
        if selected_chunk not in self._range_tokens:
            raise RuntimeError("Range compression must run before focus_slc()")
        focus_token = (
            self._source_identity(input_path),
            self._range_tokens[selected_chunk],
            tuple(
                (
                    estimate.block.azimuth_time_s,
                    estimate.t0_s,
                    tuple(np.asarray(estimate.data_dc_polynomial)),
                )
                for estimate in doppler_centroid_estimates
            ),
            tuple(asdict(azimuth_block_layout).values()),
            float(wavelength_m),
            float(speed_of_light_mps),
            float(azimuth_sample_period_s),
            float(range_sample_period_s),
            float(range_sample_frequency_hz),
            float(processing_bandwidth_hz),
            int(fft_length),
            int(extra_overlap_samples),
            float(azimuth_time_correction_s),
            int(src_segment_samples),
            int(rcmc_kernel_length),
            int(rcmc_phases),
            self.pipeline_version,
        )
        return cached_call(
            self._focus,
            _focus_slc,
            self.policy,
            focus_token,
            range_compressed,
            slant_ranges_m,
            packet_azimuth_times_s,
            doppler_centroid_for_block,
            effective_velocity_estimator,
            azimuth_block_layout,
            wavelength_m,
            speed_of_light_mps,
            azimuth_sample_period_s,
            range_sample_period_s,
            range_sample_frequency_hz,
            processing_bandwidth_hz,
            fft_length,
            azimuth_time_correction_s,
            src_segment_samples,
            rcmc_kernel_length,
            rcmc_phases,
        )

    def _decode_token(self, input_path: str | Path, selected_chunk: int) -> tuple:
        return (
            self._source_identity(input_path),
            selected_chunk,
            self.pipeline_version,
        )

    @staticmethod
    def _source_identity(input_path: str | Path) -> tuple[str, int, int]:
        source = Path(input_path).resolve()
        stat = source.stat()
        return str(source), stat.st_size, stat.st_mtime_ns
