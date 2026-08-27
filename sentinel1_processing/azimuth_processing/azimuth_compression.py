"""Azimuth Compression from DAD Section 6.3.4."""

from ..azimuth_compression import (
    apply_filters,
    calculate_matched_filter,
    calculate_time_correction_filter,
    compress,
    fm_rate_magnitude,
    inverse_fft,
)

__all__ = [
    "fm_rate_magnitude",
    "calculate_matched_filter",
    "calculate_time_correction_filter",
    "apply_filters",
    "inverse_fft",
    "compress",
]
