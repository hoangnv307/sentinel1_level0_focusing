"""Sentinel-1 range compression."""

import numpy as np
from scipy.fft import fft, ifft


def compress_range(
    radar_data,
    raw_slant_range_times_s,
    *,
    sample_rate_hz,
    pulse_start_frequency_hz,
    pulse_ramp_rate_hz_per_s,
    pulse_length_s,
    batch_lines=256,
    output="valid",
):
    """Apply the linear matched filter and return ``valid`` or ``same`` support."""
    if output not in {"valid", "same"}:
        raise ValueError("output must be 'valid' or 'same'.")
    n_azimuth, n_range = radar_data.shape
    num_tx_samples = int(np.ceil(pulse_length_s * sample_rate_hz))
    tx_time = (
        np.arange(num_tx_samples) / sample_rate_hz
        - (num_tx_samples - 1) / (2.0 * sample_rate_hz)
    )
    phi1 = pulse_start_frequency_hz + pulse_ramp_rate_hz_per_s * pulse_length_s / 2.0
    tx_replica = np.exp(
        2.0j * np.pi * (
            phi1 * tx_time
            + pulse_ramp_rate_hz_per_s / 2.0 * tx_time**2
        )
    ).astype(np.complex64)

    matched_filter = np.conjugate(tx_replica[::-1]).astype(np.complex64)
    matched_filter /= np.sqrt(np.sum(np.abs(matched_filter) ** 2))

    fft_size = 1 << int(np.ceil(np.log2(n_range + num_tx_samples - 1)))
    filter_fft = fft(matched_filter, n=fft_size).astype(np.complex64)

    same_start = (num_tx_samples - 1) // 2
    valid_start = same_start
    valid_stop = n_range - ((num_tx_samples - 1) - same_start)
    output_slice = (
        slice(same_start + valid_start, same_start + valid_stop)
        if output == "valid"
        else slice(same_start, same_start + n_range)
    )
    output_times = (
        np.asarray(raw_slant_range_times_s)[valid_start:valid_stop]
        if output == "valid"
        else np.asarray(raw_slant_range_times_s)
    )
    compressed = np.empty((n_azimuth, output_times.size), dtype=np.complex64)

    for a0 in range(0, n_azimuth, batch_lines):
        a1 = min(a0 + batch_lines, n_azimuth)
        spectrum = fft(radar_data[a0:a1], n=fft_size, axis=1)
        filtered = ifft(spectrum * filter_fft[None, :], axis=1)
        compressed[a0:a1] = filtered[:, output_slice]

    return compressed, output_times
