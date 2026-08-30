"""Range Reference Function from DAD Section 6.1.1."""

import numpy as np
from scipy.fft import fft


def calculate(
    *,
    sample_rate_hz,
    pulse_start_frequency_hz,
    pulse_ramp_rate_hz_per_s,
    pulse_length_s,
    fft_length,
):
    """Calculate the frequency-domain range matched filter."""
    num_tx_samples = int(np.ceil(pulse_length_s * sample_rate_hz))
    if fft_length < num_tx_samples:
        raise ValueError("fft_length must cover the transmitted pulse.")
    tx_time = (
        np.arange(num_tx_samples) / sample_rate_hz
        - (num_tx_samples - 1) / (2.0 * sample_rate_hz)
    )
    phi1 = (
        pulse_start_frequency_hz
        + pulse_ramp_rate_hz_per_s * pulse_length_s / 2.0
    )
    replica = np.exp(
        2.0j * np.pi * (
            phi1 * tx_time
            + pulse_ramp_rate_hz_per_s / 2.0 * tx_time**2
        )
    ).astype(np.complex64)
    matched_filter = np.conjugate(replica[::-1]).astype(np.complex64)
    matched_filter /= np.linalg.norm(matched_filter)
    return fft(matched_filter, n=fft_length).astype(np.complex64)

__all__ = ["calculate"]
