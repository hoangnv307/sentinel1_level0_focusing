"""Range Reference Function from DAD Section 6.1.1."""

import numpy as np
from scipy.fft import fft, fftfreq


def from_coefficients(
    *, sample_rate_hz, pulse_length_s, amplitude_coefficients,
    phase_coefficients_cycles, fft_length, internal_delay_s=0.0, complex_gain=1.0,
):
    """DAD §6.1.1: centered chirp, whitening, delay advance, unit energy.

    The spectrum is the FFT of the finite chirp, without an assumed IPF taper.
    Phase coefficients are cycles, in increasing powers of seconds.
    """
    amplitude = np.asarray(amplitude_coefficients, dtype=float)
    phase = np.asarray(phase_coefficients_cycles, dtype=float)
    if (not np.isfinite([sample_rate_hz, pulse_length_s, internal_delay_s, complex_gain]).all()
            or sample_rate_hz <= 0 or pulse_length_s <= 0 or internal_delay_s < 0
            or complex_gain == 0):
        raise ValueError("Tham số thời gian/gain của RRF không hợp lệ.")
    if any(c.ndim != 1 or not c.size or not np.isfinite(c).all() for c in (amplitude, phase)):
        raise ValueError("Hệ số replica phải là vector hữu hạn, không rỗng.")
    n = int(np.ceil(sample_rate_hz * pulse_length_s))
    if n < 2 or fft_length != int(fft_length) or fft_length < n:
        raise ValueError("FFT không đủ support replica.")
    times = (np.arange(n) - (n - 1) / 2.0) / sample_rate_hz
    envelope = np.polynomial.polynomial.polyval(times, amplitude)
    if not np.isfinite(envelope).all() or np.any(envelope <= 0):
        raise ValueError("Replica amplitude không hữu hạn/dương; không thể whitening.")
    replica = np.exp(2j * np.pi * np.polynomial.polynomial.polyval(times, phase)) / envelope
    # Reversal places the peak at M-1, matching compression.extract_valid_samples.
    spectrum = fft(np.conj(replica[::-1]), n=int(fft_length))
    spectrum *= np.exp(2j * np.pi * fftfreq(int(fft_length), d=1.0 / sample_rate_hz) * internal_delay_s)
    energy = np.mean(np.abs(spectrum) ** 2)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("RRF không có năng lượng hữu hạn/dương.")
    return (spectrum / np.sqrt(energy) * complex_gain).astype(np.complex64)


def create_freq_domain(
    *,
    sample_rate_hz,
    pulse_start_frequency_hz,
    pulse_ramp_rate_hz_per_s,
    pulse_length_s,
    fft_length,
    internal_delay_s=0.0,
):
    """Create the frequency-domain Range Reference Function."""
    phi1 = (
        pulse_start_frequency_hz
        + pulse_ramp_rate_hz_per_s * pulse_length_s / 2.0
    )
    return from_coefficients(
        sample_rate_hz=sample_rate_hz, pulse_length_s=pulse_length_s,
        amplitude_coefficients=[1.0],
        phase_coefficients_cycles=[0.0, phi1, pulse_ramp_rate_hz_per_s / 2.0],
        fft_length=fft_length,
        internal_delay_s=internal_delay_s,
    )

__all__ = ["create_freq_domain", "from_coefficients"]
