"""Range Compression from DAD Section 6.2.2."""

import numpy as np
from scipy.fft import fft, ifft

from ...range_processing import reference_function


def zero_pad(range_lines, fft_length, iq_bias=0.0j):
    """Zero-pad range lines to the range FFT length."""
    data = np.asarray(range_lines)
    if data.ndim != 2 or data.shape[1] > fft_length:
        raise ValueError("range_lines must be 2-D and fit inside fft_length.")
    padded = np.zeros((data.shape[0], fft_length), dtype=np.complex64)
    padded[:, :data.shape[1]] = data - iq_bias
    return padded


def forward_fft(padded_range_lines):
    """Transform zero-padded range lines to range frequency."""
    return fft(padded_range_lines, axis=1).astype(np.complex64)


def multiply_reference_function(range_spectrum, range_reference_function):
    """Multiply the range spectrum by the Range Reference Function."""
    reference = np.asarray(range_reference_function, dtype=np.complex64)
    if reference.shape != (range_spectrum.shape[1],):
        raise ValueError("range_reference_function has the wrong FFT length.")
    range_spectrum *= reference[None, :]
    return range_spectrum


def inverse_fft(filtered_spectrum):
    """Transform range-compressed lines back to range time."""
    return ifft(filtered_spectrum, axis=1).astype(np.complex64)


def extract_valid_samples(
    filtered_lines,
    raw_slant_range_times_s,
    transmitted_pulse_samples,
    output,
):
    """Return the valid or same-size part of the range-compressed lines."""
    raw_times = np.asarray(raw_slant_range_times_s)
    n_range = raw_times.size
    same_start = (transmitted_pulse_samples - 1) // 2
    if output == "valid":
        valid_stop = n_range - (
            transmitted_pulse_samples - 1 - same_start
        )
        data_slice = slice(transmitted_pulse_samples - 1, n_range)
        return filtered_lines[:, data_slice], raw_times[same_start:valid_stop]
    if output == "same":
        data_slice = slice(same_start, same_start + n_range)
        return filtered_lines[:, data_slice], raw_times
    raise ValueError("output must be 'valid' or 'same'.")


def compress(
    radar_data,
    raw_slant_range_times_s,
    *,
    sample_rate_hz,
    pulse_start_frequency_hz,
    pulse_ramp_rate_hz_per_s,
    pulse_length_s,
    batch_lines=256,
    output="valid",
    iq_bias=0.0j,
    range_reference_function=None,
    output_array=None,
):
    """Run the Range Compression steps from DAD Section 6.2.2."""
    n_azimuth, n_range = radar_data.shape
    num_tx_samples = int(np.ceil(pulse_length_s * sample_rate_hz))
    fft_length = 1 << int(np.ceil(np.log2(n_range + num_tx_samples - 1)))
    reference = (
        reference_function.calculate(
            sample_rate_hz=sample_rate_hz,
            pulse_start_frequency_hz=pulse_start_frequency_hz,
            pulse_ramp_rate_hz_per_s=pulse_ramp_rate_hz_per_s,
            pulse_length_s=pulse_length_s,
            fft_length=fft_length,
        )
        if range_reference_function is None
        else np.asarray(range_reference_function, dtype=np.complex64)
    )
    output_length = n_range - num_tx_samples + 1 if output == "valid" else n_range
    expected_shape = (n_azimuth, output_length)
    if output_array is None:
        compressed = np.empty(expected_shape, dtype=np.complex64)
    else:
        if output_array.shape != expected_shape:
            raise ValueError(
                f"output_array shape must be {expected_shape}, got {output_array.shape}."
            )
        if not np.issubdtype(output_array.dtype, np.complexfloating):
            raise ValueError("output_array must have a complex dtype.")
        compressed = output_array

    output_times = None
    for start in range(0, n_azimuth, batch_lines):
        stop = min(start + batch_lines, n_azimuth)
        padded = zero_pad(radar_data[start:stop], fft_length, iq_bias)
        spectrum = forward_fft(padded)
        multiply_reference_function(spectrum, reference)
        filtered = inverse_fft(spectrum)
        compressed[start:stop], output_times = extract_valid_samples(
            filtered, raw_slant_range_times_s, num_tx_samples, output
        )

    return compressed, output_times


__all__ = [
    "zero_pad",
    "forward_fft",
    "multiply_reference_function",
    "inverse_fft",
    "extract_valid_samples",
    "compress",
]
