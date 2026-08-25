"""Secondary Range Compression (SRC)."""

import numpy as np
from scipy.fft import fft, ifft, fftfreq


def apply_src(
    range_doppler,
    azimuth_baseband_hz,
    doppler_centroid_hz,
    effective_velocity_mps,
    *,
    speed_of_light_mps,
    wavelength_m,
    range_sample_period_s,
    slant_ranges_m,
    segment_samples=1024,
):
    """Apply SRC in range segments and return the modified input array."""
    n_azimuth, n_range = range_doppler.shape
    carrier_frequency_hz = speed_of_light_mps / wavelength_m

    for r0 in range(0, n_range, segment_samples):
        r1 = min(r0 + segment_samples, n_range)
        segment_length = r1 - r0
        fft_size = 1 << int(np.ceil(np.log2(2 * segment_length)))

        segment_fft = fft(range_doppler[:, r0:r1], n=fft_size, axis=1)
        range_frequency_hz = fftfreq(
            fft_size, d=range_sample_period_s
        )[None, :]

        middle = (r0 + r1 - 1) // 2
        slant_range_m = slant_ranges_m[middle]
        velocity_mps = effective_velocity_mps[middle]
        centroid_hz = doppler_centroid_hz[middle]

        azimuth_frequency_hz = azimuth_baseband_hz + centroid_hz
        d = np.sqrt(np.maximum(
            1.0 - (
                wavelength_m * azimuth_frequency_hz / (2.0 * velocity_mps)
            ) ** 2,
            0.0,
        ))

        nonzero = np.abs(azimuth_frequency_hz) > 1e-12
        inverse_chirp_rate = np.zeros_like(azimuth_frequency_hz)
        inverse_chirp_rate[nonzero] = 1.0 / (
            2.0
            * velocity_mps**2
            * carrier_frequency_hz**3
            * d[nonzero] ** 2
            / (
                speed_of_light_mps
                * slant_range_m
                * azimuth_frequency_hz[nonzero] ** 2
            )
        )

        src_filter = np.exp(
            -1.0j
            * np.pi
            * range_frequency_hz**2
            * inverse_chirp_rate[:, None]
        )
        range_doppler[:, r0:r1] = ifft(
            segment_fft * src_filter, axis=1
        )[:, :segment_length]

    return range_doppler
