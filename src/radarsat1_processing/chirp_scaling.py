"""Chirp Scaling Algorithm, chuyển trực tiếp từ SAR_RADARSAT1_CSA.m."""

import numpy as np

from .ceos import RawMetadata, SPEED_OF_LIGHT_MPS


def focus(data: np.ndarray, metadata: RawMetadata, *, first_sample: int = 0) -> np.ndarray:
    """Focus một cửa sổ raw I/Q và trả về SLC ``complex64`` cùng kích thước."""

    if data.ndim != 2 or min(data.shape) < 2:
        raise ValueError("Dữ liệu I/Q phải là ma trận 2-D")

    azimuth_count, range_count = data.shape
    c = SPEED_OF_LIGHT_MPS
    fs = metadata.sample_rate_hz
    prf = metadata.prf_hz
    f0 = metadata.carrier_frequency_hz
    velocity = metadata.effective_velocity_mps
    center_pixel = first_sample + range_count / 2
    centroid = sum(
        value * center_pixel**power
        for power, value in enumerate(metadata.doppler_centroid_coefficients)
    )
    chirp_rate = metadata.chirp_rate_hz_per_s
    range_start_time = metadata.range_start_time_s + first_sample / fs
    first_slant_range = range_start_time * c / 2
    reference_range = (range_start_time + range_count / (2 * fs)) * c / 2

    eta = (np.arange(azimuth_count) - azimuth_count / 2) / prf
    azimuth_frequency = centroid + (np.arange(azimuth_count) - azimuth_count / 2) * prf / azimuth_count
    range_frequency = (np.arange(range_count) - range_count / 2) * fs / range_count
    migration_factor = np.sqrt(1 - c**2 * azimuth_frequency**2 / (4 * velocity**2 * f0**2))
    reference_factor = np.sqrt(1 - c**2 * centroid**2 / (4 * velocity**2 * f0**2))
    src_inverse = c * first_slant_range * azimuth_frequency**2 / (2 * velocity**2 * f0**3 * migration_factor**3)
    modified_chirp_rate = chirp_rate / (1 - chirp_rate * src_inverse)
    bulk_migration = (1 / migration_factor - 1 / reference_factor) * reference_range
    alpha = reference_factor / migration_factor - 1
    scaling_time = 2 / c * (first_slant_range / reference_factor + bulk_migration) - 2 * reference_range / (c * migration_factor)

    work = np.asarray(data, dtype=np.complex64) * np.exp(-2j * np.pi * centroid * eta)[:, None]
    work = np.fft.fftshift(np.fft.fft(np.fft.fftshift(work, axes=0), axis=0), axes=0)
    work *= np.exp(1j * np.pi * modified_chirp_rate * alpha * scaling_time**2)[:, None]
    work = np.fft.fftshift(np.fft.fft(np.fft.fftshift(work, axes=1), axis=1), axes=1)
    work *= np.kaiser(azimuth_count, 2.5)[:, None]
    work *= np.kaiser(range_count, 2.5)[None, :]
    work *= np.exp(
        1j * np.pi * range_frequency[None, :] ** 2
        / (modified_chirp_rate * (1 + alpha))[:, None]
        + 1j * 4 * np.pi / c * bulk_migration[:, None] * range_frequency[None, :]
    )
    work = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(work, axes=1), axis=1), axes=1)
    azimuth_filter = np.exp(-1j * 4 * np.pi * first_slant_range * f0 * migration_factor / c)
    phase_correction = np.exp(
        1j * 4 * np.pi * modified_chirp_rate / c**2
        * (1 - migration_factor / reference_factor)
        * (first_slant_range / migration_factor - reference_range / migration_factor) ** 2
    )
    work *= (azimuth_filter * phase_correction)[:, None]
    return np.flipud(np.fft.ifft(np.fft.ifftshift(work, axes=0), axis=0)).astype(np.complex64)


def to_uint8(slc: np.ndarray, dynamic_range_db: float = 60.0) -> np.ndarray:
    """Đổi biên độ SLC sang ảnh xám logarit."""

    magnitude = np.abs(slc)
    peak = float(magnitude.max())
    if peak == 0:
        return np.zeros(slc.shape, dtype=np.uint8)
    db = np.maximum(20 * np.log10(magnitude / peak + np.finfo(np.float32).eps), -dynamic_range_db)
    return np.rint((db + dynamic_range_db) * 255 / dynamic_range_db).astype(np.uint8)
