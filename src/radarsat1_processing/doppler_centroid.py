"""Doppler centroid RADARSAT-1 dùng pipeline DCE hiện có của Sentinel-1."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import struct

import numpy as np

from sentinel1_processing.azimuth_pre_processing.range.compression import compress
from sentinel1_processing.doppler_centroid_estimation import Config, Estimator

from .ceos import SPEED_OF_LIGHT_MPS, decode, read_metadata


EARTH_ANGULAR_VELOCITY_RAD_S = 7.292115e-5
WGS84_A_M = 6_378_137.0
WGS84_E2 = 6.69437999014e-3


@dataclass(frozen=True)
class L1DopplerReference:
    data_coefficients: tuple[float, float, float]
    near_slant_range_m: float
    far_slant_range_m: float
    pixel_count: int

    def evaluate(self, slant_range_m) -> np.ndarray:
        slant_range = np.asarray(slant_range_m, dtype=np.float64)
        pixel = (
            (slant_range - self.near_slant_range_m)
            / (self.far_slant_range_m - self.near_slant_range_m)
            * (self.pixel_count - 1)
        )
        return np.polynomial.polynomial.polyval(pixel, self.data_coefficients)


def _number(text: str, label: str) -> float:
    match = re.search(
        rf"^\s*{re.escape(label)}\s*(?:=|\t)\s*([-+0-9.eE]+)",
        text,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Không tìm thấy trường L1: {label}")
    return float(match.group(1))


def read_l1_reference(path: str | Path) -> L1DopplerReference:
    """Đọc CRT Doppler polynomial từ bản text của CEOS L1 leader."""

    text = Path(path).read_text()
    return L1DopplerReference(
        data_coefficients=(
            _number(text, "Doppler frequency at the near range"),
            _number(text, "Doppler frequency slope"),
            _number(text, "Doppler frequency quadratic term"),
        ),
        near_slant_range_m=1_000 * _number(text, "Slant range to the first image pixel"),
        far_slant_range_m=1_000 * _number(text, "Slant range to the last image pixel"),
        pixel_count=int(_number(text, "Actual (no filler) number of pixels per line")),
    )


def _l0_geometry(raw_path: Path):
    text = raw_path.with_suffix(".meta").read_text()

    def value(name: str) -> float:
        match = re.search(rf"^\s*{name}\s*=\s*([^\s]+)", text, re.MULTILINE)
        if not match:
            raise ValueError(f"Không tìm thấy trường L0 geometry: {name}")
        return float(match.group(1))

    center_time = re.search(r"^\s*CENTER_TIME\s*=\s*(\S+)", text, re.MULTILINE)
    if not center_time:
        raise ValueError("Không tìm thấy CENTER_TIME trong L0 metadata")
    center = datetime.strptime(center_time.group(1), "%Y-%jT%H:%M:%S.%f")
    center_seconds = center.hour * 3600 + center.minute * 60 + center.second + center.microsecond / 1e6

    leader = raw_path.with_suffix(".ldr").read_bytes()
    offset = 0
    position_record = None
    while offset < len(leader):
        _, record_type, length = struct.unpack(">III", leader[offset : offset + 12])
        if record_type == 0x121E1214:
            position_record = leader[offset : offset + length]
            break
        offset += length
    if position_record is None:
        raise ValueError("L0 leader không có Platform Position record")
    state_start_seconds = float(position_record[160:182])
    greenwich_hour_angle_rad = np.deg2rad(float(position_record[268:290]))
    earth_angle = greenwich_hour_angle_rad + EARTH_ANGULAR_VELOCITY_RAD_S * (
        center_seconds - state_start_seconds
    )

    near = np.mean([
        [value("NEAR_START_LAT"), value("NEAR_START_LON")],
        [value("NEAR_END_LAT"), value("NEAR_END_LON")],
    ], axis=0)
    far = np.mean([
        [value("FAR_START_LAT"), value("FAR_START_LON")],
        [value("FAR_END_LAT"), value("FAR_END_LON")],
    ], axis=0)
    return {
        "satellite_position_m": 1_000 * np.array([
            value("X_POSITION"), value("Y_POSITION"), value("Z_POSITION")
        ]),
        "satellite_velocity_mps": np.array([
            value("X_VELOCITY"), value("Y_VELOCITY"), value("Z_VELOCITY")
        ]),
        "near_lat_lon": near,
        "far_lat_lon": far,
        "near_slant_range_m": 1_000 * value("SL_RNG_1ST_PIX"),
        "far_slant_range_m": 1_000 * value("SL_RNG_LAST_PIX"),
        "earth_angle_rad": earth_angle,
    }


def geometry_doppler(raw_path: str | Path, slant_range_times_s) -> np.ndarray:
    """Tính Geometry DC từ orbit, Earth rotation và footprint L0, không dùng L1."""

    raw_path = Path(raw_path)
    metadata = read_metadata(raw_path)
    geometry = _l0_geometry(raw_path)
    slant_range_m = np.asarray(slant_range_times_s, dtype=np.float64) * SPEED_OF_LIGHT_MPS / 2
    fraction = (
        (slant_range_m - geometry["near_slant_range_m"])
        / (geometry["far_slant_range_m"] - geometry["near_slant_range_m"])
    )
    lat_lon = (
        geometry["near_lat_lon"][None, :]
        + fraction[:, None] * (geometry["far_lat_lon"] - geometry["near_lat_lon"])[None, :]
    )
    latitude = np.deg2rad(lat_lon[:, 0])
    longitude = np.deg2rad(lat_lon[:, 1])
    prime_vertical_radius = WGS84_A_M / np.sqrt(1 - WGS84_E2 * np.sin(latitude) ** 2)
    ground_ecef = np.column_stack((
        prime_vertical_radius * np.cos(latitude) * np.cos(longitude),
        prime_vertical_radius * np.cos(latitude) * np.sin(longitude),
        prime_vertical_radius * (1 - WGS84_E2) * np.sin(latitude),
    ))
    angle = geometry["earth_angle_rad"]
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    ground_eci = ground_ecef @ rotation.T
    ground_velocity = np.cross(
        np.array([0.0, 0.0, EARTH_ANGULAR_VELOCITY_RAD_S]), ground_eci
    )
    line_of_sight = ground_eci - geometry["satellite_position_m"]
    line_of_sight /= np.linalg.norm(line_of_sight, axis=1)[:, None]
    relative_velocity = geometry["satellite_velocity_mps"] - ground_velocity
    wavelength_m = SPEED_OF_LIGHT_MPS / metadata.carrier_frequency_hz
    return 2 / wavelength_m * np.sum(relative_velocity * line_of_sight, axis=1)


def estimate(raw_path: str | Path, *, azimuth_lines: int = 6000, num_range_blocks: int = 20):
    """Ước lượng một DCE record tại tâm frame bằng pipeline Sentinel-1."""

    raw_path = Path(raw_path)
    metadata = read_metadata(raw_path)
    if azimuth_lines < 2 or azimuth_lines > metadata.line_count:
        raise ValueError("azimuth_lines nằm ngoài frame")
    first_line = (metadata.line_count - azimuth_lines) // 2
    raw = decode(raw_path, first_line=first_line, line_count=azimuth_lines)
    raw_times = metadata.range_start_time_s + np.arange(metadata.range_samples) / metadata.sample_rate_hz
    compressed, range_times = compress(
        raw,
        raw_times,
        sample_rate_hz=metadata.sample_rate_hz,
        pulse_start_frequency_hz=(
            -metadata.chirp_rate_hz_per_s * metadata.pulse_duration_s / 2
        ),
        pulse_ramp_rate_hz_per_s=metadata.chirp_rate_hz_per_s,
        pulse_length_s=metadata.pulse_duration_s,
        batch_lines=128,
        output="valid",
    )
    physical_far_time = 2 * _l0_geometry(raw_path)["far_slant_range_m"] / SPEED_OF_LIGHT_MPS
    range_roi_stop = min(
        compressed.shape[1],
        int(np.floor((physical_far_time - range_times[0]) * metadata.sample_rate_hz)) + 1,
    )
    config = Config(
        azimuth_block_size_lines=azimuth_lines,
        num_range_blocks=num_range_blocks,
        azimuth_placement="spacing",
        azimuth_spacing_lines=azimuth_lines,
        unwrap_fft_length=4096,
        polynomial_degree=2,
        rms_threshold_hz=20.0,
        outlier_sigma=3.5,
        unwrap_weighting="coherence",
        fit_weighting="coherence",
        accc_range_weighting="power",
        range_roi_stop=range_roi_stop,
    )
    estimator = Estimator(metadata.prf_hz, config)
    azimuth_blocks, range_blocks = estimator.build_layout(
        n_azimuth_lines=azimuth_lines,
        n_range_samples=compressed.shape[1],
        slant_range_times_s=range_times,
    )
    return estimator.estimate_block(
        compressed,
        azimuth_blocks[0],
        range_blocks,
        t0_s=range_times[0],
        geometry_dc_provider=lambda _azimuth_time, times: geometry_doppler(raw_path, times),
    )


def compare(estimated, l1_reference: L1DopplerReference) -> dict:
    """Tạo record tương thích với ``dce_plotting.plot_comparisons``."""

    range_times = np.linspace(estimated.range_times_s[0], estimated.range_times_s[-1], 512)
    slant_range_m = range_times * SPEED_OF_LIGHT_MPS / 2
    estimated_hz = estimated.evaluate(range_times)
    reference_hz = l1_reference.evaluate(slant_range_m)
    geometry_hz = np.polynomial.polynomial.polyval(
        range_times - estimated.t0_s, estimated.geometry_dc_polynomial
    )
    return {
        "record": "RADARSAT-1",
        "range_times_s": range_times,
        "annotation_hz": reference_hz,
        "estimated_hz": estimated_hz,
        "geometry_hz": geometry_hz,
        "fit_range_times_s": estimated.range_times_s,
        "fit_points_hz": estimated.fine_absolute_hz,
        "fit_valid_mask": estimated.valid_mask,
        "rmse_hz": float(np.sqrt(np.mean((estimated_hz - reference_hz) ** 2))),
        "fit_rms_hz": estimated.rms_error_hz,
    }


__all__ = [
    "L1DopplerReference", "compare", "estimate", "geometry_doppler", "read_l1_reference"
]
