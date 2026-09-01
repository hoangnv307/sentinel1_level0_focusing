"""Bộ đọc RADARSAT-1 Level-0 CEOS.

Layout và phép đổi mã I/Q theo ``read_CEOS_raw.m`` của UBC/SmallC1oud.
Các tham số thay đổi theo beam được giải từ auxiliary record 50 byte.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import struct

import numpy as np


SPEED_OF_LIGHT_MPS = 299_792_458.0
MASTER_OSCILLATOR_HZ = 129.2683e6
REPLICA_DURATION_S = 44.559e-6
PULSE_DURATION_S = 42e-6
CARRIER_FREQUENCY_HZ = 5.300432e9
BEAM_NAMES = (
    "", "ST1", "ST2", "ST3", "ST4", "ST5", "ST6", "ST7",
    "WD1", "WD2", "WD3", "WD2_Recorded", "EX1", "EX2", "EH4",
    "EX4", "FN1", "FN2", "FN3", "FN4", "FN5",
)
PULSES_IN_AIR = (0, 7, 7, 8, 8, 8, 9, 9, 7, 8, 9, 8, 10, 10, 11, 11, 8, 8, 9, 9, 9)


@dataclass(frozen=True)
class RawMetadata:
    line_count: int
    range_samples: int
    sample_rate_hz: float
    prf_hz: float
    range_start_time_s: float
    pulse_duration_s: float
    chirp_rate_hz_per_s: float
    carrier_frequency_hz: float
    doppler_centroid_hz: float
    doppler_centroid_coefficients: tuple[float, float, float]
    effective_velocity_mps: float
    beam: str
    agc_db: float


def _record_header(file) -> tuple[int, int, int] | None:
    header = file.read(12)
    if not header:
        return None
    if len(header) != 12:
        raise ValueError("CEOS record header bị cắt ngắn")
    sequence, record_type, length = struct.unpack(">III", header)
    if length < 12:
        raise ValueError(f"CEOS record {sequence} có độ dài không hợp lệ: {length}")
    return sequence, record_type, length


def _descriptor(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = _record_header(file)
        if header is None:
            raise ValueError("File CEOS rỗng")
        _, _, length = header
        body = file.read(length - 12)
    try:
        line_count = int(body[168:174])
        line_prefix_bytes = 12 + int(body[264:268])
    except ValueError as error:
        raise ValueError("Không đọc được CEOS file descriptor") from error
    return line_count, line_prefix_bytes


def _auxiliary(aux: bytes) -> dict[str, float | int | bool | str]:
    if len(aux) != 50 or aux[:4] != b"5.\xf8S":
        raise ValueError("RADARSAT-1 auxiliary record không hợp lệ")

    adc_rate = (aux[22] & 0x30) >> 4
    if adc_rate > 2:
        raise ValueError(f"Mã ADC RADARSAT-1 không hỗ trợ: {adc_rate}")
    time_base_s = (4.0, 7.0, 10.0)[adc_rate] / MASTER_OSCILLATOR_HZ
    prf_code = (aux[25] << 5) | ((aux[26] & 0xF8) >> 3)
    dwp_code = (aux[27] << 4) | ((aux[28] & 0xF0) >> 4)
    rx_samples = (1 + ((aux[29] << 4) | ((aux[30] & 0xF0) >> 4))) * 6
    beam_number = ((aux[18] << 8) | aux[19]) & 0x1F
    agc = aux[49] & 0x3F
    return {
        "sample_rate_hz": 1.0 / time_base_s,
        "prf_hz": 1.0 / ((2 + prf_code) * 6 * time_base_s),
        "dwp_s": (5 + dwp_code) * 6 * time_base_s,
        "range_samples": rx_samples,
        "replica_samples": int(REPLICA_DURATION_S / time_base_s),
        "has_replica": bool(aux[49] & 0x40),
        "beam_number": beam_number,
        "beam": BEAM_NAMES[beam_number] if beam_number < len(BEAM_NAMES) else str(beam_number),
        "agc_db": agc - 24 if agc > 31 else agc,
    }


def _meta_value(text: str, name: str, default: float) -> float:
    match = re.search(rf"^\s*{name}\s*=\s*\(?([^,\s\)]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else default


def _doppler_coefficients(raw_path: Path, text: str) -> tuple[float, float, float]:
    leader = raw_path.with_suffix(".ldr")
    if leader.exists():
        data = leader.read_bytes()
        first_record_length = struct.unpack(">I", data[8:12])[0]
        start = first_record_length
        try:
            return tuple(
                float(data[start + offset : start + offset + 16])
                for offset in (1478, 1494, 1510)
            )
        except ValueError:
            pass
    match = re.search(r"^\s*DOPPLER_FREQ\s*=\s*\(([^)]+)\)", text, re.MULTILINE)
    if match:
        values = tuple(float(value) for value in match.group(1).split(","))
        if len(values) == 3:
            return values
    return (0.0, 0.0, 0.0)


def read_metadata(raw_path: str | Path, meta_path: str | Path | None = None) -> RawMetadata:
    """Đọc các tham số cần cho giải mã và chirp scaling."""

    raw_path = Path(raw_path)
    line_count, line_prefix_bytes = _descriptor(raw_path)
    with raw_path.open("rb") as file:
        descriptor = _record_header(file)
        assert descriptor is not None
        file.seek(descriptor[2] - 12, 1)
        header = _record_header(file)
        if header is None:
            raise ValueError("File CEOS không có signal data record")
        body = file.read(header[2] - 12)
    aux = _auxiliary(body[line_prefix_bytes - 12 : line_prefix_bytes - 12 + 50])

    if meta_path is None:
        candidate = raw_path.with_suffix(".meta")
        meta_path = candidate if candidate.exists() else None
    text = Path(meta_path).read_text() if meta_path else ""
    slant_range_m = 1_000.0 * _meta_value(text, "SL_RNG_1ST_PIX", 0.0)
    beam_number = int(aux["beam_number"])
    if slant_range_m:
        range_start_time_s = 2 * slant_range_m / SPEED_OF_LIGHT_MPS
    else:
        range_start_time_s = (
            float(aux["dwp_s"])
            + PULSES_IN_AIR[beam_number] / float(aux["prf_hz"])
            - 4.04e-6
        )

    vx = _meta_value(text, "X_VELOCITY", 0.0)
    vy = _meta_value(text, "Y_VELOCITY", 0.0)
    vz = _meta_value(text, "Z_VELOCITY", 0.0)
    ground_velocity = _meta_value(text, "SWATH_SPEED", 0.0)
    orbital_velocity = (vx * vx + vy * vy + vz * vz) ** 0.5
    effective_velocity = (orbital_velocity * ground_velocity) ** 0.5 if ground_velocity else 7062.0
    sample_rate_hz = float(aux["sample_rate_hz"])
    chirp_rate = -7.214e11 if sample_rate_hz > 30e6 else (-4.1619e11 if sample_rate_hz > 15e6 else -2.7931e11)
    doppler_coefficients = _doppler_coefficients(raw_path, text)
    center_pixel = int(aux["range_samples"]) / 2
    doppler_centroid = sum(value * center_pixel**power for power, value in enumerate(doppler_coefficients))

    return RawMetadata(
        line_count=line_count,
        range_samples=int(aux["range_samples"]),
        sample_rate_hz=sample_rate_hz,
        prf_hz=float(aux["prf_hz"]),
        range_start_time_s=range_start_time_s,
        pulse_duration_s=PULSE_DURATION_S,
        chirp_rate_hz_per_s=chirp_rate,
        carrier_frequency_hz=CARRIER_FREQUENCY_HZ,
        doppler_centroid_hz=doppler_centroid,
        doppler_centroid_coefficients=doppler_coefficients,
        effective_velocity_mps=effective_velocity,
        beam=str(aux["beam"]),
        agc_db=float(aux["agc_db"]),
    )


def decode(
    raw_path: str | Path,
    *,
    first_line: int = 0,
    line_count: int | None = None,
    first_sample: int = 0,
    sample_count: int | None = None,
    apply_agc: bool = False,
) -> np.ndarray:
    """Giải mã một cửa sổ Level-0 thành ma trận ``complex64`` [azimuth, range]."""

    raw_path = Path(raw_path)
    metadata = read_metadata(raw_path)
    if line_count is None:
        line_count = metadata.line_count - first_line
    if sample_count is None:
        sample_count = metadata.range_samples - first_sample
    if first_line < 0 or line_count < 1 or first_line + line_count > metadata.line_count:
        raise ValueError("Cửa sổ azimuth nằm ngoài dữ liệu")
    if first_sample < 0 or sample_count < 1 or first_sample + sample_count > metadata.range_samples:
        raise ValueError("Cửa sổ range nằm ngoài dữ liệu")

    _, line_prefix_bytes = _descriptor(raw_path)
    output = np.empty((line_count, sample_count), dtype=np.complex64)
    with raw_path.open("rb") as file:
        descriptor = _record_header(file)
        assert descriptor is not None
        file.seek(descriptor[2] - 12, 1)
        output_line = 0
        for line in range(metadata.line_count):
            record_start = file.tell()
            header = _record_header(file)
            if header is None:
                raise ValueError(f"Thiếu signal data record tại dòng {line}")
            _, _, record_length = header
            if line < first_line or line >= first_line + line_count:
                file.seek(record_start + record_length)
                continue

            file.seek(record_start + line_prefix_bytes)
            aux_bytes = file.read(50)
            aux = _auxiliary(aux_bytes)
            if int(aux["range_samples"]) != metadata.range_samples:
                raise ValueError("Số mẫu range thay đổi trong cửa sổ được chọn")
            replica_bytes = 2 * int(aux["replica_samples"]) if aux["has_replica"] else 0
            file.seek(replica_bytes + 2 * first_sample, 1)
            packed = np.frombuffer(file.read(2 * sample_count), dtype=np.uint8)
            if packed.size != 2 * sample_count or np.any(packed > 15):
                raise ValueError(f"I/Q data không hợp lệ tại dòng {line}")
            levels = np.where(packed < 8, 2 * packed + 1, 2 * packed.astype(np.int16) - 31).astype(np.float32)
            decoded = levels[0::2] + 1j * levels[1::2]
            if apply_agc:
                decoded *= np.float32(1.5 * 10 ** (float(aux["agc_db"]) / 20))
            output[output_line] = decoded
            output_line += 1
            file.seek(record_start + record_length)
    return output
