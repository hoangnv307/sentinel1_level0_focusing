"""Input DCE chẩn đoán dùng hiệu chỉnh tính sẵn từ L1, không dùng cho focus L0 độc lập."""

import numpy as np
from astropy.time import Time
from scipy.interpolate import interp1d

from ..range_processing.range_reference_function import from_coefficients
from ..common.receiver_gain_compensation import calculate_gain


PRODUCT_SPECIFICATION = (
    "https://sentinels.copernicus.eu/documents/247904/1877131/"
    "Sentinel-1-Product-Specification-18052021.pdf"
)


def _numbers(element, count=None):
    if element is None:
        raise ValueError("Thiếu vector hiệu chỉnh trong XML.")
    values = np.array([float(x) for x in (element.text or "").split()])
    expected = int(element.get("count", len(values))) if count is None else count
    if values.size != expected or not np.isfinite(values).all() or not values.size:
        raise ValueError(f"Vector {element.tag} không hợp lệ hoặc sai số phần tử.")
    return values


def _flag(parent, path):
    value = parent.findtext(path)
    if value not in ("true", "false"):
        raise ValueError(f"Thiếu hoặc sai cờ {path}.")
    return value == "true"


def _single(elements, label):
    if len(elements) != 1:
        raise ValueError(f"Demo yêu cầu đúng một {label}.")
    return elements[0]


def _time(element):
    return float(Time(element.findtext("azimuthTime"), scale="utc").gps)


def _interpolate(nodes, values, target):
    nodes = np.asarray(nodes)
    if nodes.size < 2 or np.any(np.diff(nodes) <= 0):
        raise ValueError("Cần ít nhất hai node hiệu chỉnh tăng nghiêm ngặt.")
    return interp1d(
        nodes - nodes[0], values, axis=0, bounds_error=False, fill_value="extrapolate"
    )(np.asarray(target) - nodes[0])


class ReferenceInput:
    """Hiệu chỉnh I/Q → RX → PG → RRF → EAP/RSL, theo DAD §6.1, §9.2–9.6.

    PG annotation là reciprocal của correction, kèm gain offset giữa swath.
    Chỉ phục hồi drift tương đối; scale complex hằng không ảnh hưởng lag-1 DC.
    """

    def __init__(self, annotation, auxiliary):
        header = annotation.find("adsHeader")
        if header is None or [header.findtext(k) for k in ("missionId", "swath", "polarisation")] != ["S1A", "S4", "VV"]:
            raise ValueError("Hiệu chỉnh tham chiếu chỉ hỗ trợ S1A/S4/VV.")
        processing = annotation.find("imageAnnotation/processingInformation")
        if processing is None or processing.findtext("rfiMitigationPerformed") != "Never":
            raise ValueError("Demo chưa tái hiện RFI mitigation của L1.")
        if not _flag(processing, "rawDataAnalysisUsed"):
            raise ValueError("L1 không dùng raw-data analysis để lấy I/Q bias.")
        analysis = _single(annotation.findall("generalAnnotation/rawDataAnalysisList/rawDataAnalysis"), "raw analysis")
        self.bias = complex(
            float(analysis.findtext("support/iBiasUsedForCorrection")),
            float(analysis.findtext("support/qBiasUsedForCorrection")),
        )
        self.time_offset_s = float(annotation.findtext("imageAnnotation/imageInformation/zeroDopMinusAcqTime"))
        info = _single([
            e for e in annotation.findall("generalAnnotation/replicaInformationList/replicaInformation")
            if e.findtext("swath") == "S4"
        ], "replica information S4")
        reference = info.find("referenceReplica")
        if reference is None or reference.findtext("chirpSource") != "Nominal" or reference.findtext("pgSource") != "Extracted":
            raise ValueError("Demo yêu cầu nominal chirp và extracted PG như cảnh S4 hiện tại.")
        self.annotated_amplitude = _numbers(reference.find("amplitudeCoefficients"), 4)
        self.annotated_phase = _numbers(reference.find("phaseCoefficients"), 4)
        self.amplitude = np.asarray(auxiliary["nominal_amplitude_coefficients"], dtype=float)
        self.phase = np.asarray(auxiliary["nominal_phase_coefficients_cycles"], dtype=float)
        if any(c.shape != (4,) or not np.isfinite(c).all() for c in (self.amplitude, self.phase)):
            raise ValueError("Cần bốn hệ số amplitude/phase nominal hữu hạn từ AUX_INS.")
        self.internal_delay_s = float(reference.findtext("timeDelay"))
        self.replica_gain = complex(float(reference.findtext("gain/re")), float(reference.findtext("gain/im")))
        valid = [
            e for e in info.findall("replicaList/replica")
            if all(_flag(e, name) for name in (
                "reconstructedReplicaValidFlag", "relativePgProductValidFlag", "absolutePgProductValidFlag"
            ))
        ]
        self.pg_times = np.array([_time(e) for e in valid])
        pg_amplitude = np.array([float(e.findtext("pgProductAmplitude")) for e in valid])
        pg_phase = np.array([float(e.findtext("pgProductPhase")) for e in valid])
        if len(valid) < 2 or np.any(np.diff(self.pg_times) <= 0) or np.any(pg_amplitude <= 0):
            raise ValueError("Không đủ hai PG valid tăng theo thời gian với amplitude dương.")
        self.pg_annotated = pg_amplitude * np.exp(1j * pg_phase)
        # Annotation is already inverse instrument PG: normalize directly, do not invert again.
        self.pg_corrections = self.pg_annotated / self.pg_annotated[0]
        self.rx_enabled = auxiliary["correct_rx_variation"]
        self.rx_enabled_annotation = _flag(processing, "rxVariationCorrectionApplied")
        self.rx_trend = np.asarray(auxiliary["rx_gain_trend"], dtype=float)
        self.rx_overshoot = np.asarray(auxiliary["rx_gain_overshoot"], dtype=float)
        if self.rx_trend.shape != (8,) or self.rx_overshoot.shape != (5,):
            raise ValueError("Sai số hệ số receiver variation (cần 8 và 5).")
        self.eap_enabled = _flag(processing, "antennaElevationPatternApplied")
        self.rsl_enabled = _flag(processing, "rangeSpreadingLossCompensationApplied")
        self.reference_range_m = float(processing.findtext("referenceRange"))
        self.rrf_spectrum = processing.findtext("rrfSpectrum")
        self.patterns = [e for e in annotation.findall("antennaPattern/antennaPatternList/antennaPattern") if e.findtext("swath") == "S4"]
        self.pattern_times = np.array([_time(e) for e in self.patterns])
        self.pattern_ranges, self.pattern_values = [], []
        for pattern in self.patterns:
            tau = _numbers(pattern.find("slantRangeTime"))
            pairs = _numbers(pattern.find("elevationPattern"), 2 * tau.size).reshape(-1, 2)
            values = pairs[:, 0] + 1j * pairs[:, 1]
            if tau.size < 2 or np.any(np.diff(tau) <= 0) or np.any(np.abs(values) == 0):
                raise ValueError("Range grid hoặc complex EAP không hợp lệ.")
            self.pattern_ranges.append(tau)
            self.pattern_values.append(values)
        if self.eap_enabled and (len(self.patterns) < 2 or np.any(np.diff(self.pattern_times) <= 0)):
            raise ValueError("Cần bank EAP có ít nhất hai thời điểm tăng nghiêm ngặt.")
        scalars = [self.bias, self.internal_delay_s, self.replica_gain, self.time_offset_s, self.reference_range_m]
        if (not np.isfinite(scalars).all() or self.reference_range_m <= 0
                or self.internal_delay_s < 0 or not np.isfinite(self.pg_annotated).all()):
            raise ValueError("Hiệu chỉnh tham chiếu chứa scalar không hợp lệ.")

    def receiver_gain(self, n_samples, sample_rate_hz, swst_s, pulse_length_s, suppressed_s):
        """Power GV (§9.2.3); complex samples require division by sqrt(GV)."""
        return calculate_gain(
            n_samples, sample_rate_hz=sample_rate_hz, swst_s=swst_s,
            pulse_length_s=pulse_length_s, suppressed_s=suppressed_s,
            gain_trend_coefficients=self.rx_trend, gain_overshoot_coefficients=self.rx_overshoot,
            enabled=self.rx_enabled,
        )

    def pg_gain(self, acquisition_times_s):
        """Nội/ngoại suy correction complex, không nội suy phase đã wrap (§9.4)."""
        return _interpolate(self.pg_times, self.pg_corrections, np.asarray(acquisition_times_s) + self.time_offset_s)

    def range_reference(self, sample_rate_hz, pulse_length_s, fft_length, *, internal_delay_s=None):
        return from_coefficients(
            sample_rate_hz=sample_rate_hz, pulse_length_s=pulse_length_s, fft_length=fft_length,
            amplitude_coefficients=self.amplitude, phase_coefficients_cycles=self.phase,
            internal_delay_s=self.internal_delay_s if internal_delay_s is None else internal_delay_s,
            complex_gain=self.replica_gain,
        )

    def range_gain_bank(self, tau):
        """Nội suy complex LUT, căn magnitude giữ phase, rồi nghịch đảo EAP (§9.5)."""
        tau = np.asarray(tau)
        bank = []
        for grid, pattern in zip(self.pattern_ranges, self.pattern_values):
            if tau[0] < grid[0] - 1e-12 or tau[-1] > grid[-1] + 1e-12:
                raise ValueError("Range support nằm ngoài bank EAP annotation.")
            value = _interpolate(grid, pattern, tau)
            if np.any(np.abs(value) == 0):
                raise ValueError("EAP nội suy bằng 0.")
            bank.append(np.sqrt(np.abs(value)) / value if self.eap_enabled else np.ones(tau.size))
        bank = np.asarray(bank)
        if self.rsl_enabled:
            bank *= (tau * 299_792_458.0 / (2 * self.reference_range_m)) ** 1.5
        return bank

    def apply_range_gain(self, data, bank, acquisition_times_s):
        times = np.asarray(acquisition_times_s) + self.time_offset_s
        if times[0] < self.pattern_times[0] - 1e-5 or times[-1] > self.pattern_times[-1] + 1e-5:
            raise ValueError("Echo nằm ngoài time support bank EAP tham chiếu.")
        data *= _interpolate(self.pattern_times, bank, times)

    def summary(self):
        return {
            "source": "Annotation L1 S4/VV + receiver coefficients AUX_INS; chế độ chẩn đoán được người dùng yêu cầu.",
            "product_specification": PRODUCT_SPECIFICATION,
            "iq_bias": [self.bias.real, self.bias.imag],
            "rx_variation_applied": self.rx_enabled,
            "rx_variation_annotated_l1": self.rx_enabled_annotation,
            "rx_variation_flag_source": "AUX_PP1 commonProcParams/correctRxVariationFlag",
            "rx_gain_trend": self.rx_trend.tolist(), "rx_gain_overshoot": self.rx_overshoot.tolist(),
            "replica_amplitude_coefficients": self.amplitude.tolist(),
            "replica_phase_coefficients_cycles": self.phase.tolist(),
            "replica_coefficients_source": "AUX_INS nominal S4/VV; L1 coefficients are diagnostic only",
            "annotated_replica_phase_coefficients_cycles": self.annotated_phase.tolist(),
            "replica_gain": [self.replica_gain.real, self.replica_gain.imag],
            "internal_delay_compensation_s": self.internal_delay_s,
            "rrf_spectrum_annotation": self.rrf_spectrum,
            "rrf_spectrum_exactly_reproduced": False,
            "pg_times_gps_s": self.pg_times.tolist(),
            "pg_annotated": np.column_stack((self.pg_annotated.real, self.pg_annotated.imag)).tolist(),
            "pg_relative_correction": np.column_stack((self.pg_corrections.real, self.pg_corrections.imag)).tolist(),
            "pg_absolute_scale_recovered": False,
            "eap_applied": self.eap_enabled, "eap_times_gps_s": self.pattern_times.tolist(),
            "range_spreading_loss_applied": self.rsl_enabled, "reference_range_m": self.reference_range_m,
            "not_applied": ["Hamming/AAP (post-processing)", "noise subtraction (không áp cho complex SLC)", "RFI mitigation (L1 Never)"],
        }
