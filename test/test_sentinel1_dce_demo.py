"""Kiểm tra suy layout L1 và DCE độc lập bằng tín hiệu biết trước."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from astropy.time import Time
import numpy as np

from sentinel1_processing.doppler_centroid import estimation
from sentinel1_processing.azimuth_pre_processing.range import compression
from sentinel1_processing.utils.dce_reference_input import ReferenceInput


spec = importlib.util.spec_from_file_location(
    "sentinel1_dce_demo",
    Path(__file__).parents[1] / "workflows/sentinel-1/estimate_doppler_centroid.py",
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

audit_spec = importlib.util.spec_from_file_location(
    "sentinel1_dce_audit", Path(__file__).parents[1] / "workflows/sentinel-1/audit_doppler_centroid.py"
)
audit = importlib.util.module_from_spec(audit_spec)
with patch.dict("sys.modules", {"estimate_doppler_centroid": demo}):
    audit_spec.loader.exec_module(audit)

rrf_spec = importlib.util.spec_from_file_location(
    "sentinel1_rrf_check", Path(__file__).parents[1] / "workflows/sentinel-1/check_range_reference.py"
)
rrf_check = importlib.util.module_from_spec(rrf_spec)
with patch.dict("sys.modules", {"estimate_doppler_centroid": demo, "audit_doppler_centroid": audit}):
    rrf_spec.loader.exec_module(rrf_check)


class Sentinel1DceDemoTest(unittest.TestCase):
    def setUp(self):
        self.pri = 0.01
        self.rate = 1e6
        epoch = float(Time("2025-09-03T18:36:28", scale="utc").gps)
        eta = epoch + np.arange(240) * self.pri
        tau = 0.005 + np.arange(64) / self.rate
        tone = np.exp(2j * np.pi * 7 * np.arange(240) * self.pri)
        self.segments = [
            estimation.Segment(np.tile(tone[:160, None], (1, 64)), tau, eta[:160]),
            estimation.Segment(np.tile(tone[160:, None], (1, 64)), tau + 4 / self.rate, eta[160:]),
        ]
        self.reference = ET.fromstring("""
            <product><adsHeader><missionId>S1A</missionId><swath>S4</swath>
            <polarisation>VV</polarisation><productType>SLC</productType></adsHeader>
            <imageAnnotation><imageInformation><zeroDopMinusAcqTime>0.3</zeroDopMinusAcqTime>
            </imageInformation></imageAnnotation><dopplerCentroid><dcEstimateList/>
            </dopplerCentroid></product>
        """)
        for start in (0, 140):
            record = ET.SubElement(self.reference.find("dopplerCentroid/dcEstimateList"), "dcEstimate")
            ET.SubElement(record, "t0").text = str(tau[4])
            for field, seconds in (("fineDceAzimuthStartTime", start * self.pri),
                                   ("fineDceAzimuthStopTime", (start + 100) * self.pri)):
                ET.SubElement(record, field).text = Time(
                    epoch + seconds + 0.3, format="gps", precision=9
                ).utc.isot
            points = ET.SubElement(record, "fineDceList")
            for centre in (6, 18, 30, 42, 54):
                point = ET.SubElement(points, "fineDce")
                ET.SubElement(point, "slantRangeTime").text = str(tau[4] + centre / self.rate)
                ET.SubElement(point, "frequency").text = "-999"

    def infer(self):
        return demo.infer_configuration(self.reference, self.segments, self.pri, self.rate, 20)

    def reference_input(self):
        def field(parent, name, value):
            ET.SubElement(parent, name).text = str(value)

        processing = ET.SubElement(self.reference.find("imageAnnotation"), "processingInformation")
        for key in ("rawDataAnalysisUsed", "rxVariationCorrectionApplied", "antennaElevationPatternApplied", "rangeSpreadingLossCompensationApplied"):
            field(processing, key, "true")
        field(processing, "rfiMitigationPerformed", "Never")
        field(processing, "referenceRange", 800000)
        general = ET.SubElement(self.reference, "generalAnnotation")
        analysis = ET.SubElement(ET.SubElement(general, "rawDataAnalysisList"), "rawDataAnalysis")
        support = ET.SubElement(analysis, "support")
        field(support, "iBiasUsedForCorrection", 0.25)
        field(support, "qBiasUsedForCorrection", -0.125)
        info = ET.SubElement(ET.SubElement(general, "replicaInformationList"), "replicaInformation")
        field(info, "swath", "S4")
        reference = ET.SubElement(info, "referenceReplica")
        for key, value in (("chirpSource", "Nominal"), ("pgSource", "Extracted"),
                           ("amplitudeCoefficients", "1 0 0 0"),
                           ("phaseCoefficients", "0 -1000 2000000000 0"),
                           ("timeDelay", 5e-6)):
            field(reference, key, value)
        gain = ET.SubElement(reference, "gain")
        field(gain, "re", 1)
        field(gain, "im", 0)
        epoch = float(self.segments[0].azimuth_times_s[0])
        replicas = ET.SubElement(info, "replicaList")
        for seconds, amplitude, phase in ((0, 1, 0), (3, 2, np.pi / 4)):
            replica = ET.SubElement(replicas, "replica")
            field(replica, "azimuthTime", Time(epoch + seconds + 0.3, format="gps", precision=9).utc.isot)
            field(replica, "pgProductAmplitude", amplitude)
            field(replica, "pgProductPhase", phase)
            for flag in ("reconstructedReplicaValidFlag", "relativePgProductValidFlag", "absolutePgProductValidFlag"):
                field(replica, flag, "true")
        patterns = ET.SubElement(ET.SubElement(self.reference, "antennaPattern"), "antennaPatternList")
        for seconds in (0, 3):
            pattern = ET.SubElement(patterns, "antennaPattern")
            field(pattern, "swath", "S4")
            field(pattern, "azimuthTime", Time(epoch + seconds + 0.3, format="gps", precision=9).utc.isot)
            field(pattern, "slantRangeTime", "0.004 0.005 0.006")
            field(pattern, "elevationPattern", "0 9 0 9 0 9")
        auxiliary = {
            "correct_iq_bias": True, "correct_iq_gain": False, "correct_iq_quadrature": False,
            "rx_gain_trend": [10, 0, 0, 0, 0, 1, 0, 0],
            "rx_gain_overshoot": [0, 0, 0, 0, 0],
            "nominal_amplitude_coefficients": [1, 0, 0, 0],
            "nominal_phase_coefficients_cycles": [0, -1000, 2000000000, 0],
        }
        self.auxiliary = auxiliary
        return ReferenceInput(self.reference, auxiliary)

    def test_precomputed_bias_receiver_and_inverse_pg_recover_known_tone(self):
        correction = self.reference_input()
        eta = np.asarray(self.segments[0].azimuth_times_s)
        receiver = correction.receiver_gain(64, self.rate, 90e-6, 50e-6, 1e-6)
        np.testing.assert_allclose(receiver, 10)
        np.testing.assert_allclose(correction.pg_corrections, [1, 0.5 * np.exp(-1j * np.pi / 4)])
        instrument_gain = 1 / correction.pg_gain(eta)
        tone = np.exp(2j * np.pi * 7 * np.arange(eta.size) * self.pri)
        raw = tone[:, None] * instrument_gain[:, None] * receiver[None, :] + correction.bias
        recovered = (raw - correction.bias) / receiver[None, :] * correction.pg_gain(eta)[:, None]
        dc = -np.angle(estimation.lag1_accc(recovered)) / (2 * np.pi * self.pri)
        np.testing.assert_allclose(dc, 7, atol=1e-10)

    def test_receiver_timing_uses_microseconds_after_suppressed_samples(self):
        correction = self.reference_input()
        correction.rx_trend = np.array([0, 0.1, 0, 0, 0, 1, 0, 0])
        gain = correction.receiver_gain(3, 1e6, 90e-6, 50e-6, 1e-6)
        np.testing.assert_allclose(gain, 10 ** (np.array([4.1, 4.2, 4.3]) / 10))

    def test_eap_preserves_phase_when_taking_square_root_of_magnitude(self):
        correction = self.reference_input()
        tau = np.array([0.0045, 0.0055])
        bank = correction.range_gain_bank(tau)
        expected = -1j / 3 * (tau * 299792458 / (2 * 800000)) ** 1.5
        np.testing.assert_allclose(bank, np.tile(expected, (2, 1)))
        data = np.ones((1, 2), dtype=np.complex64)
        correction.apply_range_gain(data, bank, [self.segments[0].azimuth_times_s[0] + 1])
        np.testing.assert_allclose(data[0], expected, rtol=1e-6)

    def test_reference_filter_removes_positive_hardware_delay(self):
        correction = self.reference_input()
        n = 128
        pulse_length = n / self.rate
        t = (np.arange(n) - (n - 1) / 2) / self.rate
        chirp = np.exp(2j * np.pi * np.polynomial.polynomial.polyval(t, correction.phase))
        raw = np.zeros((1, 512), dtype=np.complex64)
        raw[0, 85:85 + n] = chirp
        compressed, _ = compression.compress(
            raw, np.arange(512) / self.rate, sample_rate_hz=self.rate,
            pulse_length_s=pulse_length, pulse_start_frequency_hz=0,
            pulse_ramp_rate_hz_per_s=0, output="slc",
            range_reference_function=correction.range_reference(self.rate, pulse_length, 1024),
        )
        self.assertEqual(int(np.argmax(abs(compressed[0]))), 80)

    def test_rejects_invalid_pg_instead_of_using_its_values(self):
        correction = self.reference_input()
        flag = self.reference.find("generalAnnotation/replicaInformationList/replicaInformation/replicaList/replica/absolutePgProductValidFlag")
        flag.text = "false"
        with self.assertRaisesRegex(ValueError, "Không đủ hai PG valid"):
            ReferenceInput(self.reference, {
                "correct_iq_bias": True, "correct_iq_gain": False, "correct_iq_quadrature": False,
                "rx_gain_trend": correction.rx_trend, "rx_gain_overshoot": correction.rx_overshoot,
                "nominal_amplitude_coefficients": correction.amplitude,
                "nominal_phase_coefficients_cycles": correction.phase,
            })

    def test_nominal_rrf_uses_aux_instead_of_rounded_l1_coefficients(self):
        correction = self.reference_input()
        expected = correction.range_reference(self.rate, 128 / self.rate, 1024)
        field = self.reference.find("generalAnnotation/replicaInformationList/replicaInformation/referenceReplica/phaseCoefficients")
        field.text = "0 99999 0 0"
        rebuilt = ReferenceInput(self.reference, self.auxiliary)
        np.testing.assert_array_equal(rebuilt.range_reference(self.rate, 128 / self.rate, 1024), expected)
        self.assertEqual(rebuilt.annotated_phase[1], 99999)

    def test_candidate_spectra_preserve_peak_coordinate_energy_and_delay(self):
        correction = self.reference_input()
        correction.phase = np.array([0., 1., 96., 0.])
        correction.internal_delay_s = 7/128
        for mode in ("flat", "cosine"):
            for n in (63, 64):
                with self.subTest(mode=mode, pulse_samples=n):
                    h = rrf_check.candidate_reference(correction, 128., n/128, 512, mode)
                    self.assertAlmostEqual(float(np.mean(abs(h.astype(complex))**2)), 1., places=6)
                    t = (np.arange(n)-(n-1)/2)/128
                    raw = np.zeros((1, 256), dtype=np.complex64)
                    raw[0, 30:30+n] = np.exp(2j*np.pi*(t+96*t**2))
                    data, axis = compression.compress(
                        raw, np.arange(256)/128, sample_rate_hz=128., pulse_length_s=n/128,
                        pulse_start_frequency_hz=0., pulse_ramp_rate_hz_per_s=0.,
                        range_reference_function=h, output="slc")
                    self.assertEqual(int(np.argmax(abs(data[0]))), 23)
                    self.assertEqual(axis[0], 0.)
                    if mode == "cosine":
                        self.assertAlmostEqual(abs(h[256]), 0., places=12)

    def test_candidate_rejects_unsupported_chirp_instead_of_ignoring_coefficients(self):
        correction = self.reference_input()
        correction.phase[3] = 1.
        with self.assertRaisesRegex(ValueError, "nominal LFM"):
            rrf_check.candidate_reference(correction, self.rate, 128/self.rate, 1024, "cosine")

    def test_layout_and_dc_use_l0_signal_instead_of_l1_frequencies(self):
        config, starts, t0, _ = self.infer()
        self.assertEqual(starts, [0, 140])
        self.assertEqual(config.azimuth_block_size_lines, 100)
        self.assertEqual(config.range_block_size_samples, 12)
        self.assertEqual(config.range_roi_stop, 60)
        records = estimation.Estimator(1 / self.pri, config).estimate_segments(
            self.segments, custom_azimuth_starts=starts, dce_range_start_s=t0,
            geometry_dc_provider=lambda _time, tau: np.zeros_like(tau),
        )
        self.assertEqual(len(records), 2)
        for record in records:
            np.testing.assert_allclose(record.fine_baseband_hz, 7, atol=1e-10)
            np.testing.assert_allclose(record.evaluate(record.range_times_s), 7, atol=1e-10)
            self.assertTrue(record.absolute_ambiguity_resolved)

    def test_rejects_range_layout_that_does_not_match_l0_support(self):
        point = self.reference.find("dopplerCentroid/dcEstimateList/dcEstimate/fineDceList/fineDce/slantRangeTime")
        point.text = str(float(point.text) + 1 / self.rate)
        with self.assertRaisesRegex(ValueError, "Range centres"):
            self.infer()

    def test_rejects_time_gap_instead_of_correlating_across_it(self):
        segment = self.segments[1]
        self.segments[1] = estimation.Segment(
            segment.range_compressed, segment.slant_range_times_s,
            np.asarray(segment.azimuth_times_s) + 0.1,
        )
        with self.assertRaisesRegex(ValueError, "không có gap"):
            self.infer()

    def test_hardware_delay_guard_keeps_dce_inside_valid_range_support(self):
        trimmed = [estimation.Segment(
            s.range_compressed[:, :-5], np.asarray(s.slant_range_times_s)[:-5], s.azimuth_times_s
        ) for s in self.segments]
        config, starts, t0, _ = demo.infer_configuration(
            self.reference, trimmed, self.pri, self.rate, 20, far_range_throwaway_samples=5
        )
        self.assertEqual(config.range_roi_stop, 55)
        self.assertEqual(starts, [0, 140])
        records = estimation.Estimator(1 / self.pri, config).estimate_segments(
            trimmed, custom_azimuth_starts=starts, dce_range_start_s=t0,
            geometry_dc_provider=lambda _time, tau: np.zeros_like(tau),
        )
        for record in records:
            self.assertLessEqual(record.range_blocks[-1].stop_sample, 4 + 55)
            np.testing.assert_allclose(record.fine_baseband_hz, 7, atol=1e-10)

    def test_signed_aliases_and_tiny_amplitudes_preserve_lag1_frequency(self):
        for amplitude in (1., 1e-12):
            for frequency in (-49., -7., 7., 49., 107.):
                with self.subTest(amplitude=amplitude, frequency=frequency):
                    tone = amplitude * np.exp(2j * np.pi * frequency * np.arange(240) * self.pri)
                    signal = np.tile(tone[:, None], (1, 8))
                    measured = -np.angle(estimation.lag1_accc(signal)) / (2 * np.pi * self.pri)
                    expected = (frequency + 50) % 100 - 50
                    np.testing.assert_allclose(measured, expected, atol=1e-10)

    def test_rejects_eap_extrapolation_and_zero_receiver_denominator(self):
        correction = self.reference_input()
        with self.assertRaisesRegex(ValueError, "Range support"):
            correction.range_gain_bank(np.array([0.0039, 0.005]))
        with self.assertRaisesRegex(ValueError, "time support"):
            correction.apply_range_gain(np.ones((1, 2), complex), np.ones((2, 2)),
                                        [correction.pattern_times[-1] + 1])
        correction.rx_trend[5:] = 0
        with self.assertRaisesRegex(ValueError, "mẫu số"):
            correction.receiver_gain(64, self.rate, 90e-6, 50e-6, 1e-6)

    def test_audit_comparison_masks_points_outside_reference_support(self):
        config, starts, t0, _ = self.infer()
        record = estimation.Estimator(1 / self.pri, config).estimate_segments(
            self.segments, custom_azimuth_starts=starts, t0_s=t0, dce_range_start_s=t0,
            geometry_dc_provider=lambda _time, tau: np.zeros_like(tau),
        )[0]
        annotated = self.reference.find("dopplerCentroid/dcEstimateList/dcEstimate")
        points = annotated.find("fineDceList")
        points.remove(points[0])
        points.remove(points[-1])
        for p in points:
            p.find("frequency").text = "7"
        ET.SubElement(annotated, "dataDcPolynomial").text = "7 0 0"
        ET.SubElement(annotated, "geometryDcPolynomial").text = "0 0 0"
        metrics = audit.compare(record, annotated)
        self.assertEqual(metrics["compared_range_points"], 3)
        self.assertLess(metrics["fine_rmse_hz"], 1e-10)
        self.assertLess(metrics["polynomial_rmse_hz"], 1e-10)
        points.remove(points[-1])
        with self.assertRaisesRegex(ValueError, "Không đủ giao"):
            audit.compare(record, annotated)


if __name__ == "__main__":
    unittest.main()
