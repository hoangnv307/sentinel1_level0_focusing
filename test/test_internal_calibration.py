"""Calibration/PG: mô hình thuận độc lập, validity và timing từ L0 thật."""

from pathlib import Path
import unittest

import numpy as np
import sentinel1decoder

from sentinel1_processing.pre_processing import internal_calibration as cal
from sentinel1_processing.utils.pg_calibration_input import analyze_calibration
from sentinel1_processing.doppler_centroid.estimation import lag1_accc


class InternalCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.signal = np.exp(2j*np.pi*np.arange(64)**2/200)
        self.qc = dict(pg_amp_std_fraction=1., pg_phase_std_fraction=100., max_pg_amp_error=100.,
                       max_pg_phase_error=6.28, invalid_pg_max_fraction=.5, use_model=False)

    def test_pcc_cancels_leakage_and_honors_one_based_order_and_sign(self):
        leakage = 3+2j
        pulses = np.array([self.signal+leakage, -self.signal+leakage])
        np.testing.assert_allclose(cal.extract_pulse(pulses, order=[1, 2], method="PCC2"), self.signal)
        np.testing.assert_allclose(cal.extract_pulse(pulses, order=[2, 1], method="PCC2"), -self.signal)

    def test_average_uses_configured_pulses_instead_of_fixed_eight(self):
        raw = np.array([self.signal*k for k in (1, 3, 100, 5)])
        np.testing.assert_allclose(cal.extract_pulse(raw, order=[2, 4], method="Average"), 4*self.signal)

    def test_rejects_missing_duplicate_invalid_order_and_unsupported_method(self):
        raw = np.array([self.signal]*4)
        for order, method in (([0, 1], "PCC2"), ([4, 5], "PCC2"), ([1, 1], "PCC2"),
                              ([1.5, 2.], "PCC2"), ([1], "PCC2"), ([], "Average"), ([1], "Other")):
            with self.subTest(order=order, method=method), self.assertRaises(ValueError):
                cal.extract_pulse(raw, order=order, method=method)

    def test_reconstruction_matches_independent_delta_transfer_functions(self):
        delta = np.eye(1, 64, dtype=complex)[0]
        actual = cal.reconstruct_replica(self.signal, 2*delta, 4*delta, 3*delta, 5*delta,
                                         denominator_relative_floor=0.)
        np.testing.assert_allclose(actual, .3*self.signal, atol=1e-15)

    def test_reconstruction_rejects_missing_spectral_support_and_shape_mismatch(self):
        zero = np.zeros(64, complex)
        for pulses in ((self.signal, self.signal, zero, self.signal, self.signal),
                       (self.signal[:-1], self.signal, self.signal, self.signal, self.signal)):
            with self.assertRaises(ValueError):
                cal.reconstruct_replica(*pulses, denominator_relative_floor=0.)

    def test_zoom_parabola_recovers_signed_fractional_lag_and_complex_peak(self):
        frequencies = np.fft.fftfreq(64)
        reference = np.fft.ifft(np.exp(-(frequencies/.1)**2)*np.exp(1j*np.arange(64)))
        gain = .8*np.exp(.4j)
        for lag in (-2.3, 0., .3, 1.5):
            signal = gain*np.fft.ifft(np.fft.fft(reference)*np.exp(-2j*np.pi*frequencies*lag))
            peak, measured = cal.correlation_peak(signal, reference, oversampling_factor=16)
            self.assertAlmostEqual(measured, lag, places=4)
            np.testing.assert_allclose(peak, gain*np.vdot(reference, reference), rtol=2e-5)

    def test_pg_is_inverse_instrument_gain_with_first_valid_replica_reference(self):
        gains = np.array([1., .8*np.exp(.4j), 1.1*np.exp(-.3j)])
        replicas = gains[:, None]*self.signal
        products, peaks, lags = cal.pg_products(replicas, pg_reference=707.+2j, oversampling_factor=16)
        np.testing.assert_allclose(products/products[0], 1/gains, atol=1e-14)
        np.testing.assert_allclose(peaks, gains*np.vdot(self.signal, self.signal), rtol=1e-14)
        np.testing.assert_allclose(lags, 0., atol=1e-12)

    def test_constant_pg_reference_scale_cancels_in_relative_drift(self):
        replicas = np.array([self.signal, self.signal*(.9+.1j)])
        a, _, _ = cal.pg_products(replicas, pg_reference=1., oversampling_factor=16)
        b, _, _ = cal.pg_products(replicas, pg_reference=707.+3j, oversampling_factor=16)
        np.testing.assert_allclose(a/a[0], b/b[0], atol=1e-14)

    def test_relative_qc_discards_outlier_and_uses_only_valid_nodes(self):
        quality, chosen = cal.validate_pg(np.array([1, 1, 5], complex), np.ones(3, complex), **self.qc)
        self.assertEqual(quality["valid"], [True, True, False])
        self.assertEqual(quality["selected"], [True, True, False])
        self.assertEqual(quality["source"], "extracted")
        np.testing.assert_array_equal(chosen, [1, 1, 5])

    def test_invalid_fraction_and_explicit_model_flag_control_fallback(self):
        for override in ({"invalid_pg_max_fraction": .2}, {"use_model": True}):
            quality, chosen = cal.validate_pg(np.array([1, 1, 5], complex), np.ones(3, complex),
                                              **{**self.qc, **override})
            self.assertEqual(quality["source"], "model")
            self.assertEqual(quality["selected"], [True]*3)
            np.testing.assert_array_equal(chosen, np.ones(3))

    def test_absolute_qc_and_phase_wrap_are_distinct_from_relative_qc(self):
        values = np.exp(1j*np.array([np.pi-.01, -np.pi+.01]))
        quality, _ = cal.validate_pg(values, np.full(2, -1+0j), **{**self.qc, "max_pg_phase_error": .05})
        self.assertEqual(quality["absolute_valid"], [True, True])
        quality, _ = cal.validate_pg(np.array([1, 1, 2], complex), np.ones(3, complex),
                                     **{**self.qc, "pg_amp_std_fraction": 100., "max_pg_amp_error": .1})
        self.assertEqual(quality["relative_valid"], [True]*3)
        self.assertEqual(quality["absolute_valid"], [True, True, False])

    def test_no_two_valid_nodes_fails_instead_of_inventing_an_unrequested_fallback(self):
        with self.assertRaisesRegex(ValueError, "Không đủ hai PG valid"):
            cal.validate_pg(np.ones(2, complex), np.ones(2, complex),
                            **{**self.qc, "max_pg_amp_error": 0., "invalid_pg_max_fraction": 1.})

    def test_interpolation_extrapolation_is_complex_linear_and_preserves_phase_wrap(self):
        np.testing.assert_allclose(cal.interpolate_pg([10, 20], [1+0j, 3j], [0, 15, 30]),
                                   [2-3j, .5+1.5j, -1+6j])
        values = np.exp(1j*np.array([np.pi-.1, -np.pi+.1]))
        np.testing.assert_allclose(cal.interpolate_pg([0, 1], values, [.5]), [-np.cos(.1)], atol=1e-15)

    def test_rejects_degenerate_interpolation_and_invalid_inputs(self):
        for times, values, queries in (([0, 0], [1+0j]*2, [0]), ([0], [1+0j], [0]),
                                        ([0, 1], [1+0j, -1+0j], [.5]),
                                        ([0, 1], [1+0j]*2, [np.nan])):
            with self.assertRaises(ValueError):
                cal.interpolate_pg(times, values, queries)
        with self.assertRaises(ValueError):
            cal.correlation_peak(np.zeros(4, complex), np.ones(4, complex), oversampling_factor=16)
        with self.assertRaises(ValueError):
            cal.extract_pulse(np.ones((2, 4)), order=[1, 2], method="PCC2")

    def test_pg_drift_removal_recovers_known_seven_hz_signal(self):
        times = np.arange(256)*.01
        gains = np.array([1., .8*np.exp(.4j)])
        replicas = gains[:, None]*self.signal
        corrections, _, _ = cal.pg_products(replicas, pg_reference=1., oversampling_factor=16)
        correction = cal.interpolate_pg([times[0], times[-1]], corrections/corrections[0], times)
        ideal = np.tile(np.exp(2j*np.pi*7*times)[:, None], (1, 4))
        raw = ideal/correction[:, None]
        corrected = raw*correction[:, None]
        np.testing.assert_allclose(corrected, ideal, atol=1e-15)
        np.testing.assert_allclose(-np.angle(lag1_accc(corrected))/(2*np.pi*.01), 7., atol=1e-12)

    def test_configurable_values_are_validated_before_numerical_operations(self):
        for floor in (None, np.nan, -.1, 1., [0.], 1j):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                cal.reconstruct_replica(*[self.signal]*5, denominator_relative_floor=floor)
        for override in ({"max_pg_amp_error": None}, {"pg_phase_std_fraction": [1.]},
                         {"invalid_pg_max_fraction": 1.1}, {"use_model": "true"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                cal.validate_pg(np.ones(2, complex), np.ones(2, complex), **{**self.qc, **override})
        for ref in (None, 0., [1.]):
            with self.subTest(reference=ref), self.assertRaises(ValueError):
                cal.pg_products(np.array([self.signal]*2), pg_reference=ref, oversampling_factor=16)


PRODUCT = Path.home()/"sar_data/focusing_data/product_1"


@unittest.skipUnless(
    any(PRODUCT.glob("level-0/s1a-s4-raw-s-vv-*.dat")) and
    all(any(PRODUCT.glob(f"auxiliary/*/data/s1a-aux-{kind}.xml")) for kind in ("ins", "pp1")),
    "Cần raw S4/VV và AUX ngoài repo.")
class RealCalibrationTest(unittest.TestCase):
    def test_inventory_aux_orders_and_relative_pg_match_independent_numpy_oracle(self):
        raw = next(PRODUCT.glob("level-0/s1a-s4-raw-s-vv-*.dat"))
        level0 = sentinel1decoder.Level0File(str(raw))
        times, correction, report = analyze_calibration(level0, PRODUCT, oversampling_factor=16,
                                                       denominator_relative_floor=0.)
        self.assertEqual(len(report["inventory"]), 28)
        self.assertEqual(sum(r.get("kind") == "pg" for r in report["inventory"]), 12)
        self.assertEqual([n["chunks"] for n in report["nodes"]], [[7, 8, 9, 10, 11, 12], [15, 16, 17, 18, 19, 20]])
        decoder = sentinel1decoder.Level0Decoder(str(raw))
        replicas = []
        for chunks in ((7, 9, 10, 11, 12), (15, 17, 18, 19, 20)):
            pulses = []
            for index, chunk in enumerate(chunks):
                samples = decoder.decode_packets(level0.get_acquisition_chunk_metadata(chunk)).astype(complex)
                if index == 4:
                    pulses.append(samples[12:20].mean(0))
                else:
                    count = 12 if index == 0 else 20
                    pulses.append((samples[count-1]-samples[count-2])/2)
            tx, rx, epdn, ta, apdn = map(np.fft.fft, pulses)
            replicas.append(np.fft.ifft(tx*rx*ta/(epdn*apdn)))
        spectrum = np.fft.fft(replicas[1])*np.fft.fft(replicas[0]).conj()
        frequencies = np.fft.fftfreq(len(spectrum))
        nyquist = spectrum[len(spectrum)//2]
        spectrum[len(spectrum)//2] = 0
        # Independent dense local Fourier evaluation includes the fractional peak.
        candidates = np.linspace(-.001, .001, 1001)
        peaks = np.array([(np.sum(spectrum*np.exp(2j*np.pi*frequencies*lag)) +
                           nyquist*np.cos(np.pi*lag))/len(spectrum) for lag in candidates])
        peak = peaks[np.argmax(abs(peaks))]
        oracle = np.vdot(replicas[0], replicas[0])/peak
        np.testing.assert_allclose(correction, [1., oracle], rtol=1e-6)
        self.assertGreater(np.angle(correction[-1]), 0)
        self.assertEqual(report["quality"]["valid"], [True, True])
        self.assertFalse(report["quality"]["absolute_scale_verified"])
        self.assertTrue(report["echo_coverage_without_extrapolation"])
        for node in report["nodes"]:
            self.assertTrue(4e-7 < node["internal_delay_diagnostic_s"] < 5e-7)
        self.assertLess(times[0], times[1])


if __name__ == "__main__":
    unittest.main()
