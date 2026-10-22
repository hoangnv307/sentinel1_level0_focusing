"""Calibration/PG: mô hình thuận độc lập, validity và timing từ L0 thật."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import sentinel1decoder
from sentinel1decoder.enums import SignalType, RangeDecimation

from sentinel1_processing.pre_processing import internal_calibration as cal
from sentinel1_processing.utils import pg_calibration_input as adapter
from sentinel1_processing.utils.pg_calibration_input import analyze_calibration, read_configuration
from sentinel1_processing.doppler_centroid.estimation import lag1_accc


class InternalCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.signal = np.exp(2j*np.pi*np.arange(64)**2/200)
        self.qc = dict(pg_amp_std_fraction=1., pg_phase_std_fraction=100., max_pg_amp_error=100.,
                       max_pg_phase_error=6.28, invalid_pg_max_fraction=.5, use_model=False)

    def test_internal_delay_averages_valid_peaks_with_configurable_guard_and_suppression(self):
        actual = cal.mean_internal_delay([12.25, np.nan, 14.75], sample_rate_hz=100.,
                                         delta_t_guard1_s=.04, delta_t_suppr_s=.01,
                                         valid=[True, False, True])
        self.assertAlmostEqual(actual, .105)
        self.assertAlmostEqual(cal.mean_internal_delay([12.25], sample_rate_hz=100.,
            delta_t_guard1_s=.04, delta_t_suppr_s=.01, valid=[True]), .0925)

    def test_internal_delay_rejects_missing_peaks_invalid_masks_and_configuration(self):
        config = dict(sample_rate_hz=100., delta_t_guard1_s=.04, delta_t_suppr_s=.01, valid=[True, True])
        for override in ({"valid": [False, False]}, {"valid": [1, 1]}, {"valid": [True]},
                         {"sample_rate_hz": 0.}, {"delta_t_guard1_s": np.nan}, {"delta_t_suppr_s": -1.}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                cal.mean_internal_delay([12., 14.], **{**config, **override})
        with self.assertRaises(ValueError):
            cal.mean_internal_delay([np.nan, 14.], **config)

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
        products, peaks, lags = cal.pg_products(replicas, reference=self.signal, pg_reference=707.+2j, oversampling_factor=16)
        np.testing.assert_allclose(products/products[0], 1/gains, atol=1e-14)
        np.testing.assert_allclose(peaks, gains*np.vdot(self.signal, self.signal), rtol=1e-14)
        np.testing.assert_allclose(lags, 0., atol=1e-12)

    def test_constant_pg_reference_scale_cancels_in_relative_drift(self):
        replicas = np.array([self.signal, self.signal*(.9+.1j)])
        a, _, _ = cal.pg_products(replicas, reference=self.signal, pg_reference=1., oversampling_factor=16)
        b, _, _ = cal.pg_products(replicas, reference=self.signal, pg_reference=707.+3j, oversampling_factor=16)
        np.testing.assert_allclose(a/a[0], b/b[0], atol=1e-14)

    def test_nominal_reference_removes_compression_gain_for_different_pulse_lengths_and_padding(self):
        gain = 813*np.exp(.4j)
        for count in (31, 64, 129):
            time = np.arange(count)/1024.
            transmitted = np.exp(2j*np.pi*(-100*time + .5*500*time**2))
            nominal = cal.nominal_pg_replica(sample_rate_hz=1024., pulse_length_s=count/1024.,
                                              start_frequency_hz=-100., ramp_rate_hz_per_s=500.)
            np.testing.assert_allclose(nominal, transmitted/count)
            for window in (count+100, 2*(count+100)):
                raw = np.zeros(window, complex)
                raw[10:10+count] = gain*transmitted
                products, peaks, lags = cal.pg_products(raw[None, :], reference=nominal,
                                                        pg_reference=707., oversampling_factor=16)
                np.testing.assert_allclose(peaks, [gain], rtol=1e-13)
                np.testing.assert_allclose(products, [707/gain], rtol=1e-13)
                np.testing.assert_allclose(lags, [10.], atol=1e-12)

    def test_measured_reference_has_different_units_and_depends_on_its_gain(self):
        count = len(self.signal)
        raw = np.array([2*self.signal, 3*self.signal])
        nominal, _, _ = cal.pg_products(raw, reference=self.signal/count, pg_reference=1., oversampling_factor=16)
        measured, _, _ = cal.pg_products(raw, reference=raw[0], pg_reference=1., oversampling_factor=16)
        np.testing.assert_allclose(nominal, [1/2, 1/3])
        np.testing.assert_allclose(measured, nominal/(2*count))

    def test_fft_zoom_preserves_peak_amplitude_instead_of_scaling_by_oversampling(self):
        for factor in (8, 16, 32):
            peak, lag = cal.correlation_peak(3j*self.signal, self.signal/len(self.signal),
                                             oversampling_factor=factor)
            np.testing.assert_allclose(peak, 3j, atol=1e-14)
            self.assertAlmostEqual(lag, 0., places=12)

    def test_nominal_replica_rejects_invalid_header_parameters(self):
        valid = dict(sample_rate_hz=10., pulse_length_s=.3, start_frequency_hz=-2., ramp_rate_hz_per_s=1.)
        for field, value in (("sample_rate_hz", 0.), ("pulse_length_s", -1.),
                              ("start_frequency_hz", np.nan), ("ramp_rate_hz_per_s", [1.])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                cal.nominal_pg_replica(**{**valid, field: value})

    def test_pg_reference_must_be_explicit_valid_and_fit_the_calibration_window(self):
        for reference in (np.zeros(64, complex), np.ones(65, complex), np.ones((1, 64), complex)):
            with self.subTest(shape=reference.shape), self.assertRaises(ValueError):
                cal.pg_products(self.signal[None, :], reference=reference,
                                pg_reference=1., oversampling_factor=16)

    def test_pcc_reconstruction_nominal_pg_recovers_known_chain_gain(self):
        gain = 800*np.exp(.7j)
        delta = np.eye(1, 64, dtype=complex)[0]
        paths = [gain*self.signal, 2*delta, 4*delta, 3*delta, 5*delta]
        pulses = [cal.extract_pulse(np.array([path+1j, -path+1j]), order=[1, 2], method="PCC2")
                  for path in paths[:4]]
        pulses.append(cal.extract_pulse(np.array([paths[4]]*3), order=[1, 2, 3], method="Average"))
        replica = cal.reconstruct_replica(*pulses, denominator_relative_floor=0.)
        correction, _, _ = cal.pg_products(replica[None, :], reference=self.signal/64,
                                           pg_reference=707., oversampling_factor=16)
        np.testing.assert_allclose(correction, [707/(.3*gain)], rtol=1e-14)

    def test_reader_rejects_old_schema_with_ambiguous_threshold_units(self):
        with TemporaryDirectory() as folder:
            data = Path(folder)/"auxiliary/fixture/data"
            data.mkdir(parents=True)
            (data/"s1a-aux-ins.xml").write_text('<auxiliaryInstrument schemaVersion="3.16"/>')
            (data/"s1a-aux-pp1.xml").write_text('<l1AuxiliaryProcessorParameters schemaVersion="3.14"/>')
            with self.assertRaisesRegex(ValueError, "AUX_PP1 3.15/3.16"):
                read_configuration(folder)

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

    def test_pg_qc_reports_shortage_and_interpolation_enforces_two_nodes_without_extra_fallback(self):
        quality, chosen = cal.validate_pg(np.ones(2, complex), np.ones(2, complex),
                                          **{**self.qc, "max_pg_amp_error": 0., "invalid_pg_max_fraction": 1.})
        self.assertFalse(quality["interpolation_available"])
        self.assertEqual(quality["source"], "extracted")
        self.assertEqual(quality["selected"], [False, False])
        self.assertEqual(quality["rejection_reasons"], [["absolute_amplitude"]]*2)
        with self.assertRaises(ValueError):
            cal.interpolate_pg(np.array([0., 1.])[quality["selected"]], chosen[quality["selected"]], [0.5])

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
        corrections, _, _ = cal.pg_products(replicas, reference=self.signal, pg_reference=1., oversampling_factor=16)
        correction = cal.interpolate_pg([times[0], times[-1]], corrections/corrections[0], times)
        ideal = np.tile(np.exp(2j*np.pi*7*times)[:, None], (1, 4))
        weight = (times-times[0])/(times[-1]-times[0])
        instrument_gain = 1/((1-weight)/gains[0]+weight/gains[1])
        raw = ideal*instrument_gain[:, None]
        corrected = raw*correction[:, None]
        np.testing.assert_allclose(corrected, ideal, atol=1e-15)
        np.testing.assert_allclose(-np.angle(lag1_accc(corrected))/(2*np.pi*.01), 7., atol=1e-12)

    def test_configurable_values_are_validated_before_numerical_operations(self):
        for floor in (None, np.nan, -.1, 1., [0.], 1j):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                cal.reconstruct_replica(*[self.signal]*5, denominator_relative_floor=floor)
        for override in ({"max_pg_amp_error": None}, {"pg_phase_std_fraction": [1.]},
                         {"invalid_pg_max_fraction": 1.1}, {"use_model": "true"},
                         {"measurement_valid": [1, 1]}, {"measurement_valid": [True]}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                cal.validate_pg(np.ones(2, complex), np.ones(2, complex), **{**self.qc, **override})
        for ref in (None, 0., [1.]):
            with self.subTest(reference=ref), self.assertRaises(ValueError):
                cal.pg_products(np.array([self.signal]*2), reference=self.signal, pg_reference=ref, oversampling_factor=16)


class CalibrationAvailabilityTest(unittest.TestCase):
    """Synthetic PCC sequences with known transfer functions; no external data."""

    def setUp(self):
        self.rate = sentinel1decoder.utilities.range_dec_to_sample_rate(RangeDecimation(0))
        orders = {name: dict(order=[1, 2], method="Average" if name == "ApdnCal" else "PCC2")
                  for name in ("TxCal", "RxCal", "EpdnCal", "TaCal", "ApdnCal")}
        self.configuration = dict(orders={"pg": orders, "replica": orders},
            perform_internal_calibration=True, pg_source="Extracted", pg_reference=[1., 0.],
            constant_pg_model=[1., 0.], delta_t_guard1_s=0., delta_t_suppr_s=0.,
            thresholds=dict(pg_amp_std_fraction=1e6, pg_phase_std_fraction=1e6, max_pg_amp_error=10.,
                            max_pg_phase_error=6.28, invalid_pg_max_fraction=1.))

    def analyze(self, *, gains=(1., 1.), lags=(2, 4), singular=(), **overrides):
        frames, samples = {}, {}
        delta = np.eye(1, 16, dtype=complex)[0]
        for sequence, (gain, lag) in enumerate(zip(gains, lags)):
            replica = np.roll(np.r_[np.ones(8), np.zeros(8)], lag).astype(complex)*gain
            for offset, code in enumerate((8, 9, 10, 11, 12)):
                chunk = 5*sequence+offset
                frame = pd.DataFrame({
                    "Signal Type": [SignalType(code)]*2, "Coarse Time": [100.+sequence]*2,
                    "Fine Time": [offset*.01, offset*.01+.001], "Error Flag": [False]*2,
                    "BAQ Mode": [RangeDecimation(0)]*2, "Range Decimation": [RangeDecimation(0)]*2,
                    "Swath Number": [89]*2, "Cal Type": [0]*2, "Calibration Beam Address": [0]*2,
                    "Rx Gain": [0.]*2, "Rx Channel ID": [RangeDecimation(0)]*2,
                    "Tx Pulse Number": [1, 2], "Tx Pulse Length": [8/self.rate]*2,
                    "Tx Pulse Start Frequency": [0.]*2, "Tx Ramp Rate": [0.]*2,
                    "Number of Quads": [8]*2, "Space Packet Count": [2*chunk, 2*chunk+1],
                }, index=[2*chunk, 2*chunk+1])
                frames[chunk] = frame
                pulse = replica if code == 8 else np.zeros(16, complex) if code == 10 and sequence in singular else delta
                samples[2*chunk] = np.array([pulse, pulse if code == 12 else -pulse])
        level0 = SimpleNamespace(filename="s1a-s4-raw-s-vv-synthetic.dat", acquisition_chunks=list(frames),
                                 get_acquisition_chunk_metadata=frames.__getitem__)
        decoder = Mock()
        decoder.decode_packets.side_effect = lambda metadata: samples[metadata.index[0]]
        with patch.object(adapter, "read_configuration", return_value=self.configuration), \
                patch.object(adapter.sentinel1decoder, "Level0Decoder", return_value=decoder):
            result = adapter.analyze_calibration(level0, Path("unused"),
                **{**dict(oversampling_factor=16, denominator_relative_floor=0.), **overrides})
        json.dumps(result[2], allow_nan=False)
        return result

    def test_one_sequence_measures_delay_but_cannot_interpolate_pg(self):
        times, values, report = self.analyze(gains=(1.,), lags=(2,))
        self.assertTrue(report["internal_delay"]["available"])
        self.assertAlmostEqual(report["internal_delay"]["mean_s"]*self.rate, 2., places=9)
        self.assertFalse(report["quality"]["interpolation_available"])
        with self.assertRaises(ValueError):
            cal.interpolate_pg(times, values, [100.])

    def test_invalid_first_peak_is_reported_and_second_peak_alone_measures_delay(self):
        _, _, report = self.analyze(gains=(0., 1.))
        self.assertEqual(report["internal_delay"]["valid_sequence_indices"], [1])
        self.assertAlmostEqual(report["internal_delay"]["mean_s"]*self.rate, 4., places=9)
        self.assertIn("correlation peak", report["nodes"][0]["replica_rejection_reason"])
        self.assertIsNone(report["nodes"][0]["internal_delay_diagnostic_s"])
        self.assertEqual(report["nodes"][0]["pg_rejection_reasons"], ["measurement_invalid"])

    def test_all_pg_qc_failures_do_not_change_delay_or_invent_model_fallback(self):
        self.configuration["thresholds"]["max_pg_amp_error"] = 0.
        times, values, report = self.analyze()
        self.assertAlmostEqual(report["internal_delay"]["mean_s"]*self.rate, 3., places=9)
        self.assertEqual(report["quality"]["source"], "extracted")
        self.assertFalse(report["quality"]["interpolation_available"])
        self.assertEqual(report["quality"]["invalid_fraction"], 1.)
        self.assertEqual(report["quality"]["rejection_reasons"], [["absolute_amplitude"]]*2)
        self.assertEqual((len(times), len(values)), (0, 0))

    def test_pg_qc_rejects_one_node_while_delay_still_averages_both_peaks(self):
        self.configuration["thresholds"]["max_pg_amp_error"] = .1
        _, _, report = self.analyze(gains=(1., 2.))
        self.assertEqual(report["quality"]["selected"], [True, False])
        self.assertEqual(report["internal_delay"]["valid_sequence_indices"], [0, 1])
        self.assertAlmostEqual(report["internal_delay"]["mean_s"]*self.rate, 3., places=9)

    def test_all_invalid_peaks_report_unavailable_delay_and_no_fabricated_pg(self):
        times, values, report = self.analyze(gains=(0., 0.))
        self.assertFalse(report["internal_delay"]["available"])
        self.assertIsNone(report["internal_delay"]["mean_s"])
        self.assertEqual(report["internal_delay"]["valid_sequence_indices"], [])
        self.assertEqual(report["quality"]["invalid_fraction"], 1.)
        self.assertFalse(report["quality"]["interpolation_available"])
        self.assertTrue(all(n["replica_rejection_reason"] for n in report["nodes"]))
        self.assertEqual((len(times), len(values)), (0, 0))

    def test_invalid_peak_fraction_counts_all_sequences_and_obeys_aux_model_policy(self):
        self.configuration["thresholds"]["invalid_pg_max_fraction"] = .25
        _, values, report = self.analyze(gains=(0., 1.))
        self.assertEqual(report["quality"]["invalid_fraction"], .5)
        self.assertEqual(report["quality"]["source"], "model")
        np.testing.assert_array_equal(values, [1., 1.])
        self.assertEqual(report["internal_delay"]["valid_sequence_indices"], [1])

    def test_model_can_supply_pg_nodes_but_never_manufactures_a_delay_peak(self):
        self.configuration["pg_source"] = "Model"
        _, values, report = self.analyze(gains=(0., 0.))
        self.assertTrue(report["quality"]["interpolation_available"])
        self.assertEqual(report["quality"]["source"], "model")
        np.testing.assert_array_equal(values, [1., 1.])
        self.assertFalse(report["internal_delay"]["available"])

    def test_singular_reconstruction_and_outside_support_have_sequence_rejection_reasons(self):
        _, _, report = self.analyze(singular=(0,), lags=(2, 15))
        self.assertIn("spectral support", report["nodes"][0]["replica_rejection_reason"])
        self.assertIn("ngoài cửa sổ", report["nodes"][1]["replica_rejection_reason"])
        self.assertFalse(report["internal_delay"]["available"])

    def test_invalid_algorithm_configuration_is_fatal_instead_of_bad_sequence(self):
        for override in ({"oversampling_factor": 1}, {"oversampling_factor": True},
                         {"denominator_relative_floor": -1.}, {"denominator_relative_floor": np.nan}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.analyze(**override)


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
        header = level0.get_acquisition_chunk_metadata(7).iloc[0]
        rate = 112604166.72
        count = int(np.ceil(float(header["Tx Pulse Length"])*rate))
        time = np.arange(count)/rate
        # Independent commanded chirp; no calibration or L1 amplitude is fitted.
        nominal = np.exp(2j*np.pi*(float(header["Tx Pulse Start Frequency"])*time +
                                   .5*float(header["Tx Ramp Rate"])*time**2))/count
        independent_peaks, independent_lags = [], []
        for replica in replicas:
            spectrum = np.fft.fft(replica)*np.fft.fft(nominal, n=len(replica)).conj()
            frequencies = np.fft.fftfreq(len(spectrum))
            nyquist = spectrum[len(spectrum)//2]
            spectrum[len(spectrum)//2] = 0
            # Independent dense local Fourier evaluation includes the fractional peak.
            candidates = np.linspace(48.78, 48.81, 3001)
            peaks = np.array([(np.sum(spectrum*np.exp(2j*np.pi*frequencies*lag)) +
                               nyquist*np.cos(np.pi*lag))/len(spectrum) for lag in candidates])
            independent_peaks.append(peaks[np.argmax(abs(peaks))])
            independent_lags.append(candidates[np.argmax(abs(peaks))])
        peak_first, peak = independent_peaks
        oracle = peak_first/peak
        np.testing.assert_allclose(correction, [1., oracle], rtol=2e-5)
        expected = complex(*report["configuration"]["pg_reference"])/peak
        np.testing.assert_allclose(complex(*report["nodes"][1]["normalized_pg_correction"]), expected, rtol=2e-5)
        absolute_times,absolute_corrections,absolute_report=analyze_calibration(level0,PRODUCT,
            oversampling_factor=16,denominator_relative_floor=0.,output_normalization="absolute")
        np.testing.assert_array_equal(absolute_times,times)
        np.testing.assert_allclose(absolute_corrections[-1],expected,rtol=2e-5)
        np.testing.assert_allclose(absolute_corrections/absolute_corrections[0],correction,rtol=1e-12)
        self.assertEqual(absolute_report["output_normalization"],"absolute")
        self.assertTrue(.85 < abs(expected) < .90)
        self.assertGreater(np.angle(correction[-1]), 0)
        self.assertEqual(report["quality"]["valid"], [True, True])
        self.assertTrue(report["quality"]["absolute_scale_verified"])
        self.assertFalse(report["absolute_radiometric_scale_recovered"])
        self.assertEqual(report["pg_reference_function"]["sample_count"], 5799)
        self.assertTrue(report["echo_coverage_without_extrapolation"])
        for node in report["nodes"]:
            self.assertTrue(4e-7 < node["internal_delay_diagnostic_s"] < 5e-7)
        self.assertAlmostEqual(report["internal_delay"]["mean_s"],
                               np.mean([n["internal_delay_diagnostic_s"] for n in report["nodes"]]), places=16)
        self.assertEqual(report["internal_delay"]["valid_sequence_indices"], [0, 1])
        independent_delay = np.mean(independent_lags)/rate - (
            report["configuration"]["delta_t_guard1_s"]-report["configuration"]["delta_t_suppr_s"])
        self.assertLess(abs(report["internal_delay"]["mean_s"]-independent_delay), 1e-12)
        self.assertLess(times[0], times[1])


if __name__ == "__main__":
    unittest.main()
