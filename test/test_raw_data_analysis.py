"""Đối chứng DAD §4.1 với dữ liệu biết trước và bộ chọn echo L0 giả lập."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import sentinel1decoder
from sentinel1decoder.enums import RangeDecimation, RxChannelId, SignalType

from sentinel1_processing.common.raw_data_analysis import analyze_batches, correction_parameters

spec = importlib.util.spec_from_file_location(
    "sentinel1_rda_demo", Path(__file__).parents[1]/"workflows/sentinel-1/estimate_doppler_centroid.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def signal(correlations, *, i_bias=.25, q_bias=-.125, sigma_i=2., sigma_q=4.):
    x = np.tile([1., 1., -1., -1.], 16)
    y = np.tile([1., -1., 1., -1.], 16)
    rows = []
    for c in correlations:
        rows.append(i_bias+sigma_i*x + 1j*(q_bias+sigma_q*(c*x+np.sqrt(1-c*c)*y)))
    return np.array(rows)


class RawDataAnalysisTest(unittest.TestCase):
    def test_exact_population_statistics_and_three_sigma_bounds(self):
        result = analyze_batches([signal([.2]*8)])
        self.assertEqual(result["sample_count"], 512)
        for key, expected in (("mean_i", .25), ("mean_q", -.125), ("std_i", 2.),
                              ("std_q", 4.), ("iq_gain_imbalance", .5),
                              ("iq_quadrature_departure_rad", np.arcsin(.2))):
            self.assertAlmostEqual(result[key], expected, places=13)
        np.testing.assert_allclose(result["bounds"]["i_bias"], [-6/np.sqrt(512), 6/np.sqrt(512)])
        np.testing.assert_allclose(result["bounds"]["iq_gain_imbalance"], [1-3/np.sqrt(512), 1+3/np.sqrt(512)])
        self.assertFalse(result["outside_null_bounds"]["i_bias"])
        self.assertTrue(result["outside_null_bounds"]["iq_gain_imbalance"])

    def test_fisher_mean_is_not_pooled_or_arithmetic_correlation(self):
        correlations = np.array([-.4, .1, .2, .8])
        result = analyze_batches([signal(correlations)])
        z = np.arctanh(correlations)
        expected = np.arcsin(np.tanh(z.mean()))
        self.assertAlmostEqual(result["iq_quadrature_departure_rad"], expected, places=13)
        self.assertGreater(abs(expected-np.arcsin(correlations.mean())), .01)
        np.testing.assert_allclose(result["bounds"]["iq_quadrature_departure_rad"],
                                   np.arcsin(np.tanh([z.mean()-z.std(), z.mean()+z.std()])), atol=1e-13)

    def test_line_centering_removes_line_offsets_from_quadrature(self):
        data = signal([-.3, .1, .5, .7])
        offsets = np.arange(4)[:, None]*(1e6+2e6j)
        original = analyze_batches([data])
        shifted = analyze_batches([data+offsets])
        self.assertAlmostEqual(shifted["iq_quadrature_departure_rad"], original["iq_quadrature_departure_rad"], places=9)

    def test_batch_partition_does_not_change_statistics_or_flags(self):
        data = signal(np.linspace(-.5, .7, 41)) + 1e5 - 2e5j
        full = analyze_batches([data])
        streamed = analyze_batches(data[i:i+3] for i in range(0, len(data), 3))
        for key in ("mean_i", "mean_q", "std_i", "std_q", "iq_gain_imbalance", "iq_quadrature_departure_rad"):
            self.assertAlmostEqual(full[key], streamed[key], places=9)
        self.assertEqual(full["outside_null_bounds"], streamed["outside_null_bounds"])

    def test_ideal_iq_has_no_significant_departures(self):
        result = analyze_batches([signal([0.]*8, i_bias=0., q_bias=0., sigma_i=1., sigma_q=1.)])
        self.assertEqual(set(result["outside_null_bounds"].values()), {False})
        self.assertEqual(result["bounds"]["iq_quadrature_departure_rad"], [0., 0.])

    def test_negative_quadrature_uses_a_negative_null_lower_bound(self):
        result = analyze_batches([signal([-.5, -.4, -.3, -.2])])
        self.assertLess(result["bounds"]["null_quadrature_correlation"][0], 0.)
        self.assertTrue(result["outside_null_bounds"]["iq_quadrature_departure"])

    def test_degenerate_rows_are_reported_without_nan_or_infinite_fisher_z(self):
        constant = np.full((2, 64), 2+3j)
        perfect = signal([1., -1.])
        result = analyze_batches([constant, perfect, signal([.1, .2])])
        self.assertEqual(result["quadrature"]["excluded_constant_lines"], 2)
        self.assertEqual(result["quadrature"]["excluded_perfect_correlation_lines"], 2)
        self.assertEqual(result["quadrature"]["valid_lines"], 2)
        invalid = analyze_batches([constant])
        self.assertIsNone(invalid["iq_gain_imbalance"])
        self.assertIsNone(invalid["iq_quadrature_departure_rad"])
        self.assertIsNone(invalid["outside_null_bounds"]["iq_quadrature_departure"])

    def test_rejects_empty_nonfinite_noncomplex_and_mixed_width(self):
        for batches in ([], [np.ones((2, 8))], [np.ones((0, 8), complex)],
                        [np.array([[1+1j, np.nan+1j]])], [signal([.1, .2]), np.ones((1, 8), complex)]):
            with self.subTest(batches=str(batches)[:40]), self.assertRaises(ValueError):
                analyze_batches(batches)

    def test_flags_and_predefined_values_are_explicit_api_inputs(self):
        measured = analyze_batches([signal([.2]*8)])
        disabled = correction_parameters(measured, correct_iq_bias=False, correct_iq_gain=False,
                                         correct_iq_quadrature=False)
        self.assertEqual(disabled["applied_parameters"],
                         {"i_bias": 0., "q_bias": 0., "gain_imbalance": 1., "quadrature_rad": 0.})
        enabled = correction_parameters(measured, correct_iq_bias=True, correct_iq_gain=True,
                                        correct_iq_quadrature=True, predefined_q_bias=-.5)
        self.assertEqual(enabled["applied_parameters"]["i_bias"], .25)
        self.assertEqual(enabled["applied_parameters"]["q_bias"], -.5)
        self.assertEqual(enabled["sources"]["q_bias"], "predefined")
        self.assertAlmostEqual(enabled["applied_parameters"]["gain_imbalance"], .5)
        np.testing.assert_allclose(np.mean(signal([.2]*8)-(.25-.125j)), 0., atol=1e-14)

    def test_undefined_measurement_requires_predefined_value_only_if_enabled(self):
        measured = analyze_batches([np.full((2, 8), 2+3j)])
        with self.assertRaisesRegex(ValueError, "predefined"):
            correction_parameters(measured, correct_iq_bias=True, correct_iq_gain=True, correct_iq_quadrature=True)
        used = correction_parameters(measured, correct_iq_bias=True, correct_iq_gain=True,
                                     correct_iq_quadrature=True, predefined_gain_imbalance=1., predefined_quadrature_rad=0.)
        self.assertEqual(used["applied_parameters"]["gain_imbalance"], 1.)

    def test_rejects_invalid_configurable_flags_and_predefined_statistics(self):
        measured = analyze_batches([signal([.1, .2])])
        config = dict(correct_iq_bias=True, correct_iq_gain=False, correct_iq_quadrature=False)
        for change in ({"correct_iq_bias": "true"}, {"predefined_i_bias": float("nan")},
                       {"predefined_i_bias": [0.]}, {"predefined_gain_imbalance": 0.},
                       {"predefined_quadrature_rad": np.pi/2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                correction_parameters(measured, **{**config, **change})


class RawDataSelectionTest(unittest.TestCase):
    def setUp(self):
        self.raw = signal([.1, .2, .3, .4, .5, .6, .7, .8])[:, :8]
        fs = sentinel1decoder.utilities.range_dec_to_sample_rate(RangeDecimation.RGDEC_5)
        self.frames = {}
        for chunk, indices in ((13, range(4)), (14, range(4, 8))):
            indices = np.array(list(indices))
            self.frames[chunk] = pd.DataFrame({
                "Signal Type": SignalType.ECHO, "Error Flag": False,
                "Swath Number": 3, "Rx Channel ID": RxChannelId.RXV_POL_CHANNEL,
                "Rx Gain": -6., "Range Decimation": RangeDecimation.RGDEC_5,
                "Number of Quads": 4, "Instrument Configuration ID": 7,
                "PRI": .01, "Tx Pulse Length": 2/fs, "Tx Ramp Rate": 0.,
                "Tx Pulse Start Frequency": 0., "Rank": 9, "SWST": 0., "SWL": 8/fs,
                "Coarse Time": 100., "Fine Time": indices*.01,
            }, index=pd.Index(indices, name="Packet Number"))
        self.frames[0] = pd.DataFrame({"Signal Type": ["calibration"]})
        self.level0 = SimpleNamespace(filename="synthetic.dat", acquisition_chunks=[0, 13, 14],
                                      get_acquisition_chunk_metadata=lambda c: self.frames[c])
        self.decoder = Mock()
        self.decoder.decode_packets.side_effect = lambda frame: self.raw[frame.index.to_numpy()].copy()
        self.config = dict(batch_lines=2, **demo.RAW_ANALYSIS_SELECTION)

    def test_selection_excludes_error_calibration_and_crosses_chunk_boundaries(self):
        self.frames[13].loc[1, "Error Flag"] = True
        self.raw[1] = 1e9+2e9j
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder):
            actual = demo.analyze_echoes(self.level0, **{**self.config, "azimuth_start_line": 1,
                                                        "azimuth_step": 2, "range_start_sample": 1,
                                                        "range_stop_sample": 7})
        expected = analyze_batches([self.raw[[2, 4, 6], 1:7]])
        self.assertEqual(actual["line_count"], 3)
        self.assertEqual(actual["excluded_error_flag_lines"], 1)
        self.assertAlmostEqual(actual["mean_i"], expected["mean_i"])
        self.assertAlmostEqual(actual["iq_quadrature_departure_rad"], expected["iq_quadrature_departure_rad"])
        self.assertEqual([r["chunk"] for r in actual["sources"]], [13, 14])

    def test_analysis_scope_is_independent_of_batch_size(self):
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder):
            a = demo.analyze_echoes(self.level0, **self.config)
            b = demo.analyze_echoes(self.level0, **{**self.config, "batch_lines": 3})
        for key in ("mean_i", "mean_q", "std_i", "std_q", "iq_quadrature_departure_rad"):
            self.assertAlmostEqual(a[key], b[key], places=13)

    def test_configurable_line_limit_really_limits_the_analysis_scope(self):
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder):
            result = demo.analyze_echoes(self.level0, **{**self.config, "azimuth_start_line": 2,
                                                       "azimuth_step": 2, "max_azimuth_lines": 2})
        expected = analyze_batches([self.raw[[2, 4]]])
        self.assertEqual(result["line_count"], 2)
        self.assertAlmostEqual(result["iq_quadrature_departure_rad"], expected["iq_quadrature_departure_rad"])

    def test_rejects_invalid_selection_and_mixed_receiver_configuration(self):
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder):
            for change in ({"azimuth_step": 0}, {"max_azimuth_lines": 0},
                           {"range_stop_sample": 99}, {"azimuth_start_line": 99}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    demo.analyze_echoes(self.level0, **{**self.config, **change})
            self.frames[14]["Rx Gain"] = -4.
            with self.assertRaisesRegex(ValueError, "scope riêng"):
                demo.analyze_echoes(self.level0, **self.config)

    def test_dce_uses_one_l0_bias_across_chunks_and_obeys_aux_flag(self):
        aux = dict(correct_iq_bias=True, correct_iq_gain=False, correct_iq_quadrature=False,
                   correct_rx_variation=False, swst_bias_s=0.)
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder), TemporaryDirectory() as tmp:
            analysis = demo.analyze_echoes(self.level0, **self.config)
            _, _, _, diagnostics = demo.compress_echoes(self.level0, aux, tmp, raw_analysis=analysis, batch_lines=2)
            self.assertEqual([d["iq_bias"] for d in diagnostics], [[.25, -.125], [.25, -.125]])
            self.assertEqual([d["iq_analysis_lines"] for d in diagnostics], [8, 8])
            aux["correct_iq_bias"] = False
            _, _, _, disabled = demo.compress_echoes(self.level0, aux, tmp, raw_analysis=analysis, batch_lines=2)
            self.assertEqual([d["iq_bias"] for d in disabled], [[0., 0.], [0., 0.]])

    def test_dce_applies_predefined_iq_before_receiver_pg_and_range_compression(self):
        aux = dict(correct_iq_bias=False, correct_iq_gain=False, correct_iq_quadrature=False,
                   correct_rx_variation=True, swst_bias_s=0.,
                   rx_gain_trend=[10*np.log10(4), 0, 0, 0, 0, 1, 0, 0], rx_gain_overshoot=[0]*5)
        predefined = dict(predefined_i_bias=.5, predefined_q_bias=-.25,
                          predefined_gain_imbalance=1.5, predefined_quadrature_rad=.2)
        flags = dict(correct_iq_bias=True, correct_iq_gain=True, correct_iq_quadrature=True)
        reference = SimpleNamespace(
            bias=100+200j, internal_delay_s=0.,
            range_reference=lambda rate, pulse, length: np.ones(length, np.complex64),
            pg_gain=lambda eta: np.full(len(eta), 3.),
            range_gain_bank=lambda tau: None,
            apply_range_gain=lambda *args: None,
        )
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder), TemporaryDirectory() as tmp:
            analysis = demo.analyze_echoes(self.level0, **self.config)
            with patch.object(demo.compression, "compress", wraps=demo.compression.compress) as compress:
                _, _, _, diagnostics = demo.compress_echoes(
                    self.level0, aux, tmp, reference, raw_analysis=analysis,
                    batch_lines=3, **flags, **predefined)
            used_rows = np.concatenate([call.args[0] for call in compress.call_args_list])
            i, q = self.raw.real-.5, (self.raw.imag+.25)*1.5
            expected = (i + 1j*(q/np.cos(.2)-i*np.tan(.2))) / 2 * 3
            np.testing.assert_allclose(used_rows, expected, atol=2e-14)
            self.assertTrue(all(call.kwargs["iq_bias"] == 0j for call in compress.call_args_list))
            for diagnostic in diagnostics:
                selected = diagnostic["raw_correction_parameters"]
                self.assertEqual(set(selected["sources"].values()), {"predefined"})
                self.assertTrue(all(selected["enabled"].values()))

    def test_post_correction_analysis_uses_same_scope_and_matches_independent_formula(self):
        aux_flags = dict(correct_iq_bias=True, correct_iq_gain=True, correct_iq_quadrature=True)
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder):
            before = demo.analyze_echoes(self.level0, **self.config)
            correction = correction_parameters(before, **aux_flags)
            after = demo.analyze_echoes(self.level0, **self.config, iq_correction=correction)
        i = self.raw.real - before["mean_i"]
        theta = before["iq_quadrature_departure_rad"]
        q = ((self.raw.imag-before["mean_q"])*before["iq_gain_imbalance"] - i*np.sin(theta))/np.cos(theta)
        expected = analyze_batches([i+1j*q])
        for key in ("mean_i", "mean_q", "std_i", "std_q", "iq_gain_imbalance", "iq_quadrature_departure_rad"):
            self.assertAlmostEqual(after[key], expected[key], places=13)
        self.assertEqual(after["selection"], before["selection"])
        self.assertEqual(after["sources"], before["sources"])

    def test_reference_mode_keeps_l1_bias_when_optional_gain_needs_l0_statistics(self):
        aux = dict(correct_iq_bias=True, correct_iq_gain=False, correct_iq_quadrature=False,
                   correct_rx_variation=False, swst_bias_s=0.)
        reference = SimpleNamespace(
            bias=.8-.6j, internal_delay_s=0.,
            range_reference=lambda rate, pulse, length: np.ones(length, np.complex64),
            pg_gain=lambda eta: np.ones(len(eta)), range_gain_bank=lambda tau: None,
            apply_range_gain=lambda *args: None,
        )
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder), TemporaryDirectory() as tmp:
            _, _, _, diagnostics = demo.compress_echoes(
                self.level0, aux, tmp, reference, batch_lines=2, correct_iq_gain=True)
        selected = diagnostics[0]["raw_correction_parameters"]
        self.assertEqual(diagnostics[0]["iq_bias"], [.8, -.6])
        self.assertEqual(diagnostics[0]["iq_analysis_lines"], 8)
        self.assertEqual(selected["sources"]["gain_imbalance"], "measured")
        self.assertEqual(selected["sources"]["i_bias"], "predefined")
        self.assertEqual(diagnostics[0]["iq_bias_source"], "L1 support/*BiasUsedForCorrection")

    def test_receiver_compensation_uses_each_chunk_swst_and_obeys_aux_override_and_batch_size(self):
        aux = dict(correct_iq_bias=True, correct_iq_gain=False, correct_iq_quadrature=False,
                   correct_rx_variation=True, swst_bias_s=0.,
                   rx_gain_trend=[0, .1, 0, 0, 0, 1, 0, 0], rx_gain_overshoot=[0]*5)
        fs = sentinel1decoder.utilities.range_dec_to_sample_rate(RangeDecimation.RGDEC_5)
        # Integer sample offset isolates receiver timing from fractional range alignment.
        self.frames[14]["SWST"] = 500/fs
        pulse = float(self.frames[13]["Tx Pulse Length"].iloc[0])
        n_tx = int(np.ceil(pulse*fs))
        before = self.raw.copy()
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder), TemporaryDirectory() as tmp:
            analysis = demo.analyze_echoes(self.level0, **self.config)
            baseline = None
            for aux_enabled, override, batch_size in ((True, None, 2), (True, None, 3),
                                                       (True, False, 2), (False, True, 2)):
                aux["correct_rx_variation"] = aux_enabled
                segments, _, _, diagnostics = demo.compress_echoes(
                    self.level0, aux, tmp, raw_analysis=analysis, batch_lines=batch_size,
                    correct_rx_variation=override)
                enabled = aux_enabled if override is None else override
                corrected = []
                for chunk, rows in ((13, slice(0, 4)), (14, slice(4, 8))):
                    swst = float(self.frames[chunk]["SWST"].iloc[0])
                    t_us = (np.arange(8)/fs + swst + 320/(8*sentinel1decoder.constants.F_REF) - pulse)*1e6
                    gain = 10**(.1*t_us/10) if enabled else np.ones(8)
                    corrected.extend((self.raw[rows]-(.25-.125j))/np.sqrt(gain))
                expected = np.array([np.convolve(row, np.ones(n_tx)/np.sqrt(n_tx))[n_tx-1:7]
                                     for row in corrected])
                actual = np.concatenate([s.range_compressed for s in segments])
                np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
                self.assertEqual([d["receiver_compensation"]["enabled"] for d in diagnostics], [enabled]*2)
                self.assertEqual(diagnostics[0]["receiver_compensation"]["flag_source"],
                                 "explicit override" if override is not None else "AUX_PP1 commonProcParams/correctRxVariationFlag")
                if enabled:
                    if baseline is None:
                        baseline = actual.copy()
                    else:
                        np.testing.assert_array_equal(actual, baseline)
                    self.assertNotEqual(diagnostics[0]["receiver_compensation"]["power_gain_first"],
                                        diagnostics[1]["receiver_compensation"]["power_gain_first"])
        np.testing.assert_array_equal(self.raw, before)

    def test_auxiliary_receiver_flag_is_required_and_selected_from_matching_product(self):
        instrument = ET.fromstring('''<auxiliaryInstrument><radarFrequency>5405000000</radarFrequency>
          <internalCalibrationParamsList><internalCalibrationParams>
            <swath>S4</swath><polarisation>VV</polarisation><swstBias>0</swstBias>
          </internalCalibrationParams></internalCalibrationParamsList>
          <swathParamsList><swathParams><swath>S4</swath>
            <rxVariationCorrectionParamsList><rxVariationCorrectionParams><rxPolarisation>V</rxPolarisation>
              <gainTrendCoefficients>0 0 0 0 0 1 0 0</gainTrendCoefficients>
              <gainOvershootCoefficients>0 0 0 0 0</gainOvershootCoefficients>
            </rxVariationCorrectionParams></rxVariationCorrectionParamsList>
            <pulseParamsList><pulseParams><polarisation>VV</polarisation>
              <amplitudeCoefficients>1 0 0 0</amplitudeCoefficients>
              <phaseCoefficients>0 0 0 0</phaseCoefficients>
            </pulseParams></pulseParamsList>
          </swathParams></swathParamsList></auxiliaryInstrument>''')
        processor = ET.fromstring('''<l1AuxiliaryProcessorParameters><productList>
          <product><productId>WRONG</productId></product><product><productId>SM_SL1__1</productId>
            <dcProcParams><dcMethod>Data Analysis</dcMethod><dcInputData>Range Compressed</dcInputData>
              <dcRmsErrorThreshold>20</dcRmsErrorThreshold></dcProcParams>
            <commonProcParams><correctIQBiasFlag>true</correctIQBiasFlag>
              <correctIQGainImbalanceFlag>false</correctIQGainImbalanceFlag>
              <correctIQOrthogonalityFlag>false</correctIQOrthogonalityFlag>
              <correctRxVariationFlag>true</correctRxVariationFlag>
            </commonProcParams>
          </product></productList></l1AuxiliaryProcessorParameters>''')
        product = Mock()
        product.glob.side_effect = lambda pattern: [Path("ins.xml" if "ins" in pattern else "pp1.xml")]
        with patch.object(demo.ET, "parse", side_effect=lambda path: ET.ElementTree(
                instrument if path.name == "ins.xml" else processor)):
            flag = processor.find("productList/product[productId='SM_SL1__1']/commonProcParams/correctRxVariationFlag")
            for value, expected in (("true", True), ("false", False)):
                flag.text = value
                self.assertIs(demo.read_auxiliary(product)["correct_rx_variation"], expected)
            for value in (None, "1", "bad"):
                flag.text = value
                with self.assertRaisesRegex(ValueError, "correctRxVariationFlag"):
                    demo.read_auxiliary(product)

    def test_receiver_overflow_is_rejected_before_range_fft(self):
        self.raw = self.raw.astype(np.complex64)
        aux = dict(correct_iq_bias=False, correct_iq_gain=False, correct_iq_quadrature=False,
                   correct_rx_variation=True, swst_bias_s=0.,
                   rx_gain_trend=[-3000, 0, 0, 0, 0, 1, 0, 0], rx_gain_overshoot=[0]*5)
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder), TemporaryDirectory() as tmp:
            analysis = demo.analyze_echoes(self.level0, **self.config)
            with patch.object(demo.compression, "compress", side_effect=AssertionError("Không được tới FFT")):
                with self.assertRaisesRegex(ValueError, "sau receiver/PG correction không hữu hạn"):
                    demo.compress_echoes(self.level0, aux, tmp, raw_analysis=analysis, batch_lines=2)

    def test_explicit_pg_provider_replaces_reference_pg_and_invalid_values_stop_before_fft(self):
        aux = dict(correct_iq_bias=True, correct_iq_gain=False, correct_iq_quadrature=False,
                   correct_rx_variation=False, swst_bias_s=0.)
        reference = SimpleNamespace(
            bias=.25-.125j, internal_delay_s=0.,
            range_reference=lambda rate, pulse, length: np.ones(length, np.complex64),
            pg_gain=Mock(side_effect=AssertionError("Không được dùng PG L1 khi caller cấp PG L0")),
            range_gain_bank=lambda tau: None, apply_range_gain=lambda *args: None)
        with patch.object(demo.sentinel1decoder, "Level0Decoder", return_value=self.decoder), TemporaryDirectory() as tmp:
            analysis = demo.analyze_echoes(self.level0, **self.config)
            with patch.object(demo.compression, "compress", wraps=demo.compression.compress) as compress:
                _, _, _, diagnostics = demo.compress_echoes(
                    self.level0, aux, tmp, reference, raw_analysis=analysis, batch_lines=3,
                    pg_correction_provider=lambda eta: np.full(len(eta), 3+0j))
            rows = np.concatenate([call.args[0] for call in compress.call_args_list])
            np.testing.assert_allclose(rows, 3*(self.raw-(.25-.125j)))
            self.assertEqual([d["pg_source"] for d in diagnostics], ["caller-supplied correction"]*2)
            for invalid in (lambda eta: np.zeros(len(eta), complex), lambda eta: np.full(len(eta), np.nan),
                            lambda eta: np.ones((len(eta), 1), complex)):
                with patch.object(demo.compression, "compress", side_effect=AssertionError("Không được tới FFT")):
                    with self.assertRaisesRegex(ValueError, "PG correction provider"):
                        demo.compress_echoes(self.level0, aux, tmp, raw_analysis=analysis,
                                             pg_correction_provider=invalid)


PRODUCT_RAW = Path.home()/"sar_data/focusing_data/product_1/level-0/s1a-s4-raw-s-vv-20250903t183628-20250903t183638-060827-0792bb.dat"


@unittest.skipUnless(PRODUCT_RAW.exists(), "Cần raw S4/VV product_1 ngoài repo để kiểm tra integration.")
class RealProductRawDataTest(unittest.TestCase):
    def test_selected_actual_echoes_match_independent_numpy_oracle(self):
        level0 = sentinel1decoder.Level0File(str(PRODUCT_RAW))
        frames = [level0.get_acquisition_chunk_metadata(c) for c in level0.acquisition_chunks]
        echo = pd.concat([m for m in frames if (m["Signal Type"] == SignalType.ECHO).all()])
        selected = echo.loc[~echo["Error Flag"]].iloc[::1024].iloc[:16]
        raw = sentinel1decoder.Level0Decoder(str(PRODUCT_RAW)).decode_packets(selected)
        measured = demo.analyze_echoes(level0, batch_lines=3, azimuth_start_line=0,
                                      azimuth_step=1024, max_azimuth_lines=16,
                                      range_start_sample=0, range_stop_sample=None)
        self.assertEqual(measured["line_count"], 16)
        self.assertEqual({s["chunk"] for s in measured["sources"]}, {13, 14})
        self.assertAlmostEqual(measured["mean_i"], np.mean(raw.real, dtype=float), places=12)
        self.assertAlmostEqual(measured["mean_q"], np.mean(raw.imag, dtype=float), places=12)
        self.assertAlmostEqual(measured["std_i"], np.std(raw.real, dtype=float), places=12)
        self.assertAlmostEqual(measured["std_q"], np.std(raw.imag, dtype=float), places=12)
        c = [np.corrcoef(row.real, row.imag)[0, 1] for row in raw]
        theta = np.arcsin(np.tanh(np.mean(np.arctanh(c))))
        self.assertAlmostEqual(measured["iq_quadrature_departure_rad"], theta, places=12)
        zero_mean = raw.astype(complex) - complex(measured["mean_i"], measured["mean_q"])
        self.assertLess(abs(np.mean(zero_mean)), 1e-12)


if __name__ == "__main__":
    unittest.main()
