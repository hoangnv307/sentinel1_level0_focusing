"""Đối chứng công thức DAD §9.2.1 bằng tín hiệu có sai lệch biết trước."""

from itertools import product
import unittest

import numpy as np

from sentinel1_processing.common.raw_data_analysis import analyze_batches, correction_parameters
from sentinel1_processing.common.raw_data_correction import apply
from sentinel1_processing.doppler_centroid.estimation import lag1_accc


class RawDataCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.i = np.tile([1., 1., -1., -1.], (8, 16))
        self.q = np.tile([1., -1., 1., -1.], (8, 16))
        self.parameters = dict(i_bias=.25, q_bias=-.125, gain_imbalance=1.7, quadrature_rad=-.21)
        self.flags = dict(correct_iq_bias=True, correct_iq_gain=True, correct_iq_quadrature=True)
        self.raw = self.distort(self.i + 1j*self.q)

    def distort(self, ideal):
        p = self.parameters
        return (ideal.real + p["i_bias"] + 1j*(p["q_bias"] +
                (ideal.imag*np.cos(p["quadrature_rad"]) + ideal.real*np.sin(p["quadrature_rad"])) /
                p["gain_imbalance"]))

    def test_measured_parameters_recover_zero_mean_equal_gain_orthogonal_signal(self):
        measured = analyze_batches([self.raw[:3], self.raw[3:]])
        selected = correction_parameters(measured, **self.flags)
        corrected = apply(self.raw, **selected["applied_parameters"], **self.flags)
        np.testing.assert_allclose(corrected, self.i + 1j*self.q, atol=1e-14)
        after = analyze_batches([corrected])
        self.assertAlmostEqual(after["mean_i"], 0., places=14)
        self.assertAlmostEqual(after["mean_q"], 0., places=14)
        self.assertAlmostEqual(after["iq_gain_imbalance"], 1., places=14)
        self.assertAlmostEqual(after["iq_quadrature_departure_rad"], 0., places=14)

    def test_each_flag_combination_applies_only_the_requested_steps_in_order(self):
        p = self.parameters
        for bias, gain, quad in product((False, True), repeat=3):
            with self.subTest(bias=bias, gain=gain, quadrature=quad):
                flags = dict(correct_iq_bias=bias, correct_iq_gain=gain, correct_iq_quadrature=quad)
                i = self.raw.real - (p["i_bias"] if bias else 0.)
                q = (self.raw.imag - (p["q_bias"] if bias else 0.)) * (p["gain_imbalance"] if gain else 1.)
                if quad:
                    q = (q-i*np.sin(p["quadrature_rad"])) / np.cos(p["quadrature_rad"])
                np.testing.assert_allclose(apply(self.raw, **p, **flags), i+1j*q, atol=1e-14)

    def test_positive_and_negative_quadrature_and_gain_ratios(self):
        for theta, gain in product((-.7, 0., .7), (.4, 1., 2.5)):
            self.parameters.update(quadrature_rad=theta, gain_imbalance=gain)
            actual = apply(self.distort(self.i+1j*self.q), **self.parameters, **self.flags)
            np.testing.assert_allclose(actual, self.i+1j*self.q, atol=1e-14)

    def test_recovers_known_doppler_instead_of_matching_an_annotation(self):
        pri, frequency = .01, 7.
        phase = 2*np.pi*(frequency*np.arange(128)[:, None]*pri + np.arange(64)[None, :]/64)
        ideal = np.exp(1j*phase)
        raw = self.distort(ideal)
        measured = correction_parameters(analyze_batches([raw]), **self.flags)
        recovered = apply(raw, **measured["applied_parameters"], **self.flags)
        np.testing.assert_allclose(recovered, ideal, atol=2e-14)
        np.testing.assert_allclose(-np.angle(lag1_accc(recovered))/(2*np.pi*pri), frequency, atol=1e-12)

    def test_predefined_statistics_override_measurements_without_hidden_fallback(self):
        measured = analyze_batches([self.raw])
        selected = correction_parameters(measured, **self.flags, predefined_i_bias=.5,
                                         predefined_q_bias=.25, predefined_gain_imbalance=.8,
                                         predefined_quadrature_rad=.1)
        expected = apply(self.raw, i_bias=.5, q_bias=.25, gain_imbalance=.8, quadrature_rad=.1, **self.flags)
        np.testing.assert_array_equal(apply(self.raw, **selected["applied_parameters"], **self.flags), expected)
        self.assertEqual(set(selected["sources"].values()), {"predefined"})

    def test_all_disabled_accepts_undefined_statistics_and_returns_independent_copy(self):
        result = apply(self.raw, **dict.fromkeys(self.parameters),
                       **dict.fromkeys(self.flags, False))
        np.testing.assert_array_equal(result, self.raw)
        self.assertFalse(np.shares_memory(result, self.raw))

    def test_input_is_not_modified_and_complex_precision_is_preserved(self):
        for dtype, tolerance in ((np.complex64, 3e-7), (np.complex128, 1e-14)):
            raw = self.raw.astype(dtype)
            before = raw.copy()
            raw.flags.writeable = False
            result = apply(raw, **self.parameters, **self.flags)
            self.assertEqual(result.dtype, dtype)
            np.testing.assert_array_equal(raw, before)
            np.testing.assert_allclose(result, self.i+1j*self.q, atol=tolerance)

    def test_batch_partition_does_not_change_the_correction(self):
        entire = apply(self.raw, **self.parameters, **self.flags)
        partitioned = np.concatenate([apply(part, **self.parameters, **self.flags)
                                      for part in (self.raw[:3], self.raw[3:])])
        np.testing.assert_array_equal(partitioned, entire)

    def test_rejects_invalid_samples_parameters_and_flags(self):
        for raw in (np.ones((2, 3)), np.ones(3, complex), np.empty((0, 2), complex),
                    np.full((2, 3), np.nan+1j), np.full((2, 3), np.inf+1j)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                apply(raw, **self.parameters, **self.flags)
        for change in ({"i_bias": None}, {"i_bias": "bad"}, {"q_bias": [0.]},
                       {"gain_imbalance": 0.}, {"gain_imbalance": -1.},
                       {"quadrature_rad": np.pi/2}, {"quadrature_rad": -np.pi/2},
                       {"quadrature_rad": np.nan}, {"i_bias": 1j}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                apply(self.raw, **{**self.parameters, **change}, **self.flags)
        with self.assertRaises(ValueError):
            apply(self.raw, **self.parameters, **{**self.flags, "correct_iq_gain": "true"})

    def test_rejects_overflow_instead_of_returning_invalid_samples(self):
        raw = np.full((2, 4), 1+1e30j, dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "không hữu hạn"):
            apply(raw, **{**self.parameters, "gain_imbalance": 1e30}, **self.flags)


if __name__ == "__main__":
    unittest.main()
