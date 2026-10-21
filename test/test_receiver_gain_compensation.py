"""Đối chứng độc lập DAD §9.2.3: gain, đơn vị, mốc TX/RX và lỗi đầu vào."""

import unittest

import numpy as np

from sentinel1_processing.common.receiver_gain_compensation import calculate_gain


class ReceiverGainCompensationTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(sample_rate_hz=1e6, swst_s=90e-6, pulse_length_s=50e-6,
                           suppressed_s=1e-6, gain_trend_coefficients=[10, 0, 0, 0, 0, 1, 0, 0],
                           gain_overshoot_coefficients=[0]*5, enabled=True)

    def test_ten_db_power_gain_requires_sqrt_for_complex_amplitude_correction(self):
        gain = calculate_gain(8, **self.config)
        np.testing.assert_allclose(gain, 10.)
        ideal = np.array([[1+2j, -3+4j, 5-6j, 7+8j]*2])
        np.testing.assert_allclose(ideal*np.sqrt(10) / np.sqrt(gain), ideal)

    def test_full_rational_trend_and_overshoot_match_explicit_scalar_formula(self):
        coefficients = [.1, .02, .003, .00004, .0000005, 2., .04, .0006]
        overshoot = [.8, .2, -.3, .7, -.4]
        actual = calculate_gain(7, **{**self.config, "gain_trend_coefficients": coefficients,
                                     "gain_overshoot_coefficients": overshoot})
        expected = []
        for sample in range(7):
            ts = float(sample)
            t1 = ts+41.
            trend = (.1+.02*t1+.003*t1**2+.00004*t1**3+.0000005*t1**4)/(2+.04*t1+.0006*t1**2)
            oscillation = .8*np.exp(-.2*ts)*np.sin(.7*ts-.4)-.3
            expected.append(10**((trend+oscillation)/10))
        np.testing.assert_allclose(actual, expected, rtol=2e-15)

    def test_microseconds_and_suppressed_time_are_applied_once(self):
        actual = calculate_gain(3, **{**self.config, "gain_trend_coefficients": [0, .1, 0, 0, 0, 1, 0, 0]})
        np.testing.assert_allclose(actual, 10**(np.array([4.1, 4.2, 4.3])/10), rtol=2e-15)

    def test_swst_changes_trend_while_overshoot_restarts_at_first_retained_sample(self):
        coefficients = [0, .1, 0, 0, 0, 1, 0, 0]
        common = {**self.config, "gain_trend_coefficients": coefficients,
                  "gain_overshoot_coefficients": [1, .1, .2, .3, .4]}
        first = calculate_gain(16, **common)
        second = calculate_gain(16, **{**common, "swst_s": 100e-6})
        np.testing.assert_allclose(second/first, 10**(.1), rtol=2e-15)

    def test_disabled_returns_identity_without_evaluating_missing_or_singular_coefficients(self):
        for coefficients in (None, [0]*8):
            gain = calculate_gain(3, **{**self.config, "enabled": False,
                                        "gain_trend_coefficients": coefficients,
                                        "gain_overshoot_coefficients": None})
            np.testing.assert_array_equal(gain, np.ones(3))

    def test_single_sample_and_input_coefficients_are_not_modified(self):
        before = np.array(self.config["gain_trend_coefficients"])
        before.flags.writeable = False
        gain = calculate_gain(1, **{**self.config, "gain_trend_coefficients": before})
        np.testing.assert_allclose(gain, [10.])
        np.testing.assert_array_equal(before, self.config["gain_trend_coefficients"])

    def test_rejects_invalid_count_timing_and_flag(self):
        for count in (0, -1, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                calculate_gain(count, **self.config)
        for change in ({"sample_rate_hz": 0.}, {"swst_s": -1.}, {"pulse_length_s": 0.},
                       {"suppressed_s": -1.}, {"swst_s": np.nan}, {"sample_rate_hz": np.inf},
                       {"swst_s": [0.]}, {"swst_s": "bad"}, {"swst_s": 1j}, {"enabled": "true"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                calculate_gain(4, **{**self.config, **change})

    def test_rejects_invalid_coefficient_vectors(self):
        for change in ({"gain_trend_coefficients": [0]*7}, {"gain_overshoot_coefficients": [0]*4},
                       {"gain_trend_coefficients": [np.nan]*8}, {"gain_overshoot_coefficients": [np.inf]*5},
                       {"gain_trend_coefficients": np.ones((1, 8))}, {"gain_overshoot_coefficients": [1j]*5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                calculate_gain(4, **{**self.config, **change})

    def test_rejects_zero_denominator_at_any_sample_and_nonfinite_polynomial(self):
        for denominator in ([0., 0., 0.], [-42., 1., 0.], [1e308, 1e308, 1e308]):
            with self.subTest(denominator=denominator), self.assertRaisesRegex(ValueError, "mẫu số"):
                calculate_gain(4, **{**self.config, "gain_trend_coefficients": [1, 0, 0, 0, 0, *denominator]})

    def test_rejects_gain_overflow_and_underflow(self):
        for value in (10000., -10000.):
            with self.subTest(db=value), self.assertRaisesRegex(ValueError, "không hữu hạn hoặc không dương"):
                calculate_gain(4, **{**self.config, "gain_trend_coefficients": [value, 0, 0, 0, 0, 1, 0, 0]})


if __name__ == "__main__":
    unittest.main()
