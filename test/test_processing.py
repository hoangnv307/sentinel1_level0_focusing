import unittest

import numpy as np
from scipy.fft import fftfreq, fftshift, ifft, ifftshift

import sentinel1_processing.azimuth_pre_processing as azimuth_pre_processing
import sentinel1_processing.azimuth_processing as azimuth_processing
import sentinel1_processing.doppler_centroid_estimation as doppler_centroid_estimation
import sentinel1_processing.range_processing as range_processing
import sentinel1_processing.raw_data_correction as raw_data_correction


class ProcessingTest(unittest.TestCase):
    def test_public_api_follows_dad_processing_steps(self):
        self.assertEqual(
            azimuth_pre_processing.__all__,
            ["azimuth_zero_padding", "range", "azimuth_forward_fft"],
        )
        self.assertEqual(
            range_processing.__all__,
            ["reference_function", "dependent_gain", "swst_bias"],
        )
        self.assertEqual(
            azimuth_processing.__all__,
            [
                "secondary_range_compression",
                "range_cell_migration_correction",
                "azimuth_compression",
                "processing_blocks",
            ],
        )
        self.assertTrue(callable(azimuth_pre_processing.range.compression.compress))
        self.assertEqual(doppler_centroid_estimation.Segment.__name__, "Segment")
        self.assertEqual(doppler_centroid_estimation.Estimator.__name__, "Estimator")

    def test_range_compression_matches_linear_convolution(self):
        data = np.arange(16, dtype=np.float32)[None, :].astype(np.complex64)
        times = np.arange(data.shape[1], dtype=np.float64)
        reference_function = range_processing.reference_function.calculate(
            sample_rate_hz=4.0,
            pulse_start_frequency_hz=0.25,
            pulse_ramp_rate_hz_per_s=0.5,
            pulse_length_s=1.0,
            fft_length=32,
        )
        result, result_times = azimuth_pre_processing.range.compression.compress(
            data,
            times,
            sample_rate_hz=4.0,
            pulse_start_frequency_hz=0.25,
            pulse_ramp_rate_hz_per_s=0.5,
            pulse_length_s=1.0,
            batch_lines=1,
            range_reference_function=reference_function,
        )

        n_tx = 4
        tx_time = np.arange(n_tx) / 4.0 - (n_tx - 1) / 8.0
        replica = np.exp(2j * np.pi * (0.5 * tx_time + 0.25 * tx_time**2))
        matched_filter = np.conjugate(replica[::-1]) / np.linalg.norm(replica)
        same = np.convolve(data[0], matched_filter, mode="same")
        valid = np.convolve(data[0], matched_filter, mode="valid")
        np.testing.assert_allclose(result[0], valid, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(result_times, times[1:-2])

        same_result, same_times = azimuth_pre_processing.range.compression.compress(
            data,
            times,
            sample_rate_hz=4.0,
            pulse_start_frequency_hz=0.25,
            pulse_ramp_rate_hz_per_s=0.5,
            pulse_length_s=1.0,
            batch_lines=1,
            output="same",
        )
        np.testing.assert_allclose(same_result[0], same, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(same_times, times)

        biased = data + (2.0 - 3.0j)
        self.assertAlmostEqual(
            raw_data_correction.estimate_iq_bias(np.full((2, 3), 2.0 - 3.0j)),
            2.0 - 3.0j,
        )
        corrected, _ = azimuth_pre_processing.range.compression.compress(
            biased,
            times,
            sample_rate_hz=4.0,
            pulse_start_frequency_hz=0.25,
            pulse_ramp_rate_hz_per_s=0.5,
            pulse_length_s=1.0,
            batch_lines=1,
            iq_bias=2.0 - 3.0j,
        )
        np.testing.assert_allclose(corrected, result, rtol=2e-6, atol=2e-6)
        gained = range_processing.dependent_gain.apply(
            np.ones((1, 2), dtype=np.complex64), [2.0, 3.0]
        )
        np.testing.assert_array_equal(gained, [[2.0, 3.0]])
        np.testing.assert_array_equal(
            range_processing.swst_bias.correct([1.0, 2.0], 0.25),
            [0.75, 1.75],
        )

    def test_absolute_dce_uses_first_geometry_range_block(self):
        absolute, coefficients, _, ambiguity, _, _ = (
            doppler_centroid_estimation.resolve_absolute_dc(
                [0.0, 1.0, 2.0],
                [200.0, 210.0, 220.0],
                [1200.0, 100.0, 110.0],
                1000.0,
                t0_s=0.0,
                degree=1,
            )
        )

        self.assertEqual(ambiguity, 1)
        np.testing.assert_allclose(absolute, [1200.0, 1210.0, 1220.0])
        np.testing.assert_allclose(
            np.polynomial.polynomial.polyval([0.0, 1.0, 2.0], coefficients),
            absolute,
        )

    def test_positive_azimuth_filter_focuses_point_target(self):
        n = 64
        sample_period_s = 1.0 / 1600.0
        wavelength_m = 0.055
        slant_ranges_m = np.array([800_000.0])
        velocity_mps = np.array([7200.0])
        centroid_hz = np.array([30.0])
        frequency_hz = fftshift(fftfreq(n, d=sample_period_s)) + centroid_hz[0]
        d = np.sqrt(
            1.0 - (wavelength_m * frequency_hz / (2.0 * velocity_mps[0])) ** 2
        )
        point_echo_spectrum = np.exp(
            -4.0j * np.pi * slant_ranges_m[0] * d / wavelength_m
        )
        block = ifft(ifftshift(point_echo_spectrum))[:, None].astype(np.complex64)

        baseband_hz, range_doppler = (
            azimuth_pre_processing.azimuth_forward_fft.apply(
                block, sample_period_s
            )
        )
        focused = azimuth_processing.azimuth_compression.compress(
            range_doppler,
            baseband_hz,
            centroid_hz,
            velocity_mps,
            wavelength_m=wavelength_m,
            slant_ranges_m=slant_ranges_m,
        )

        self.assertAlmostEqual(abs(focused[0, 0]), 1.0, places=5)
        self.assertLess(np.max(np.abs(focused[1:, 0])), 1e-5)
        time_filter = (
            azimuth_processing.azimuth_compression
            .calculate_time_correction_filter(
                frequency_hz[:, None], 0.25
            )
        )
        np.testing.assert_allclose(
            time_filter[:, 0], np.exp(2j * np.pi * frequency_hz * 0.25)
        )

    def test_dce_interpolation_and_sinc_normalization(self):
        records = doppler_centroid_estimation.parse_annotation_records([
            {"azimuthTime": "2025-01-01T00:00:00", "t0": 0.0,
             "dataDcPolynomial": [1.0, 2.0]},
            {"azimuthTime": "2025-01-01T00:00:02", "t0": 0.0,
             "dataDcPolynomial": [3.0, 4.0]},
        ])
        middle = (records[0]["azimuth_s"] + records[1]["azimuth_s"]) / 2.0
        np.testing.assert_allclose(
            doppler_centroid_estimation.evaluate_annotation_records(
                records, middle, [0.0, 1.0]
            ),
            [2.0, 5.0],
        )
        _, table = (
            azimuth_processing.range_cell_migration_correction
            .build_interpolation_table()
        )
        np.testing.assert_allclose(table.sum(axis=1), 1.0)

        estimate = doppler_centroid_estimation.Estimate(
            block=doppler_centroid_estimation.AzimuthBlock(
                0, 2, 1.0, middle, middle, middle
            ),
            t0_s=0.0,
            range_blocks=[],
            fine_baseband_hz=np.array([0.0]),
            fine_unwrapped_hz=np.array([0.0]),
            fine_absolute_hz=np.array([0.0]),
            data_dc_polynomial=np.array([2.0, 3.0]),
            coherence=np.array([0.75]),
            rms_error_hz=0.5,
        )
        comparison = doppler_centroid_estimation.compare_with_annotations(
            [records[0]], [estimate], [0.0, 1.0], prf_hz=1000.0
        )[0]
        self.assertAlmostEqual(comparison["rmse_hz"], np.sqrt(2.5))


if __name__ == "__main__":
    unittest.main()
