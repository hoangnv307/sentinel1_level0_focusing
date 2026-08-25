import unittest

import numpy as np

from sentinel1_processing.dce import (
    evaluate_annotation_dce,
    prepare_annotation_records,
)
from sentinel1_processing.range_compression import compress_range
from sentinel1_processing.rcmc import build_sinc_table


class ProcessingTest(unittest.TestCase):
    def test_range_compression_matches_linear_convolution(self):
        data = np.arange(16, dtype=np.float32)[None, :].astype(np.complex64)
        times = np.arange(data.shape[1], dtype=np.float64)
        result, result_times = compress_range(
            data,
            times,
            sample_rate_hz=4.0,
            pulse_start_frequency_hz=0.25,
            pulse_ramp_rate_hz_per_s=0.5,
            pulse_length_s=1.0,
            batch_lines=1,
        )

        n_tx = 4
        tx_time = np.arange(n_tx) / 4.0 - (n_tx - 1) / 8.0
        replica = np.exp(2j * np.pi * (0.5 * tx_time + 0.25 * tx_time**2))
        matched_filter = np.conjugate(replica[::-1]) / np.linalg.norm(replica)
        same = np.convolve(data[0], matched_filter, mode="same")
        np.testing.assert_allclose(result[0], same[1:-2], rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(result_times, times[1:-2])

    def test_dce_interpolation_and_sinc_normalization(self):
        records = prepare_annotation_records([
            {"azimuthTime": "2025-01-01T00:00:00", "t0": 0.0,
             "dataDcPolynomial": [1.0, 2.0]},
            {"azimuthTime": "2025-01-01T00:00:02", "t0": 0.0,
             "dataDcPolynomial": [3.0, 4.0]},
        ])
        middle = (records[0]["azimuth_s"] + records[1]["azimuth_s"]) / 2.0
        np.testing.assert_allclose(
            evaluate_annotation_dce(records, middle, [0.0, 1.0]),
            [2.0, 5.0],
        )
        _, table = build_sinc_table()
        np.testing.assert_allclose(table.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
