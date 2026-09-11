import unittest

import numpy as np
import pandas as pd

import sentinel1_processing.common as common


class EphemerisTest(unittest.TestCase):
    def test_sorts_and_deduplicates_epochs(self):
        ephemeris = pd.DataFrame({
            "POD Solution Data Timestamp": [2, 1, 2, 3],
            "X-axis position ECEF": [2, 1, 2, 3],
            "Y-axis position ECEF": [0, 0, 0, 0],
            "Z-axis position ECEF": [0, 0, 0, 0],
            "X-axis velocity ECEF": [1, 1, 1, 1],
            "Y-axis velocity ECEF": [0, 0, 0, 0],
            "Z-axis velocity ECEF": [0, 0, 0, 0],
        })

        estimator = common.effective_velocity.Estimator.from_ephemeris(
            ephemeris, 0.05
        )

        np.testing.assert_array_equal(estimator.orbit_times_s, [1, 2, 3])
        estimator.validate_time_coverage([1.5, 2.5])
        estimator.validate_time_coverage(
            [0.5, 3.5], max_extrapolation_s=0.5
        )
        with self.assertRaises(ValueError):
            estimator.validate_time_coverage([0.5, 2.5])

        np.testing.assert_allclose(estimator.position([1, 2, 3])[:, 0], [1, 2, 3])
        np.testing.assert_allclose(estimator.velocity(2), [1, 0, 0], atol=1e-12)
        smooth = common.effective_velocity.Estimator.from_ephemeris(
            ephemeris, 0.05, smooth_positions=True
        )
        np.testing.assert_allclose(smooth.velocity(2), [1, 0, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
