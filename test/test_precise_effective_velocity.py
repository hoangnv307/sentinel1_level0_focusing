import unittest

import numpy as np
import pandas as pd

import sentinel1_processing.effective_velocity as effective_velocity


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

        estimator = effective_velocity.Estimator.from_ephemeris(
            ephemeris, 0.05
        )

        np.testing.assert_array_equal(estimator.orbit_times_s, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
