from pathlib import Path
import unittest

import numpy as np

import radarsat1_processing


RAW = Path(__file__).parents[1] / "data/radarsat-1/R1_20979_ST7_L0_F53/R1_20979_ST7_L0_F53.000.raw"
L1 = Path(__file__).parents[1] / "data/radarsat-1/R1_20979_ST7_F053/R1_20979_ST7_F053.L.txt"


@unittest.skipUnless(RAW.exists(), "không có dữ liệu RADARSAT-1 Level-0")
class Radarsat1Test(unittest.TestCase):
    def test_decodes_supplied_ceos_data(self):
        metadata = radarsat1_processing.read_metadata(RAW)
        data = radarsat1_processing.decode(RAW, line_count=2, sample_count=4)

        self.assertEqual(metadata.line_count, 22158)
        self.assertEqual(metadata.range_samples, 7650)
        self.assertEqual(metadata.beam, "ST7")
        self.assertAlmostEqual(metadata.sample_rate_hz, 12_926_830.0)
        self.assertAlmostEqual(metadata.prf_hz, 1293.976976976977)
        self.assertAlmostEqual(metadata.chirp_rate_hz_per_s, -11.78e6 / 42e-6)
        self.assertAlmostEqual(metadata.doppler_centroid_hz, -10910.71, places=2)
        self.assertAlmostEqual(metadata.effective_velocity_mps, 7045.4, places=0)
        self.assertEqual(data.dtype, np.complex64)
        np.testing.assert_array_equal(data.shape, (2, 4))
        np.testing.assert_array_equal(data.real % 2, 1)
        np.testing.assert_array_equal(data.imag % 2, 1)

    def test_extracts_l0_leader(self):
        leader = radarsat1_processing.read_leader(RAW.with_suffix(".ldr"))
        self.assertEqual([record["length"] for record in leader["records"]], [720, 4096, 8960])
        self.assertEqual(leader["records"][1]["scene_id"], "RSAT-1-SAR-RAW")
        np.testing.assert_allclose(
            leader["records"][1]["crt_doppler_hz"],
            [-10549.263, -0.10592052, 2.9867758e-6],
        )
        self.assertEqual(len(leader["records"][2]["state_vectors"]), 15)

    @unittest.skipUnless(L1.exists(), "không có metadata RADARSAT-1 L1")
    def test_orbit_geometry_and_l1_reference_are_independent(self):
        metadata = radarsat1_processing.read_metadata(RAW)
        times = 2 * np.array([1057.810275e3, 1132.358666e3]) / 299_792_458.0
        geometry = radarsat1_processing.geometry_doppler(RAW, times)
        reference = radarsat1_processing.read_l1_reference(L1)

        np.testing.assert_allclose(geometry, [-10371.2, -11224.5], atol=2.0)
        pixels = np.array([0, reference.pixel_count - 1])
        np.testing.assert_allclose(
            reference.evaluate(reference.slant_range_for_pixel(pixels)),
            np.polynomial.polynomial.polyval(pixels, reference.data_coefficients),
            atol=1e-6,
        )
        linear_midpoint = (reference.near_slant_range_m + reference.far_slant_range_m) / 2
        self.assertGreater(
            abs(reference.slant_range_for_pixel((reference.pixel_count - 1) / 2) - linear_midpoint),
            500,
        )


if __name__ == "__main__":
    unittest.main()
