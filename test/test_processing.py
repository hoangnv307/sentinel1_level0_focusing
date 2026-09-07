from pathlib import Path
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import sentinel1decoder
from scipy.fft import fftfreq, fftshift, ifft, ifftshift

import sentinel1_processing.azimuth_pre_processing as azimuth_pre_processing
import sentinel1_processing.azimuth_processing as azimuth_processing
import sentinel1_processing.dce_plotting as dce_plotting
import sentinel1_processing.doppler_centroid_estimation as doppler_centroid_estimation
import sentinel1_processing.range_processing as range_processing
import sentinel1_processing.raw_data_correction as raw_data_correction
import sentinel1_processing.s6_parameters as s6_parameters


class ProcessingTest(unittest.TestCase):
    def test_focus_slc_writes_into_supplied_array(self):
        source = np.zeros((4, 3), dtype=np.complex64)
        output = np.empty_like(source)
        layout = azimuth_processing.processing_blocks.ProcessingBlockLayout(
            matched_filter_support_samples=0,
            overlap_samples=0,
            step_samples=4,
            support_probe_indices=np.array([0]),
            support_probe_samples=(0,),
        )
        velocity = SimpleNamespace(
            evaluate_block=lambda **_kwargs: np.ones(3)
        )

        with patch.object(
            azimuth_processing.processing_blocks,
            "focus_block",
            return_value=np.ones_like(source),
        ):
            result = azimuth_processing.processing_blocks.focus_slc(
                source,
                np.arange(3.0) + 1.0,
                np.arange(4.0),
                lambda _line: np.zeros(3),
                velocity,
                layout,
                wavelength_m=1.0,
                speed_of_light_mps=1.0,
                azimuth_sample_period_s=1.0,
                range_sample_period_s=1.0,
                range_sample_frequency_hz=1.0,
                processing_bandwidth_hz=1.0,
                fft_length=4,
                output=output,
            )

        self.assertIs(result, output)
        np.testing.assert_array_equal(output, np.ones_like(output))

    def test_focus_slc_crops_invalid_azimuth_support(self):
        source = np.zeros((8, 3), dtype=np.complex64)
        layout = azimuth_processing.processing_blocks.ProcessingBlockLayout(
            matched_filter_support_samples=1,
            overlap_samples=2,
            step_samples=2,
            support_probe_indices=np.array([0]),
            support_probe_samples=(1,),
        )
        geometry = (
            azimuth_processing.processing_blocks.L1OutputGeometry.from_focus_support(
                np.arange(8.0), 3, layout, azimuth_sample_period_s=1.0
            )
        )
        output = np.empty(geometry.shape, dtype=np.complex64)
        velocity = SimpleNamespace(evaluate_block=lambda **_kwargs: np.ones(3))

        with patch.object(
            azimuth_processing.processing_blocks,
            "focus_block",
            return_value=np.ones((4, 3), dtype=np.complex64),
        ):
            result = azimuth_processing.processing_blocks.focus_slc(
                source,
                np.arange(3.0) + 1.0,
                np.arange(8.0),
                lambda _line: np.zeros(3),
                velocity,
                layout,
                wavelength_m=1.0,
                speed_of_light_mps=1.0,
                azimuth_sample_period_s=1.0,
                range_sample_period_s=1.0,
                range_sample_frequency_hz=1.0,
                processing_bandwidth_hz=1.0,
                fft_length=4,
                output_geometry=geometry,
                output=output,
            )

        self.assertEqual(geometry.shape, (4, 3))
        self.assertEqual(geometry.first_zero_doppler_time_s, 2.0)
        self.assertEqual(geometry.last_zero_doppler_time_s, 5.0)
        self.assertIs(result, output)
        np.testing.assert_array_equal(output, np.ones_like(output))

    def test_output_geometry_an_chors_zero_doppler_slice_to_pri(self):
        # DAD §8.3.1: pass zero-Doppler first/last output times so the SLC slice
        # is laid on the PRI grid instead of only the symmetric aperture margin.
        n = 64
        pri = 0.5
        times = np.arange(n, dtype=np.float64) * pri
        layout = azimuth_processing.processing_blocks.ProcessingBlockLayout(
            matched_filter_support_samples=0,
            overlap_samples=0,
            step_samples=10,
            support_probe_indices=np.array([0]),
            support_probe_samples=(0,),
        )
        # DAD §8.3.1 moves the first time to the next PRI, while the last
        # included line must not extend beyond its requested time.
        required_first = times[8] + 0.25 * pri
        required_last = times[55] + 0.25 * pri
        geometry = (
            azimuth_processing.processing_blocks.L1OutputGeometry.from_focus_support(
                times,
                3,
                layout,
                azimuth_sample_period_s=pri,
                required_first_time_s=required_first,
                required_last_time_s=required_last,
            )
        )
        self.assertEqual(geometry.azimuth_start_line, 9)
        self.assertEqual(geometry.azimuth_stop_line, 56)
        self.assertEqual(geometry.shape[0], 56 - 9)
        self.assertEqual(geometry.first_zero_doppler_time_s, times[9])
        self.assertEqual(geometry.last_zero_doppler_time_s, times[55])

        equation_8_15 = (
            azimuth_processing.processing_blocks.L1OutputGeometry.from_focus_support(
                times,
                3,
                azimuth_processing.processing_blocks.ProcessingBlockLayout(
                    matched_filter_support_samples=4,
                    overlap_samples=6,
                    step_samples=10,
                    support_probe_indices=np.array([0]),
                    support_probe_samples=(4,),
                ),
                azimuth_sample_period_s=pri,
                nominal_dc_time_offset_s=0.25,
                slice_overlap_s=1.0,
            )
        )
        # 1 slice-overlap + 0.5 nominal-DC + 2 Tmf + 2 extra = 5.5 PRI.
        self.assertEqual(equation_8_15.azimuth_start_line, 6)

    def test_output_geometry_uses_asymmetric_dc_support(self):
        layout = azimuth_processing.processing_blocks.ProcessingBlockLayout(
            matched_filter_support_samples=10,
            overlap_samples=12,
            step_samples=10,
            support_probe_indices=np.array([0]),
            support_probe_samples=(10,),
        )
        geometry = (
            azimuth_processing.processing_blocks.L1OutputGeometry.from_focus_support(
                np.arange(100.0),
                3,
                layout,
                azimuth_sample_period_s=1.0,
                dc_time_offsets_s=np.array([-4.0, -1.0]),
            )
        )

        self.assertEqual(geometry.azimuth_start_line, 6)
        self.assertEqual(geometry.azimuth_stop_line, 89)

    def test_prepared_scene_aligns_segments_into_supplied_array(self):
        first = doppler_centroid_estimation.Segment(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.complex64),
            np.arange(4.0),
            np.array([0.0, 1.0]),
            name="first",
        )
        second = doppler_centroid_estimation.Segment(
            np.array([[9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.complex64),
            np.arange(1.0, 5.0),
            np.array([2.0, 3.0]),
            name="second",
        )
        prepared = doppler_centroid_estimation.prepare_segments(
            [first, second], prf_hz=1.0
        )
        output = np.empty((4, 5), dtype=np.complex64)

        result = prepared.align_into(output, batch_lines=1)

        self.assertIs(result, output)
        np.testing.assert_array_equal(
            output,
            [[1, 2, 3, 4, 0], [5, 6, 7, 8, 0], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16]],
        )

        product_grid = doppler_centroid_estimation.prepare_segments(
            [first, second],
            prf_hz=1.0,
            common_range_start_s=1.0,
            common_range_samples=3,
        )
        cropped = np.empty((4, 3), dtype=np.complex64)
        product_grid.align_into(cropped)
        np.testing.assert_array_equal(
            cropped,
            [[2, 3, 4], [6, 7, 8], [9, 10, 11], [13, 14, 15]],
        )

        fractional = doppler_centroid_estimation.Segment(
            np.ones((1, 2), dtype=np.complex64),
            np.array([0.25, 1.25]),
            np.array([4.0]),
            name="fractional",
        )
        with self.assertRaisesRegex(ValueError, "range_time_shift_s"):
            doppler_centroid_estimation.prepare_segments(
                [first, fractional], prf_hz=1.0
            )

    def test_dce_fit_ignores_zero_quality_points(self):
        coefficients, valid, _ = doppler_centroid_estimation.fit_polynomial(
            np.arange(5.0),
            np.array([0.0, 1.0, 100.0, 3.0, 4.0]),
            t0_s=0.0,
            degree=1,
            outlier_sigma=1e9,
            weights=np.array([1.0, 1.0, 0.0, 1.0, 1.0]),
        )

        np.testing.assert_allclose(coefficients, [0.0, 1.0], atol=1e-12)
        np.testing.assert_array_equal(valid, [True, True, False, True, True])

    def test_s6_annotation_fit_reproduces_ipf_valid_points(self):
        root = ElementTree.parse(
            Path(__file__).parents[1]
            / "references"
            / "s1a-s6-slc-vv-20251226t214357-20251226t214426-062491-07d496-002.xml"
        ).getroot()
        records = root.findall("./dopplerCentroid/dcEstimateList/dcEstimate")

        for record_index, (record, rejected) in enumerate(
            zip(records, ((), (), (0,))), start=1
        ):
            points = record.findall("./fineDceList/fineDce")
            times = np.array([
                float(point.findtext("slantRangeTime")) for point in points
            ])
            frequencies = np.array([
                float(point.findtext("frequency")) for point in points
            ])
            coefficients, valid, rms = doppler_centroid_estimation.fit_polynomial(
                times,
                frequencies,
                t0_s=float(record.findtext("t0")),
                degree=2,
                outlier_sigma=2.5,
            )
            expected_valid = np.ones(len(points), dtype=bool)
            expected_valid[list(rejected)] = False

            with self.subTest(record=record_index):
                np.testing.assert_array_equal(valid, expected_valid)
                np.testing.assert_allclose(
                    coefficients,
                    np.fromstring(record.findtext("dataDcPolynomial"), sep=" "),
                    rtol=1e-5,
                )
                self.assertAlmostEqual(
                    rms, float(record.findtext("dataDcRmsError")), places=5
                )

    def test_s6_range_blocks_match_annotation_layout(self):
        config = doppler_centroid_estimation.Config.for_stripmap_s6()
        sample_rate_hz = 46_918_402.8
        root = ElementTree.parse(
            Path(__file__).parents[1]
            / "references"
            / "s1a-s6-slc-vv-20251226t214357-20251226t214426-062491-07d496-002.xml"
        ).getroot()
        record = root.find("./dopplerCentroid/dcEstimateList/dcEstimate")
        dce_range_start_s = float(record.findtext("t0"))
        annotation_times = np.array([
            float(point.findtext("slantRangeTime"))
            for point in record.findall("./fineDceList/fineDce")
        ])
        common_range_start_s = dce_range_start_s - 81.25 / sample_rate_hz
        common_range_times_s = (
            common_range_start_s + np.arange(17634) / sample_rate_hz
        )
        blocks = doppler_centroid_estimation.build_range_blocks(
            common_range_times_s.size,
            common_range_times_s,
            config,
            range_grid_start_s=dce_range_start_s,
        )

        self.assertEqual(
            [block.start_sample for block in blocks],
            [
                81, 949, 1818, 2687, 3556, 4424, 5293, 6162, 7031, 7900,
                8768, 9637, 10506, 11375, 12244, 13112, 13981, 14850,
                15719, 16588,
            ],
        )
        self.assertEqual(blocks[-1].stop_sample, 17588)
        np.testing.assert_allclose(
            [block.center_slant_range_time_s for block in blocks],
            annotation_times,
            atol=7e-11,
            rtol=0.0,
        )

    def test_s6_output_geometry_contract_matches_annotation(self):
        root = ElementTree.parse(
            Path(__file__).parents[1]
            / "references"
            / "s1a-s6-slc-vv-20251226t214357-20251226t214426-062491-07d496-002.xml"
        ).getroot()
        info = root.find("./imageAnnotation/imageInformation")
        first = datetime.fromisoformat(info.findtext("productFirstLineUtcTime"))
        last = datetime.fromisoformat(info.findtext("productLastLineUtcTime"))
        sensing_start = datetime.fromisoformat(
            info.findtext("./sliceList/slice/sensingStartTime")
        )

        self.assertEqual(
            (s6_parameters.SLC_AZIMUTH_LINES, s6_parameters.SLC_RANGE_SAMPLES),
            (int(info.findtext("numberOfLines")), int(info.findtext("numberOfSamples"))),
        )
        self.assertEqual(
            s6_parameters.SLC_RANGE_START_TIME_S,
            float(info.findtext("slantRangeTime")),
        )
        self.assertAlmostEqual(
            s6_parameters.SLC_ZERO_DOP_MINUS_ACQ_TIME_S,
            (first - sensing_start).total_seconds(),
            places=6,
        )
        self.assertAlmostEqual(
            (last - first).total_seconds(),
            (s6_parameters.SLC_AZIMUTH_LINES - 1)
            * s6_parameters.SLC_AZIMUTH_TIME_INTERVAL_S,
            places=6,
        )

    def test_s6_estimated_fine_dc_matches_l1_annotation(self):
        project = Path(__file__).parents[1]
        raw_path = (
            project / "data/sao_paulo/"
            "s1a-s6-raw-s-vv-20251226t214356-20251226t214427-062491-07d496.dat"
        )
        pair_cache = project / ".cache/sentinel1/chunks-13-14"
        cache_paths = {
            13: pair_cache / "range-compression-13/data.npy",
            14: pair_cache / "range-compression-14/data.npy",
        }
        legacy_paths = {
            13: project / ".cache/sentinel1/dce-range-compression-13/data.npy",
            14: project / ".cache/sentinel1/range-compression-14/data.npy",
        }
        cache_paths = {
            chunk: path if path.exists() else legacy_paths[chunk]
            for chunk, path in cache_paths.items()
        }
        if not raw_path.exists() or not all(path.exists() for path in cache_paths.values()):
            self.skipTest("Cần dữ liệu L0 và cache range-compression S6 để test parity.")

        l0file = sentinel1decoder.Level0File(str(raw_path))
        metadata = {
            chunk: l0file.get_acquisition_chunk_metadata(chunk)
            for chunk in cache_paths
        }
        selected = metadata[14]
        sample_rate_hz = sentinel1decoder.utilities.range_dec_to_sample_rate(
            selected["Range Decimation"].iloc[0]
        )
        suppressed_data_time_s = 320.0 / (
            8.0 * sentinel1decoder.constants.F_REF
        )

        def axes(chunk):
            table = metadata[chunk]
            count = 2 * int(table["Number of Quads"].iloc[0])
            range_times = (
                table["Rank"].iloc[0] * table["PRI"].iloc[0]
                + table["SWST"].iloc[0]
                + suppressed_data_time_s
                + np.arange(count) / sample_rate_hz
                - s6_parameters.SWST_BIAS_S
            )
            azimuth_times = (
                table["Coarse Time"].to_numpy(dtype=float)
                + table["Fine Time"].to_numpy(dtype=float)
            )
            return range_times, azimuth_times

        native_axes = {chunk: axes(chunk) for chunk in cache_paths}
        common_start_s = min(values[0][0] for values in native_axes.values())
        pulse_samples = int(np.ceil(
            selected["Tx Pulse Length"].iloc[0] * sample_rate_hz
        ))
        segments = []
        for chunk in (13, 14):
            range_times, azimuth_times = native_axes[chunk]
            delta_samples = (range_times[0] - common_start_s) * sample_rate_hz
            fractional_shift_s = (
                round(delta_samples) / sample_rate_hz
                - (range_times[0] - common_start_s)
            )
            segments.append(doppler_centroid_estimation.Segment(
                np.load(cache_paths[chunk], mmap_mode="r"),
                range_times[:-pulse_samples + 1] + fractional_shift_s,
                azimuth_times,
                name=f"chunk-{chunk}",
            ))

        config = replace(
            doppler_centroid_estimation.Config.for_stripmap_s6(),
            accc_range_weighting="phase",
        )
        estimator = doppler_centroid_estimation.Estimator(
            1.0 / float(selected["PRI"].iloc[0]), config
        )
        first_time_s = segments[0].azimuth_times_s[0]
        last_time_s = segments[-1].azimuth_times_s[-1] + float(
            selected["PRI"].iloc[0]
        )
        estimates = estimator.estimate_segments(
            segments,
            dce_range_start_s=native_axes[13][0][0],
            known_ambiguity_number=0,
            slice_start_times_s=[first_time_s],
            last_slice_stop_time_s=last_time_s,
            product_start_time_s=first_time_s,
            product_stop_time_s=last_time_s,
            zero_dop_minus_acq_time_s=0.0,
        )
        annotation = ElementTree.parse(
            project / "references/"
            "s1a-s6-slc-vv-20251226t214357-20251226t214426-062491-07d496-002.xml"
        ).getroot().findall("./dopplerCentroid/dcEstimateList/dcEstimate")

        self.assertEqual(len(estimates), len(annotation))
        for record_index, (estimated, reference) in enumerate(
            zip(estimates, annotation), start=1
        ):
            reference_hz = np.array([
                float(point.findtext("frequency"))
                for point in reference.findall("./fineDceList/fineDce")
            ])
            rmse_hz = float(np.sqrt(np.mean(
                (estimated.fine_absolute_hz - reference_hz) ** 2
            )))
            with self.subTest(record=record_index):
                self.assertLessEqual(
                    rmse_hz,
                    s6_parameters.DCE_L1_FINE_RMSE_THRESHOLD_HZ,
                    f"DCE{record_index} Fine-DC RMSE = {rmse_hz:.3f} Hz",
                )

    def test_segment_dce_grid_is_independent_of_union_buffer(self):
        prf_hz = 100.0
        eta = np.arange(4) / prf_hz
        phase = np.exp(2j * np.pi * 10.0 * eta)
        first = doppler_centroid_estimation.Segment(
            phase[:2, None] * np.ones((2, 6)),
            2.0 + np.arange(6),
            eta[:2],
            name="first",
        )
        second = doppler_centroid_estimation.Segment(
            phase[2:, None] * np.ones((2, 8)),
            np.arange(8.0),
            eta[2:],
            name="second",
        )
        estimator = doppler_centroid_estimation.Estimator(
            prf_hz,
            doppler_centroid_estimation.Config(
                azimuth_block_size_lines=4,
                num_range_blocks=1,
                range_block_size_samples=4,
                range_roi_stop=4,
                azimuth_placement="custom",
                polynomial_degree=0,
            ),
        )

        records, prepared = estimator.estimate_segments(
            [first, second],
            dce_range_start_s=2.25,
            known_ambiguity_number=0,
            custom_azimuth_starts=[0],
            return_prepared_scene=True,
        )

        self.assertEqual(prepared.common_slant_range_times_s[0], 0.0)
        self.assertEqual(records[0].t0_s, 2.25)
        self.assertEqual(records[0].range_blocks[0].start_sample, 2)
        self.assertEqual(records[0].range_blocks[0].center_slant_range_time_s, 4.25)
        np.testing.assert_allclose(records[0].fine_baseband_hz, 10.0)

    def test_dce_plotting_splits_records_and_marks_rejected_points(self):
        records = [
            {
                "record": index,
                "range_times_s": np.array([0.001, 0.002]),
                "annotation_hz": np.array([10.0, 11.0]),
                "estimated_hz": np.array([10.1, 10.9]),
                "geometry_hz": np.array([9.5, 9.7]),
                "fit_range_times_s": np.array([0.001, 0.002]),
                "fit_points_hz": np.array([10.2, 15.0]),
                "fit_valid_mask": np.array([True, False]),
                "rmse_hz": 0.1,
                "fit_rms_hz": 0.2,
            }
            for index in range(1, 4)
        ]

        figures = dce_plotting.plot_comparisons(records)
        try:
            self.assertEqual(len(figures), 3)
            for figure in figures:
                self.assertEqual(len(figure.axes), 1)
                self.assertEqual(len(figure.axes[0].lines), 3)
                self.assertEqual(len(figure.axes[0].collections), 2)
        finally:
            for figure in figures:
                plt.close(figure)

        velocity_figure = dce_plotting.plot_effective_velocity(
            np.array([800_000.0, 900_000.0]),
            SimpleNamespace(
                vr_mps=np.array([7200.0, 7210.0]),
                control_ranges_m=np.array([800_000.0, 900_000.0]),
                control_vr_mps=np.array([7201.0, 7209.0]),
            ),
        )
        try:
            self.assertEqual(len(velocity_figure.axes[0].lines), 1)
            self.assertEqual(len(velocity_figure.axes[0].collections), 1)
        finally:
            plt.close(velocity_figure)

    def test_public_api_follows_dad_processing_steps(self):
        self.assertEqual(
            doppler_centroid_estimation.Config.for_stripmap_s6().unwrap_weighting,
            "coherence",
        )
        self.assertEqual(
            doppler_centroid_estimation.Config.for_stripmap_s6().fit_weighting,
            "uniform",
        )
        self.assertEqual(
            doppler_centroid_estimation.Config.for_stripmap_s6().accc_range_weighting,
            "phase",
        )
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
        np.testing.assert_array_equal(result_times, times[:13])

        supplied_output = np.empty_like(result)
        supplied_result, _ = azimuth_pre_processing.range.compression.compress(
            data,
            times,
            sample_rate_hz=4.0,
            pulse_start_frequency_hz=0.25,
            pulse_ramp_rate_hz_per_s=0.5,
            pulse_length_s=1.0,
            batch_lines=1,
            range_reference_function=reference_function,
            output_array=supplied_output,
        )
        self.assertIs(supplied_result, supplied_output)
        np.testing.assert_allclose(supplied_result, result, rtol=2e-6, atol=2e-6)

        shifted, shifted_times = azimuth_pre_processing.range.compression.compress(
            data,
            times,
            sample_rate_hz=4.0,
            pulse_start_frequency_hz=0.25,
            pulse_ramp_rate_hz_per_s=0.5,
            pulse_length_s=1.0,
            range_time_shift_s=-0.125,
        )
        np.testing.assert_array_equal(shifted_times, times[:13] - 0.125)
        self.assertFalse(np.allclose(shifted, result))

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

        phase_weighted, _, _ = doppler_centroid_estimation.estimate_fine_dc(
            np.array([
                [1.0, 100.0],
                [1.0, 100.0j],
            ]),
            [doppler_centroid_estimation.RangeBlock(0, 2, 1.0, 0.0)],
            1000.0,
            range_weighting="phase",
        )
        self.assertAlmostEqual(phase_weighted[0], 125.0)
        gained = range_processing.dependent_gain.apply(
            np.ones((1, 2), dtype=np.complex64), [2.0, 3.0]
        )
        np.testing.assert_array_equal(gained, [[2.0, 3.0]])
        np.testing.assert_array_equal(
            range_processing.swst_bias.correct([1.0, 2.0], 0.25),
            [0.75, 1.75],
        )

    def test_fractional_shift_direction_is_a_delay(self):
        # DAD §6.1.3: fractional SWST is realised as a phase ramp in the RRF.
        # A positive ``range_time_shift_s`` must DELAY the echoed content (the
        # compressed peak moves later in time), matching the Fourier convention
        # used to place each chunk on the common fast-time grid after black-fill.
        fs = 64.0
        num_tx = int(np.ceil(0.25 * fs))
        tx_time = np.arange(num_tx) / fs - (num_tx - 1) / (2.0 * fs)
        replica = np.exp(
            2j * np.pi * (8.0 * tx_time + 48.0 / 2.0 * tx_time**2)
        )
        n_range = 2048
        times = np.arange(n_range, dtype=np.float64) / fs
        data = np.zeros((1, n_range), dtype=np.complex64)
        data[0, 300:300 + num_tx] = replica
        reference = range_processing.reference_function.calculate(
            sample_rate_hz=fs,
            pulse_start_frequency_hz=8.0,
            pulse_ramp_rate_hz_per_s=48.0,
            pulse_length_s=0.25,
            fft_length=4096,
        )
        half_sample = 0.5 / fs

        def peak_time(shift_s):
            compressed, compressed_times = (
                azimuth_pre_processing.range.compression.compress(
                    data,
                    times,
                    sample_rate_hz=fs,
                    pulse_start_frequency_hz=8.0,
                    pulse_ramp_rate_hz_per_s=48.0,
                    pulse_length_s=0.25,
                    batch_lines=1,
                    range_reference_function=reference,
                    range_time_shift_s=shift_s,
                )
            )
            peak_index = int(np.argmax(np.abs(compressed[0])))
            return float(compressed_times[peak_index])

        base = peak_time(0.0)
        delayed = peak_time(0.5 * half_sample)
        advanced = peak_time(-0.5 * half_sample)
        # Positive shift delays the content, a half-sample shift must move the
        # peak by approximately that amount.
        self.assertGreater(delayed, base)
        self.assertLess(advanced, base)
        self.assertAlmostEqual(delayed - base, 0.5 / fs, delta=0.5 / fs * 0.5)
        self.assertAlmostEqual(base - advanced, 0.5 / fs, delta=0.5 / fs * 0.5)

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

    def test_dce_accepts_independently_known_ambiguity(self):
        estimator = doppler_centroid_estimation.Estimator(
            1000.0,
            doppler_centroid_estimation.Config(
                azimuth_block_size_lines=2,
                num_range_blocks=3,
                polynomial_degree=1,
                azimuth_spacing_lines=2,
                fit_weighting="uniform",
            ),
        )
        block = doppler_centroid_estimation.AzimuthBlock(
            0, 2, 1.0, 0.0, 2.0, 1.0
        )
        range_blocks = [
            doppler_centroid_estimation.RangeBlock(i, i + 1, i + 0.5, float(i))
            for i in range(3)
        ]
        with patch.object(
            doppler_centroid_estimation,
            "estimate_fine_dc",
            return_value=(
                np.array([10.0, 20.0, 30.0]),
                np.ones(3, dtype=np.complex128),
                np.ones(3),
            ),
        ):
            result = estimator.estimate_block(
                np.ones((2, 3), dtype=np.complex64),
                block,
                range_blocks,
                known_ambiguity_number=2,
            )

        self.assertTrue(result.absolute_ambiguity_resolved)
        self.assertEqual(result.ambiguity_number, 2)
        self.assertIsNone(result.geometry_dc_polynomial)
        np.testing.assert_allclose(
            result.fine_absolute_hz - result.fine_unwrapped_hz, 2000.0
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
            geometry_dc_polynomial=np.array([1.0, 1.0]),
            coherence=np.array([0.75]),
            rms_error_hz=0.5,
        )
        comparison = doppler_centroid_estimation.compare_with_annotations(
            [records[0]], [estimate], [0.0, 1.0], prf_hz=1000.0
        )[0]
        self.assertAlmostEqual(comparison["rmse_hz"], np.sqrt(2.5))
        np.testing.assert_allclose(comparison["geometry_hz"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
