import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from notebook_support import NotebookCheckpoints


class _Level0File:
    def __init__(self):
        self.calls = 0

    def get_acquisition_chunk_data(self, selected_chunk):
        self.calls += 1
        return np.full((2, 3), selected_chunk)


@dataclass
class _DopplerConfig:
    mode: str = "test"


class _DopplerEstimator:
    config = _DopplerConfig()
    prf_hz = 2.0

    def __init__(self):
        self.calls = 0

    def estimate_segments(self, segments, **kwargs):
        self.calls += 1

        class Scene:
            @staticmethod
            def alignment_summary():
                return [{"name": "test"}]

        return [self.calls], Scene()


class CheckpointTest(unittest.TestCase):
    def test_decode_policies_and_chunk_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.dat"
            source.write_bytes(b"sentinel")
            level0 = _Level0File()

            reuse = NotebookCheckpoints(root / "cache", verbose=False)
            first = reuse.decode(
                input_path=source, selected_chunk=13, level0_file=level0
            )
            second = reuse.decode(
                input_path=source, selected_chunk=13, level0_file=level0
            )
            chunk_14 = reuse.decode(
                input_path=source, selected_chunk=14, level0_file=level0
            )
            self.assertEqual(level0.calls, 2)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(chunk_14, np.full((2, 3), 14))

            refresh = NotebookCheckpoints(
                root / "cache", policy="refresh", verbose=False
            )
            refresh.decode(input_path=source, selected_chunk=13, level0_file=level0)
            self.assertEqual(level0.calls, 3)

            readonly = NotebookCheckpoints(
                root / "cache", policy="readonly", verbose=False
            )
            readonly.decode(input_path=source, selected_chunk=13, level0_file=level0)
            with self.assertRaises(FileNotFoundError):
                NotebookCheckpoints(
                    root / "cache", tag="missing", policy="readonly", verbose=False
                ).decode(input_path=source, selected_chunk=13, level0_file=level0)

            off = NotebookCheckpoints(root / "cache", policy="off", verbose=False)
            off.decode(input_path=source, selected_chunk=13, level0_file=level0)
            off.decode(input_path=source, selected_chunk=13, level0_file=level0)
            self.assertEqual(level0.calls, 5)

    def test_range_stage_uses_token_instead_of_hashing_arrays(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.dat"
            source.write_bytes(b"sentinel")
            checkpoints = NotebookCheckpoints(root / "cache", verbose=False)
            compressed = np.arange(6).reshape(2, 3)
            range_times = np.arange(3.0)
            arguments = {
                "input_path": source,
                "selected_chunk": 13,
                "level0_file": None,
                "raw_slant_range_times_s": range_times,
                "range_sample_frequency_hz": 1.0,
                "pulse_start_frequency_hz": 2.0,
                "pulse_ramp_rate_hz_per_s": 3.0,
                "pulse_length_s": 4.0,
                "swst_bias_s": 5.0,
                "range_reference_function": np.ones(3),
                "radar_data": np.ones((2, 3)),
                "iq_bias": 1 + 2j,
            }

            with patch(
                "notebook_support.checkpoints."
                "azimuth_pre_processing.range.compression.compress",
                return_value=(compressed, range_times),
            ) as compress:
                first = checkpoints.range_compression(**arguments)
                second = checkpoints.range_compression(
                    **{**arguments, "radar_data": np.zeros((2, 3))}
                )
                checkpoints.range_compression(
                    **{**arguments, "range_sample_frequency_hz": 2.0}
                )

            self.assertEqual(compress.call_count, 2)
            np.testing.assert_array_equal(first[0], second[0])

    def test_doppler_stage_reuses_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.dat"
            source.write_bytes(b"sentinel")
            checkpoints = NotebookCheckpoints(root / "cache", verbose=False)

            with patch(
                "notebook_support.checkpoints."
                "azimuth_pre_processing.range.compression.compress",
                return_value=(np.ones((2, 3)), np.arange(3.0)),
            ):
                for chunk in (13, 14):
                    checkpoints.range_compression(
                        input_path=source,
                        selected_chunk=chunk,
                        level0_file=None,
                        raw_slant_range_times_s=np.arange(3.0),
                        range_sample_frequency_hz=1.0,
                        pulse_start_frequency_hz=2.0,
                        pulse_ramp_rate_hz_per_s=3.0,
                        pulse_length_s=4.0,
                        swst_bias_s=5.0,
                        radar_data=np.ones((2, 3)),
                        iq_bias=0j,
                    )

            estimator = _DopplerEstimator()
            arguments = {
                "selected_chunks": (13, 14),
                "estimator": estimator,
                "segments": [object(), object()],
                "t0_s": 1.0,
                "slice_start_times_s": [2.0],
                "last_slice_stop_time_s": 3.0,
                "product_start_time_s": 4.0,
                "product_stop_time_s": 5.0,
                "zero_doppler_offset_s": 6.0,
            }
            first = checkpoints.doppler_estimation(**arguments)
            second = checkpoints.doppler_estimation(**arguments)

            self.assertEqual(estimator.calls, 1)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
