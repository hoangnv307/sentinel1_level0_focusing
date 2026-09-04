import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from notebook_support.cache import (
    array_cache_matches,
    cache_fingerprint,
    chunk_cache_key,
    invalidate_broken_array_cache,
    open_array,
    prune_old_entries,
    save_cache_fingerprint,
    save_array,
)


class CacheTest(unittest.TestCase):
    def test_chunk_cache_key_is_order_independent(self):
        self.assertEqual(chunk_cache_key((14, 13)), "chunks-13-14")
        with self.assertRaises(ValueError):
            chunk_cache_key((13, 13))

    def test_large_array_cache_is_memory_mapped(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "array.npy"
            expected = np.arange(12, dtype=np.complex64).reshape(3, 4)

            save_array(path, expected)
            actual = open_array(path)

            self.assertIsInstance(actual, np.memmap)
            np.testing.assert_array_equal(actual, expected)

    def test_prune_old_entries_keeps_newest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stage = root / "doppler-centroid"
            stage.mkdir()
            old = stage / "old.pickle"
            newest = stage / "new.pickle"
            old.write_bytes(b"old")
            newest.write_bytes(b"new")
            old_mtime = old.stat().st_mtime_ns - 1_000_000
            os.utime(old, ns=(old_mtime, old_mtime))

            self.assertEqual(prune_old_entries(root), 1)
            self.assertFalse(old.exists())
            self.assertEqual(newest.read_bytes(), b"new")

    def test_invalid_array_removes_persistent_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stage = Path(temporary_directory)
            metadata = stage / "value.pickle"
            metadata.write_bytes(b"cached")

            self.assertEqual(
                invalidate_broken_array_cache(
                    stage, stage / "missing.npy", expected_shape=(3, 4)
                ),
                1,
            )
            self.assertFalse(metadata.exists())

            broken = stage / "broken.npy"
            broken.write_bytes(b"not a numpy array")
            metadata.write_bytes(b"cached")
            self.assertEqual(
                invalidate_broken_array_cache(stage, broken),
                1,
            )
            self.assertFalse(metadata.exists())

    def test_array_cache_requires_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stage = Path(temporary_directory)
            array = stage / "data.npy"
            save_array(array, np.zeros((2, 3), dtype=np.complex64))
            fingerprint = cache_fingerprint("input", 1)
            save_cache_fingerprint(stage, fingerprint)

            self.assertTrue(
                array_cache_matches(stage, array, fingerprint, (2, 3))
            )
            self.assertFalse(
                array_cache_matches(
                    stage, array, cache_fingerprint("input", 2), (2, 3)
                )
            )


if __name__ == "__main__":
    unittest.main()
