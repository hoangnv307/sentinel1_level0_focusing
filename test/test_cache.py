import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.cache import open_array, prune_old_entries, save_array


class CacheTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
