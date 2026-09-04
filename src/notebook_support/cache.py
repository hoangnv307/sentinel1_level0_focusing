"""Hỗ trợ lưu mảng lớn và dọn cache cho notebook marimo."""

from hashlib import sha256
import pickle
from pathlib import Path

import numpy as np


_FINGERPRINT_FILE = "fingerprint.sha256"


def chunk_cache_key(chunks) -> str:
    """Return one stable cache namespace for an unordered chunk collection."""
    values = sorted({int(chunk) for chunk in chunks})
    if len(values) < 2:
        raise ValueError("At least two distinct chunks are required.")
    return "chunks-" + "-".join(map(str, values))


def save_array(path: str | Path, array: np.ndarray) -> None:
    """Atomically save a large array without putting it in marimo's pickle."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(destination)


def open_array(path: str | Path) -> np.memmap:
    """Open a cached array without eagerly loading it into RAM."""
    return np.load(path, mmap_mode="r", allow_pickle=False)


def cache_fingerprint(*values: object) -> str:
    """Return a stable digest for cache inputs serializable by pickle."""
    return sha256(pickle.dumps(values, protocol=5)).hexdigest()


def array_cache_matches(
    cache_directory: str | Path,
    array_path: str | Path,
    fingerprint: str,
    expected_shape: tuple[int, ...],
) -> bool:
    """Check an external array and its stable input fingerprint."""
    try:
        if (Path(cache_directory) / _FINGERPRINT_FILE).read_text() != fingerprint:
            return False
        return open_array(array_path).shape == expected_shape
    except (OSError, ValueError, EOFError):
        return False


def save_cache_fingerprint(cache_directory: str | Path, fingerprint: str) -> None:
    """Atomically mark an external array as complete for these inputs."""
    destination = Path(cache_directory) / _FINGERPRINT_FILE
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(fingerprint)
    temporary.replace(destination)


def invalidate_broken_array_cache(
    cache_directory: str | Path,
    array_path: str | Path,
    expected_shape: tuple[int, ...] | None = None,
) -> int:
    """Remove persistent metadata when its external array is unusable."""
    try:
        array = open_array(array_path)
        valid = expected_shape is None or array.shape == expected_shape
    except (OSError, ValueError, EOFError):
        valid = False
    if valid:
        return 0

    removed = 0
    for entry in Path(cache_directory).glob("*.pickle"):
        entry.unlink()
        removed += 1
    return removed


def prune_old_entries(cache_root: str | Path) -> int:
    """Keep only the newest persistent value for each named cache block."""
    removed = 0
    root = Path(cache_root)
    if not root.exists():
        return removed

    # ponytail: single-notebook cleanup; add locking only for concurrent writers.
    for stage in root.iterdir():
        if not stage.is_dir():
            continue
        entries = sorted(
            stage.glob("*.pickle"), key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for old_entry in entries[1:]:
            old_entry.unlink()
            removed += 1
    return removed
