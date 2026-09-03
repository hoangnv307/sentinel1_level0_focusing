"""Hỗ trợ lưu mảng lớn và dọn cache cho notebook marimo."""

from pathlib import Path

import numpy as np


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
