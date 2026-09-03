# AGENTS.md — sentinel1_level0_decoder_demo

Sentinel-1 Level-0 → focused SLC (Stripmap/S6) processing demo, derived from
[Rich Hall's sentinel1decoder](https://github.com/Rich-Hall/sentinel1decoder).

## Overview

The repo contains **marimo notebooks** rendered as Python files under
`workflows/sentinel-1/`, backed by pure-processing modules
under `src/sentinel1_processing/`. The notebook decodes raw Level-0 I/Q data,
corrects I/Q bias, range-compresses, estimates Doppler centroid, and focuses an
SLC image. Expensive stages are cached by `mo.persistent_cache`.

All user-facing docs/commits are in Vietnamese.

## Commands

- **Install**: `pip install -r requirements-lock.txt && pip install -e . --no-deps`
- **Open notebook**: `marimo edit workflows/sentinel-1/focus_chunk_13.py`
- **Run tests** (unittest — pytest is *not* installed):
  ```bash
  MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python -m unittest discover -s test -p "test_*.py"
  ```
  (`MPLBACKEND=Agg` is required on headless boxes; a plain run errors with a
  Tk "couldn't connect to display" — an environment issue, not a code failure.)
- **Check notebooks**: `marimo check workflows/sentinel-1/*.py`
- The project interpreter lives at `/home/xiaoxin/python_envs/sentinel1/bin/python` (repo is not under that env's site-packages path).

## Layout

- `workflows/sentinel-1/` — Sentinel-1 marimo workflows (cells as `@app.cell`).
- `workflows/radarsat-1/` — RADARSAT-1 command-line workflows.
- `src/sentinel1_processing/` — pure processing package:
  - `raw_data_correction.py` — I/Q bias estimation.
  - `range_processing/` — `reference_function`, `swst_bias`, `dependent_gain`.
  - `azimuth_pre_processing/range/compression.py` — range compression
    (`compress`, `zero_pad`, `forward_fft`, `multiply_reference_function`,
    `inverse_fft`, `extract_valid_samples`).
  - `azimuth_pre_processing/` — zero padding and forward FFT.
  - `azimuth_processing/` — focus pipeline: RCMC, secondary range compression,
    azimuth compression, processing blocks.
  - `doppler_centroid_estimation.py` (~2k lines) — Doppler centroid estimation.
  - `effective_velocity.py` — effective velocity model.
  - `dce_plotting.py` — DCE diagnostic plots.
- `src/notebook_support/cache.py` — large-array storage and stale-cache cleanup helpers.
- `test/` — unittest suite for Sentinel-1, RADARSAT-1, cache, and effective velocity.
- `data/`, `output/` — raw input / produced output (gitignored, see `.gitignore`).
- `.cache/sentinel1/` — marimo persistent cache (gitignored).

## Notebook conventions

- Cells are `@app.cell`; each cell declares its global deps via its `def _(...)` params.
- **Separate computation from display**: heavy compute goes in its own cell, then a
  separate display cell (`print`, `md`, `plt`) reads its result. Do NOT put
  `print()`/plots inside a `mo.persistent_cache` block — they're skipped on cache hits.
- Cache blocks wrap only data-producing computation:
  ```python
  @app.cell
  def _(CACHE_ROOT, mo, ...):
      with mo.persistent_cache(name=..., save_path=CACHE_ROOT, pin_modules=True):
          ...
  ```
- Keep numeric/array constants (sample rates, pulse params) as named vars in early cells.

## Gotchas (marimo `persistent_cache`)

- **Don't pass Python `complex` scalars into values hashed by `persistent_cache`.** marimo's
  content hasher (`primitive_to_bytes`) can't serialize a plain `complex`:
  `TypeError: cannot convert 'complex' object to bytes` at `__enter__`, before the block runs.
- Use a numpy complex instead — `np.complex128(real, imag)` is a *data primitive* and hashes fine.
  Example: `iq_bias=np.complex128(*iq_bias_components)` instead of `complex(*...)`.
  (Add `np` to the cell's `def _(...)` params if not already a dep.)
- **A Python `complex` can also leak into notebook scope and poison a downstream cache's hash.** If a
  cache block (or a dependent display cell) returns/exposes a bare `complex` (e.g. from
  `estimate_iq_bias()`), any later `persistent_cache` block that transitively depends on it will
  fail at `__enter__` with the same `cannot convert 'complex' object to bytes`. Wrap the scalar as
  `np.complex128(value)` before it leaves the block, so it's hashed as a data primitive.
- The notebook auto-clears stale cache entries for a stage after the later Focus stage completes;
  to recompute everything, delete `.cache/sentinel1/` and restart the notebook.
- **Don't `del` a variable bound inside a `persistent_cache` block.** marimo saves the block's defs
  at `__exit__` by reading them back out of `f_locals`; deleting one makes marimo raise
  `CacheException: Cache expected a reference to a variable that is not present (...)`. Leave the
  bound name in scope (it's still returned/usable afterward).

## Testing expectations

- Tests use `unittest` classes (not pytest fixtures). 17 tests total, currently passing
  with `MPLBACKEND=Agg`.
- New pure-processing functions should get a `test_*` unittest; keep `np.testing.assert_*` for array comparisons.
