"""
sentinel1_processing.dce
================

Research-oriented Sentinel-1 Doppler Centroid Estimation (DCE) utilities.

Basis
-----
ESA Sentinel-1 Level-1 Detailed Algorithm Definition (DAD), Section 5:
- 5.2.2 Correlation Doppler Centroid Estimator (CDCE)
- 5.3 Fine DC Estimates Unwrapping
- 5.4 Absolute DC Estimation
- 5.5 Polynomial Fitting / quality measurement
- 5.6 Processing Block Dimensions

Important distinction
---------------------
The DAD publishes the DCE algorithm, but several DCE block dimensions and
spacings are INTERNAL/configurable IPF parameters.

The ``DCEConfig.s6_research()`` profile therefore contains two kinds of values:

DAD / product-supported:
- quadratic DC polynomial for single-swath processing
- CDCE lag-1 estimator
- configurable azimuth/range block dimensions
- configurable number of range blocks
- RMS quality metric
- 20-Hz RMS threshold from the supplied AUX_PP1
- 20 fine range estimates observed in the supplied S6 product

[Inference] Reverse-engineered for the supplied S6 scene:
- azimuth DCE block length = 6000 lines
- DCE block starts follow the global sliced-product timeline scheduler
- nominal range block size = 1000 samples as a research starting point.
  The exact IPF internal range-block configuration is not public.

The module separates these assumptions from the core DAD algorithms so that
they can be changed without touching the estimator itself.
"""

from __future__ import annotations

__version__ = "2.2.0-slice-timeline"
# BUILD_MARKER: MULTISEGMENT_SWST_ALIGNMENT_2026_08_21

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Optional, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float]
GeometryDcProvider = Callable[[float, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DCEConfig:
    """Configuration for Sentinel-1 DCE processing.

    Parameters marked as research/inferred are deliberately explicit so the
    notebook can change them easily when more IPF behavior is reverse-engineered.
    """

    # DAD §5.6: configurable azimuth-block size.
    azimuth_block_size_lines: int

    # DAD §5.6: number of range blocks is configurable.
    num_range_blocks: int

    # DAD §5.6: range-block size is configurable. If None, the available
    # range extent is split into approximately equal blocks.
    range_block_size_samples: Optional[int] = None

    # Optional usable range ROI [start, stop). None means the complete supplied
    # range-compressed support.
    range_roi_start: int = 0
    range_roi_stop: Optional[int] = None

    # How DCE azimuth blocks are placed.
    # "spacing"  : regular configurable spacing, with tail coverage.
    # "slice_timeline": [Inference] global Sentinel-1 sliced-product schedule.
    # "edge_mid_edge": legacy single-product research placement.
    # "custom"   : use explicit starts passed to build_azimuth_blocks().
    azimuth_placement: Literal[
        "spacing", "slice_timeline", "edge_mid_edge", "custom"
    ] = "spacing"

    # DAD §5.6: azimuth spacing between DC estimates is configurable.
    # Required for azimuth_placement="spacing".
    azimuth_spacing_lines: Optional[int] = None

    # DAD §5.3: FFT length used by the robust fine-DC unwrapping algorithm
    # is an internal/configurable parameter.
    unwrap_fft_length: int = 4096

    # Single-swath DC polynomial in DAD §5.5 is quadratic.
    polynomial_degree: int = 2

    # AUX_PP1 supplied with the project.
    rms_threshold_hz: float = 20.0

    # DAD says outliers are detected/eliminated in the LS fit, but the exact
    # operational rejection implementation is not exposed in the DAD.
    # This is a transparent research implementation choice.
    outlier_sigma: float = 3.5
    max_fit_iterations: int = 5

    # Optional weighting in the DAD robust unwrap. Uniform is closest to the
    # published unweighted expression. "coherence" is a research extension.
    unwrap_weighting: Literal["uniform", "coherence"] = "uniform"

    @classmethod
    def s6_research(cls) -> "DCEConfig":
        """Research profile inferred from the supplied Sentinel-1 S6 scene.

        [Inference]
        - 6000-line DCE azimuth blocks are directly recovered from the
          fineDceAzimuthStart/Stop duration and PRF.
        - DCE block starts follow the global sliced-product timeline rule.
        - 1250 range samples minimizes the mean annotation RMSE in the supplied
          S6 scene among the tested 750/1000/1250/1500-sample layouts. It is
          NOT claimed as a published IPF parameter.
        """
        return cls(
            azimuth_block_size_lines=6000,
            num_range_blocks=20,
            range_block_size_samples=1250,
            azimuth_placement="slice_timeline",
            azimuth_spacing_lines=None,
            unwrap_fft_length=4096,
            polynomial_degree=2,
            rms_threshold_hz=20.0,
            outlier_sigma=3.0,
        )


@dataclass(frozen=True)
class DCESegment:
    """One contiguous range-compressed acquisition segment/chunk.

    A segment may have its own SWST and therefore its own fast-time/range grid.
    ``range_compressed`` must have shape ``(azimuth_lines, range_samples)``.
    """

    range_compressed: np.ndarray
    slant_range_times_s: ArrayLike
    azimuth_times_s: ArrayLike
    name: str = "segment"

    def __post_init__(self) -> None:
        data = np.asarray(self.range_compressed)
        tau = np.asarray(self.slant_range_times_s, dtype=np.float64)
        eta = np.asarray(self.azimuth_times_s, dtype=np.float64)

        if data.ndim != 2:
            raise ValueError(
                f"{self.name}: range_compressed must have shape (azimuth, range)."
            )
        if tau.ndim != 1 or tau.size != data.shape[1]:
            raise ValueError(
                f"{self.name}: slant_range_times_s must have one value per range sample."
            )
        if eta.ndim != 1 or eta.size != data.shape[0]:
            raise ValueError(
                f"{self.name}: azimuth_times_s must have one value per azimuth line."
            )
        if tau.size < 2 or np.any(np.diff(tau) <= 0):
            raise ValueError(f"{self.name}: slant_range_times_s must be increasing.")
        if eta.size < 1 or (eta.size > 1 and np.any(np.diff(eta) <= 0)):
            raise ValueError(f"{self.name}: azimuth_times_s must be increasing.")

    @property
    def num_azimuth_lines(self) -> int:
        return int(np.asarray(self.range_compressed).shape[0])

    @property
    def num_range_samples(self) -> int:
        return int(np.asarray(self.range_compressed).shape[1])


@dataclass(frozen=True)
class SegmentAlignment:
    """Prepared mapping from one segment range grid to the common DCE grid."""

    segment: DCESegment
    global_start_line: int
    global_stop_line: int
    source_start_index: float
    integer_shift_samples: int
    fractional_shift_samples: float
    source_range_spacing_s: float


@dataclass
class PreparedSegmentScene:
    """Prepared multi-segment scene used by :meth:`estimate_segments`.

    The object stores only alignment metadata and the common range grid.  It
    does *not* materialize a huge fully aligned 2-D scene in memory.
    """

    alignments: list[SegmentAlignment]
    common_slant_range_times_s: np.ndarray
    azimuth_times_s: np.ndarray
    range_spacing_s: float
    nominal_azimuth_spacing_s: float
    azimuth_gap_tolerance_s: float
    lanczos_radius: int = 8

    @property
    def num_azimuth_lines(self) -> int:
        return int(self.azimuth_times_s.size)

    @property
    def num_range_samples(self) -> int:
        return int(self.common_slant_range_times_s.size)

    def alignment_summary(self) -> list[dict]:
        """Return notebook-friendly range-alignment diagnostics."""
        out: list[dict] = []
        for a in self.alignments:
            out.append(
                {
                    "name": a.segment.name,
                    "global_start_line": a.global_start_line,
                    "global_stop_line": a.global_stop_line,
                    "azimuth_lines": a.segment.num_azimuth_lines,
                    "native_range_samples": a.segment.num_range_samples,
                    "common_range_samples": self.num_range_samples,
                    "native_range_start_s": float(np.asarray(a.segment.slant_range_times_s)[0]),
                    "common_range_start_s": float(self.common_slant_range_times_s[0]),
                    "source_start_index": a.source_start_index,
                    "integer_shift_samples": a.integer_shift_samples,
                    "fractional_part_samples": a.fractional_shift_samples,
                    "range_spacing_s": a.source_range_spacing_s,
                }
            )
        return out


@dataclass(frozen=True)
class AzimuthDCEBlock:
    """One DCE azimuth processing block."""

    start_line: int
    stop_line: int  # exclusive
    center_line: float

    start_time_s: float
    stop_time_s: float  # right-hand block boundary
    azimuth_time_s: float

    @property
    def num_lines(self) -> int:
        return self.stop_line - self.start_line


@dataclass(frozen=True)
class RangeDCEBlock:
    """One DCE range processing block."""

    start_sample: int
    stop_sample: int  # exclusive
    center_sample: float
    center_slant_range_time_s: float

    @property
    def num_samples(self) -> int:
        return self.stop_sample - self.start_sample


@dataclass
class DCERecord:
    """DCE result for one azimuth block."""

    block: AzimuthDCEBlock
    t0_s: float

    range_blocks: list[RangeDCEBlock]

    # Fine/baseband DCE from CDCE, within the PRF ambiguity interval.
    fine_baseband_hz: np.ndarray

    # Fine DCE after DAD §5.3 range unwrapping.
    fine_unwrapped_hz: np.ndarray

    # Absolute estimates after geometry ambiguity resolution when available.
    fine_absolute_hz: np.ndarray

    # Polynomial coefficients are stored in ASCENDING order:
    # f(tau) = c0 + c1*(tau-t0) + c2*(tau-t0)^2 + ...
    data_dc_polynomial: np.ndarray

    geometry_dc_polynomial: Optional[np.ndarray] = None

    # Diagnostics
    accc: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex128))
    coherence: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    valid_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

    rms_error_hz: float = np.nan
    rms_above_threshold: bool = False

    # Whether the integer PRF ambiguity was resolved from geometry.
    ambiguity_number: Optional[int] = None
    absolute_ambiguity_resolved: bool = False

    @property
    def range_times_s(self) -> np.ndarray:
        return np.asarray(
            [b.center_slant_range_time_s for b in self.range_blocks],
            dtype=np.float64,
        )

    def evaluate(self, slant_range_time_s: ArrayLike) -> np.ndarray:
        tau = np.asarray(slant_range_time_s, dtype=np.float64)
        x = tau - self.t0_s
        return np.polynomial.polynomial.polyval(x, self.data_dc_polynomial)


# ---------------------------------------------------------------------------
# Block layout
# ---------------------------------------------------------------------------

def build_azimuth_blocks(
    n_lines: int,
    prf_hz: float,
    config: DCEConfig,
    azimuth_times_s: Optional[ArrayLike] = None,
    *,
    azimuth_time_offset_s: float = 0.0,
    custom_starts: Optional[Sequence[int]] = None,
    slice_start_times_s: Optional[Sequence[float]] = None,
    last_slice_stop_time_s: Optional[float] = None,
    product_start_time_s: Optional[float] = None,
    product_stop_time_s: Optional[float] = None,
    zero_dop_minus_acq_time_s: Optional[float] = None,
) -> list[AzimuthDCEBlock]:
    """Build DCE azimuth blocks.

    DAD §5.6 states that block size and estimate spacing are configurable.

    If ``azimuth_times_s`` is supplied, it is interpreted as the time attached
    to each input line. ``azimuth_time_offset_s`` can convert packet/acquisition
    time to the desired DCE time convention (e.g. zero-Doppler time).

    ``azimuth_placement="slice_timeline"`` uses the global schedule inferred
    from Sentinel-1 Stripmap sliced products: each slice start, each midpoint
    between consecutive slice starts, and for the final slice its midpoint and
    final complete DCE block. Records are retained by zero-Doppler product time.
    """
    if n_lines <= 0:
        raise ValueError("n_lines must be positive.")
    if prf_hz <= 0:
        raise ValueError("prf_hz must be positive.")

    B = int(config.azimuth_block_size_lines)
    if B <= 1 or B > n_lines:
        raise ValueError(
            f"azimuth_block_size_lines={B} must satisfy 1 < B <= n_lines={n_lines}."
        )

    line_times = None
    if azimuth_times_s is not None:
        line_times = np.asarray(azimuth_times_s, dtype=np.float64)
        if line_times.ndim != 1 or line_times.size != n_lines:
            raise ValueError(
                "azimuth_times_s must be a 1-D array with one value per input line."
            )
        if np.any(np.diff(line_times) <= 0):
            raise ValueError("azimuth_times_s must be strictly increasing.")

    placement = config.azimuth_placement

    if placement == "slice_timeline":
        if line_times is None:
            raise ValueError(
                "azimuth_times_s is required for placement='slice_timeline'."
            )
        if (
            slice_start_times_s is None
            or last_slice_stop_time_s is None
            or product_start_time_s is None
            or product_stop_time_s is None
            or zero_dop_minus_acq_time_s is None
        ):
            raise ValueError(
                "slice_timeline placement requires slice start/stop, product "
                "start/stop, and zero-Doppler minus acquisition time."
            )

        slice_starts = np.asarray(slice_start_times_s, dtype=np.float64)
        if slice_starts.ndim != 1 or slice_starts.size == 0:
            raise ValueError("slice_start_times_s must not be empty.")
        if np.any(np.diff(slice_starts) <= 0):
            raise ValueError("slice_start_times_s must be strictly increasing.")

        last_stop = float(last_slice_stop_time_s)
        product_start = float(product_start_time_s)
        product_stop = float(product_stop_time_s)
        zd_offset = float(zero_dop_minus_acq_time_s)
        block_duration = B / float(prf_hz)

        if last_stop - float(slice_starts[-1]) < block_duration:
            raise ValueError("The final slice is shorter than one DCE block.")
        if product_stop < product_start:
            raise ValueError("product_stop_time_s must not precede product_start_time_s.")

        global_starts: list[float] = []
        for t0, t1 in zip(slice_starts[:-1], slice_starts[1:]):
            global_starts.extend((float(t0), 0.5 * float(t0 + t1)))
        global_starts.extend((
            float(slice_starts[-1]),
            0.5 * float(slice_starts[-1] + last_stop),
            last_stop - block_duration,
        ))

        unique_starts: list[float] = []
        for value in sorted(global_starts):
            if not unique_starts or abs(value - unique_starts[-1]) > 1e-9:
                unique_starts.append(value)

        starts = []
        for start_time in unique_starts:
            center_zd = start_time + 0.5 * block_duration + zd_offset
            if product_start <= center_zd <= product_stop:
                start = int(np.argmin(np.abs(line_times - start_time)))
                if start + B <= n_lines:
                    starts.append(start)
        starts = sorted(set(starts))
        output_time_offset = zd_offset

    elif placement == "edge_mid_edge":
        starts = [0, n_lines // 2, n_lines - B]
        output_time_offset = float(azimuth_time_offset_s)

    elif placement == "spacing":
        step = config.azimuth_spacing_lines
        if step is None or step <= 0:
            raise ValueError(
                "azimuth_spacing_lines must be positive for placement='spacing'."
            )
        last_start = n_lines - B
        starts = list(range(0, last_start + 1, int(step)))
        if starts[-1] != last_start:
            # Ensure the last DCE block covers the end of the supplied
            # internal signal-data extent.
            starts.append(last_start)
        output_time_offset = float(azimuth_time_offset_s)

    elif placement == "custom":
        if not custom_starts:
            raise ValueError(
                "custom_starts must be supplied for placement='custom'."
            )
        starts = [int(s) for s in custom_starts]
        output_time_offset = float(azimuth_time_offset_s)

    else:
        raise ValueError(f"Unsupported azimuth placement: {placement}")

    if placement != "slice_timeline":
        last_start = n_lines - B
        starts = sorted({min(max(0, int(s)), last_start) for s in starts})

    blocks: list[AzimuthDCEBlock] = []

    for start in starts:
        stop = start + B

        if line_times is None:
            t_start = start / prf_hz + output_time_offset
            t_stop = stop / prf_hz + output_time_offset
        else:
            t_start = float(line_times[start] + output_time_offset)
            if stop < n_lines:
                # Right-hand boundary is exactly the next line time.
                t_stop = float(line_times[stop] + output_time_offset)
            else:
                # Extrapolate one PRI after the final line.
                t_stop = float(
                    line_times[-1] + 1.0 / prf_hz + output_time_offset
                )

        # Product observations are consistent with DCE azimuthTime being the
        # midpoint of fineDceAzimuthStartTime/fineDceAzimuthStopTime.
        t_center = 0.5 * (t_start + t_stop)

        blocks.append(
            AzimuthDCEBlock(
                start_line=start,
                stop_line=stop,
                center_line=start + 0.5 * B,
                start_time_s=t_start,
                stop_time_s=t_stop,
                azimuth_time_s=t_center,
            )
        )

    return blocks


def build_range_blocks(
    n_range_samples: int,
    slant_range_times_s: ArrayLike,
    config: DCEConfig,
) -> list[RangeDCEBlock]:
    """Build DCE range blocks according to DAD §5.6.

    The DAD allows overlap and configures both block size and number of blocks.
    If a block size is specified, block starts are distributed uniformly over
    the requested range ROI so that exactly ``num_range_blocks`` estimates are
    generated.

    If no block size is supplied, the ROI is split approximately evenly.
    """
    tau = np.asarray(slant_range_times_s, dtype=np.float64)

    if tau.ndim != 1 or tau.size != n_range_samples:
        raise ValueError(
            "slant_range_times_s must be 1-D with one value per range sample."
        )
    if np.any(np.diff(tau) <= 0):
        raise ValueError("slant_range_times_s must be strictly increasing.")

    roi0 = int(config.range_roi_start)
    roi1 = (
        n_range_samples
        if config.range_roi_stop is None
        else int(config.range_roi_stop)
    )
    roi0 = max(0, roi0)
    roi1 = min(n_range_samples, roi1)

    if roi1 <= roi0:
        raise ValueError("Invalid range ROI.")

    n_blocks = int(config.num_range_blocks)
    if n_blocks <= 0:
        raise ValueError("num_range_blocks must be positive.")

    roi_len = roi1 - roi0

    if config.range_block_size_samples is None:
        # Equal, non-overlapping blocks with all samples covered.
        edges = np.linspace(roi0, roi1, n_blocks + 1)
        edges = np.rint(edges).astype(int)

        starts = edges[:-1]
        stops = edges[1:]

    else:
        B = int(config.range_block_size_samples)
        if B <= 0 or B > roi_len:
            raise ValueError(
                f"range_block_size_samples={B} must satisfy 0 < B <= ROI length {roi_len}."
            )

        first_start = roi0
        last_start = roi1 - B

        if n_blocks == 1:
            starts = np.array(
                [int(round(0.5 * (first_start + last_start)))],
                dtype=int,
            )
        else:
            starts = np.rint(
                np.linspace(first_start, last_start, n_blocks)
            ).astype(int)

        stops = starts + B

    dtau = float(np.median(np.diff(tau)))

    blocks: list[RangeDCEBlock] = []
    for start, stop in zip(starts, stops):
        start = int(start)
        stop = int(stop)

        if stop <= start:
            continue

        # Boundary-center convention. For a B-sample block beginning at tau[start],
        # the block centre is tau[start] + B/2 * dtau. This matches the supplied
        # S6 annotation more closely than a (B-1)/2 sample-centre convention.
        center_sample = start + 0.5 * (stop - start)
        center_tau = float(tau[start] + 0.5 * (stop - start) * dtau)

        blocks.append(
            RangeDCEBlock(
                start_sample=start,
                stop_sample=stop,
                center_sample=center_sample,
                center_slant_range_time_s=center_tau,
            )
        )

    if len(blocks) != n_blocks:
        raise RuntimeError(
            f"Expected {n_blocks} range blocks, generated {len(blocks)}."
        )

    return blocks


# ---------------------------------------------------------------------------
# DAD §5.2.2 -- Correlation Doppler Centroid Estimator
# ---------------------------------------------------------------------------

def lag1_accc(range_compressed_block: np.ndarray) -> np.ndarray:
    """Calculate the lag-one ACCC vector along azimuth.

    For complex range-compressed data s[eta, r]:

        c[r] = sum_eta s[eta, r] * conj(s[eta+1, r])

    This sign convention is consistent with DAD Eq. (5-19):

        f_c = -PRF/(2*pi) * angle(c)
    """
    s = np.asarray(range_compressed_block)

    if s.ndim != 2:
        raise ValueError(
            "range_compressed_block must have shape (azimuth, range)."
        )
    if s.shape[0] < 2:
        raise ValueError("At least two azimuth lines are required.")

    # Complex128 accumulation reduces numerical error for large DCE blocks.
    return np.sum(
        s[:-1, :] * np.conj(s[1:, :]),
        axis=0,
        dtype=np.complex128,
    )


def _range_block_coherence(
    s: np.ndarray,
    start: int,
    stop: int,
) -> float:
    """Normalized lag-one coherence for diagnostics / optional weighting."""
    x = s[:-1, start:stop]
    y = s[1:, start:stop]

    num = np.abs(np.sum(x * np.conj(y), dtype=np.complex128))
    p0 = float(np.sum(np.abs(x) ** 2, dtype=np.float64))
    p1 = float(np.sum(np.abs(y) ** 2, dtype=np.float64))

    den = np.sqrt(max(p0 * p1, 0.0))
    if den == 0.0:
        return 0.0

    return float(np.clip(num / den, 0.0, 1.0))


def fine_dce_cdce(
    range_compressed_block: np.ndarray,
    range_blocks: Sequence[RangeDCEBlock],
    prf_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DAD §5.2.2 CDCE fine Doppler estimation.

    Returns
    -------
    fine_hz:
        Baseband / ambiguous DC estimate for every range block.
    accc_blocks:
        Complex ACCC averaged over each range block.
    coherence:
        Normalized lag-one coherence. This is provided as a diagnostic; it is
        not claimed to be the operational IPF quality metric.
    """
    if prf_hz <= 0:
        raise ValueError("prf_hz must be positive.")

    s = np.asarray(range_compressed_block)
    c_range = lag1_accc(s)

    c_blocks = np.empty(len(range_blocks), dtype=np.complex128)
    coherence = np.empty(len(range_blocks), dtype=np.float64)

    for i, rb in enumerate(range_blocks):
        r0, r1 = rb.start_sample, rb.stop_sample

        if r0 < 0 or r1 > s.shape[1] or r1 <= r0:
            raise ValueError(f"Invalid range block {i}: [{r0}, {r1}).")

        # DAD: divide ACCC into range sub-vectors and average each sub-vector.
        c_blocks[i] = np.mean(c_range[r0:r1])
        coherence[i] = _range_block_coherence(s, r0, r1)

    phase = np.angle(c_blocks)

    # DAD Eq. (5-19)
    fine_hz = -(float(prf_hz) / (2.0 * np.pi)) * phase

    return fine_hz, c_blocks, coherence


# ---------------------------------------------------------------------------
# Multi-segment range-grid alignment
# ---------------------------------------------------------------------------

def prepare_dce_segments(
    segments: Sequence[DCESegment],
    prf_hz: float,
    *,
    lanczos_radius: int = 8,
    range_spacing_rtol: float = 1e-6,
    azimuth_gap_tolerance_s: Optional[float] = None,
) -> PreparedSegmentScene:
    """Prepare acquisition segments having different SWST/range lengths.

    The first chronological segment supplies the fixed reference fast-time
    grid. Each segment is mapped to this grid by a constant fractional sample
    shift. Sentinel-1 chunks of the same acquisition are expected to use the
    same range sampling frequency; if not, this routine raises rather than
    silently resampling a sample-rate change.

    For the S6 case discussed in the notebook, this exposes the ~81.25-sample
    chunk-14 offset directly through ``alignment_summary()``.
    """
    if not segments:
        raise ValueError("segments must not be empty.")
    if prf_hz <= 0:
        raise ValueError("prf_hz must be positive.")
    if lanczos_radius < 2:
        raise ValueError("lanczos_radius must be >= 2.")

    segs = sorted(segments, key=lambda x: float(np.asarray(x.azimuth_times_s)[0]))

    dts = []
    for seg in segs:
        tau = np.asarray(seg.slant_range_times_s, dtype=np.float64)
        dts.append(float(np.median(np.diff(tau))))

    dt = float(np.median(dts))
    if dt <= 0:
        raise ValueError("Invalid range sample spacing.")

    for seg, dt_i in zip(segs, dts):
        if not np.isclose(dt_i, dt, rtol=range_spacing_rtol, atol=0.0):
            raise ValueError(
                f"{seg.name}: range sample spacing differs from the common spacing "
                f"({dt_i:.12e} s vs {dt:.12e} s). Sample-rate resampling is not "
                "implemented; this API currently handles SWST/grid-offset changes."
            )

    common_tau = np.asarray(segs[0].slant_range_times_s, dtype=np.float64).copy()
    common_start = float(common_tau[0])

    alignments: list[SegmentAlignment] = []
    az_times: list[np.ndarray] = []
    global_line = 0

    previous_last_time: Optional[float] = None
    nominal_pri = 1.0 / float(prf_hz)
    gap_tolerance = (
        0.05 * nominal_pri
        if azimuth_gap_tolerance_s is None
        else float(azimuth_gap_tolerance_s)
    )
    if gap_tolerance < 0:
        raise ValueError("azimuth_gap_tolerance_s must be non-negative.")

    for seg, dt_i in zip(segs, dts):
        eta = np.asarray(seg.azimuth_times_s, dtype=np.float64)
        tau = np.asarray(seg.slant_range_times_s, dtype=np.float64)

        if previous_last_time is not None:
            gap = float(eta[0] - previous_last_time)
            # Permit small packet-time quantization differences, but reject
            # reordered/overlapping segment time axes.
            if gap <= 0:
                raise ValueError(
                    f"{seg.name}: azimuth times overlap or are not ordered."
                )
        previous_last_time = float(eta[-1])

        x0 = (common_start - float(tau[0])) / dt_i
        base = int(np.floor(x0 + 1e-12))
        frac = float(x0 - base)
        if frac < 0 and abs(frac) < 1e-10:
            frac = 0.0
        if frac >= 1.0 - 1e-10:
            base += 1
            frac = 0.0

        start_line = global_line
        stop_line = start_line + seg.num_azimuth_lines

        alignments.append(
            SegmentAlignment(
                segment=seg,
                global_start_line=start_line,
                global_stop_line=stop_line,
                source_start_index=float(x0),
                integer_shift_samples=base,
                fractional_shift_samples=frac,
                source_range_spacing_s=dt_i,
            )
        )
        az_times.append(eta)
        global_line = stop_line

    all_eta = np.concatenate(az_times)
    if all_eta.size > 1 and np.any(np.diff(all_eta) <= 0):
        raise ValueError("Concatenated segment azimuth times are not strictly increasing.")

    return PreparedSegmentScene(
        alignments=alignments,
        common_slant_range_times_s=common_tau,
        azimuth_times_s=all_eta,
        range_spacing_s=dt,
        nominal_azimuth_spacing_s=nominal_pri,
        azimuth_gap_tolerance_s=gap_tolerance,
        lanczos_radius=int(lanczos_radius),
    )


def _lanczos_fractional_weights(frac: float, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return integer tap offsets and normalized Lanczos-sinc weights."""
    offsets = np.arange(-radius, radius + 1, dtype=int)
    d = frac - offsets.astype(np.float64)
    weights = np.sinc(d) * np.sinc(d / (radius + 1.0))
    weights[np.abs(d) >= radius + 1.0] = 0.0
    sw = float(np.sum(weights))
    if abs(sw) < 1e-15:
        raise RuntimeError("Degenerate fractional-delay kernel.")
    weights /= sw
    return offsets, weights


def _align_segment_rows(
    alignment: SegmentAlignment,
    local_start: int,
    local_stop: int,
    n_common: int,
    *,
    lanczos_radius: int,
) -> np.ndarray:
    """Align a small row batch to the common range grid.

    The implementation is optimized for the Sentinel-1 case where chunks have
    the same range sampling frequency and differ by a constant SWST offset.
    Only the requested azimuth rows are materialized.
    """
    src = np.asarray(alignment.segment.range_compressed[local_start:local_stop, :])
    if src.ndim != 2:
        raise ValueError("Segment data slice must remain 2-D.")

    base = alignment.integer_shift_samples
    frac = alignment.fractional_shift_samples

    out = np.zeros(
        (src.shape[0], n_common),
        dtype=np.result_type(src.dtype, np.complex128),
    )

    # Integer alignment needs no interpolation and is exact. Samples outside
    # the native support remain zero and therefore do not contribute to ACCC.
    if abs(frac) < 1e-12:
        n0 = max(0, -base)
        n1 = min(n_common, src.shape[1] - base)
        if n1 > n0:
            out[:, n0:n1] = src[:, base + n0:base + n1]
        return out

    offsets, weights = _lanczos_fractional_weights(frac, lanczos_radius)
    n0 = max(0, -base - int(offsets[0]))
    n1 = min(n_common, src.shape[1] - base - int(offsets[-1]))
    if n1 <= n0:
        return out

    for m, w in zip(offsets, weights):
        i0 = base + int(m) + n0
        i1 = i0 + (n1 - n0)
        out[:, n0:n1] += w * src[:, i0:i1]

    return out


def _stream_accc_from_segments(
    prepared: PreparedSegmentScene,
    block: AzimuthDCEBlock,
    *,
    batch_lines: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Accumulate lag-one ACCC over a DCE block without building a huge scene.

    Crucially, the last aligned line of one segment is correlated with the first
    aligned line of the next segment, so a DCE block can cross a chunk boundary.
    """
    if batch_lines < 2:
        raise ValueError("batch_lines must be >= 2.")

    nr = prepared.num_range_samples
    c_range = np.zeros(nr, dtype=np.complex128)
    p0_range = np.zeros(nr, dtype=np.float64)
    p1_range = np.zeros(nr, dtype=np.float64)

    previous_line: Optional[np.ndarray] = None
    previous_time: Optional[float] = None
    pair_count = 0
    nominal_dt = prepared.nominal_azimuth_spacing_s
    tolerance = prepared.azimuth_gap_tolerance_s

    for a in prepared.alignments:
        g0 = max(block.start_line, a.global_start_line)
        g1 = min(block.stop_line, a.global_stop_line)
        if g1 <= g0:
            continue

        local0 = g0 - a.global_start_line
        local1 = g1 - a.global_start_line

        for b0 in range(local0, local1, batch_lines):
            b1 = min(local1, b0 + batch_lines)
            y = _align_segment_rows(
                a,
                b0,
                b1,
                nr,
                lanczos_radius=prepared.lanczos_radius,
            )

            if y.shape[0] == 0:
                continue

            eta = np.asarray(a.segment.azimuth_times_s, dtype=np.float64)[b0:b1]

            if (
                previous_line is not None
                and previous_time is not None
                and abs(float(eta[0] - previous_time) - nominal_dt) <= tolerance
            ):
                first = y[0]
                c_range += previous_line * np.conj(first)
                p0_range += np.abs(previous_line) ** 2
                p1_range += np.abs(first) ** 2
                pair_count += 1

            if y.shape[0] > 1:
                valid_pairs = np.abs(np.diff(eta) - nominal_dt) <= tolerance
                x0 = y[:-1][valid_pairs]
                x1 = y[1:][valid_pairs]
                c_range += np.sum(x0 * np.conj(x1), axis=0, dtype=np.complex128)
                p0_range += np.sum(np.abs(x0) ** 2, axis=0, dtype=np.float64)
                p1_range += np.sum(np.abs(x1) ** 2, axis=0, dtype=np.float64)
                pair_count += int(np.count_nonzero(valid_pairs))

            previous_line = np.asarray(y[-1]).copy()
            previous_time = float(eta[-1])

    block_times = prepared.azimuth_times_s[block.start_line:block.stop_line]
    expected_pairs = int(np.count_nonzero(
        np.abs(np.diff(block_times) - nominal_dt) <= tolerance
    ))
    if expected_pairs == 0:
        raise RuntimeError("DCE block contains no valid lag-one azimuth pairs.")
    if pair_count != expected_pairs:
        raise RuntimeError(
            f"DCE block [{block.start_line}, {block.stop_line}) expected "
            f"{expected_pairs} adjacent azimuth pairs, accumulated {pair_count}."
        )

    return c_range, p0_range, p1_range, pair_count


def _fine_dce_from_accumulators(
    c_range: np.ndarray,
    p0_range: np.ndarray,
    p1_range: np.ndarray,
    range_blocks: Sequence[RangeDCEBlock],
    prf_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert streaming ACCC accumulators to range-block fine DCE estimates."""
    c_blocks = np.empty(len(range_blocks), dtype=np.complex128)
    coherence = np.empty(len(range_blocks), dtype=np.float64)

    for i, rb in enumerate(range_blocks):
        r0, r1 = rb.start_sample, rb.stop_sample
        if r0 < 0 or r1 > c_range.size or r1 <= r0:
            raise ValueError(f"Invalid range block {i}: [{r0}, {r1}).")

        c = c_range[r0:r1]
        c_blocks[i] = np.mean(c)

        num = float(np.abs(np.sum(c, dtype=np.complex128)))
        e0 = float(np.sum(p0_range[r0:r1], dtype=np.float64))
        e1 = float(np.sum(p1_range[r0:r1], dtype=np.float64))
        den = np.sqrt(max(e0 * e1, 0.0))
        coherence[i] = 0.0 if den == 0.0 else float(np.clip(num / den, 0.0, 1.0))

    fine_hz = -(float(prf_hz) / (2.0 * np.pi)) * np.angle(c_blocks)
    return fine_hz, c_blocks, coherence


# ---------------------------------------------------------------------------
# DAD §5.3 -- Robust fine-DCE unwrapping
# ---------------------------------------------------------------------------

def unwrap_fine_dce_dad(
    range_times_s: ArrayLike,
    fine_hz: ArrayLike,
    prf_hz: float,
    *,
    fft_length: int = 4096,
    weights: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Robust 1-D fine-DCE unwrapping following DAD §5.3.

    The DAD represents the normalized DC trend as

        f_linear(tau) = a*tau + b

    and estimates the linear component with a zero-padded FFT of

        u[k] = w[k] * exp(j*2*pi*f[k]/PRF).

    The residual phase is then added back to the linear trend.

    Notes
    -----
    - A relative range-time coordinate is used internally. This is numerically
      equivalent to shifting the origin and avoids a large phase offset.
    - The DAD treats the fine estimates as samples on an approximately uniform
      range grid. The median spacing is used as T_S here.
    """
    tau = np.asarray(range_times_s, dtype=np.float64)
    f = np.asarray(fine_hz, dtype=np.float64)

    if tau.ndim != 1 or f.ndim != 1 or tau.size != f.size:
        raise ValueError("range_times_s and fine_hz must be 1-D arrays of equal length.")
    if tau.size < 2:
        return f.copy()
    if prf_hz <= 0:
        raise ValueError("prf_hz must be positive.")
    if np.any(np.diff(tau) <= 0):
        raise ValueError("range_times_s must be strictly increasing.")

    n = tau.size
    nfft = int(fft_length)
    if nfft < n:
        # The published algorithm requires zero-padding, not truncation.
        nfft = 1 << int(np.ceil(np.log2(n)))

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != f.shape:
            raise ValueError("weights must have the same shape as fine_hz.")
        w = np.maximum(w, 0.0)
        if not np.any(w > 0):
            w = np.ones(n, dtype=np.float64)

    tau_rel = tau - tau[0]
    Ts = float(np.median(np.diff(tau_rel)))

    # DAD Eq. analogous to 5-22 / 5-23.
    u = w * np.exp(1j * 2.0 * np.pi * f / prf_hz)
    F = np.fft.fft(u, n=nfft)

    k_peak = int(np.argmax(np.abs(F) ** 2))

    # FFT bin expressed in cycles per range-block sample.
    v_hat = float(np.fft.fftfreq(nfft, d=1.0)[k_peak])

    # DAD Eq. (5-24), in normalized cycles/second and cycles.
    a = v_hat / Ts
    b = float(np.angle(F[k_peak]) / (2.0 * np.pi))

    linear_cycles = a * tau_rel + b

    # Residual normalized phase. exp(j*2*pi*f/PRF) contains the wrapped DC.
    residual_cycles = np.angle(
        np.exp(1j * 2.0 * np.pi * f / prf_hz)
        * np.exp(-1j * 2.0 * np.pi * linear_cycles)
    ) / (2.0 * np.pi)

    # DAD Eq. (5-29)
    return (linear_cycles + residual_cycles) * prf_hz


# ---------------------------------------------------------------------------
# DAD §5.5 -- Polynomial fitting and RMS quality
# ---------------------------------------------------------------------------

def robust_polynomial_fit(
    range_times_s: ArrayLike,
    values_hz: ArrayLike,
    *,
    t0_s: float,
    degree: int = 2,
    outlier_sigma: float = 3.5,
    max_iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares polynomial fit with iterative outlier rejection.

    DAD §5.5 / §5.5.1 requires LS polynomial fitting, outlier elimination and
    an RMS residual quality indicator. The exact operational outlier rejection
    rule is not published in the DAD, therefore the MAD-based rejection below
    is a transparent research implementation.

    Coefficients are returned in ASCENDING order [c0, c1, c2, ...].
    """
    tau = np.asarray(range_times_s, dtype=np.float64)
    y = np.asarray(values_hz, dtype=np.float64)

    if tau.shape != y.shape or tau.ndim != 1:
        raise ValueError("range_times_s and values_hz must be 1-D and equal length.")

    finite = np.isfinite(tau) & np.isfinite(y)
    if np.count_nonzero(finite) < degree + 1:
        raise ValueError("Not enough finite points for polynomial fit.")

    x = tau - float(t0_s)
    mask = finite.copy()

    for _ in range(max(1, int(max_iterations))):
        if np.count_nonzero(mask) < degree + 1:
            break

        coeff = np.polynomial.polynomial.polyfit(x[mask], y[mask], degree)
        fit = np.polynomial.polynomial.polyval(x, coeff)
        residual = y - fit

        r = residual[mask]
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        sigma = 1.4826 * mad

        if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
            break

        new_mask = finite & (np.abs(residual - med) <= outlier_sigma * sigma)

        if np.count_nonzero(new_mask) < degree + 1:
            break
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break

        mask = new_mask

    coeff = np.polynomial.polynomial.polyfit(x[mask], y[mask], degree)
    fit = np.polynomial.polynomial.polyval(x, coeff)
    residual = y - fit

    rms = float(np.sqrt(np.mean(residual[mask] ** 2)))

    return coeff, mask, rms


# ---------------------------------------------------------------------------
# DAD §5.4 -- Absolute ambiguity resolution
# ---------------------------------------------------------------------------

def resolve_absolute_dce_with_geometry(
    range_times_s: ArrayLike,
    fine_unwrapped_hz: ArrayLike,
    geometry_hz: ArrayLike,
    prf_hz: float,
    *,
    t0_s: float,
    degree: int = 2,
    outlier_sigma: float = 3.5,
    max_fit_iterations: int = 5,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    float,
]:
    """Resolve the PRF ambiguity with the geometry DC and fit data DC.

    DAD §5.4 uses the geometry estimate to determine the ambiguity number,
    then fits the data-vs-geometry difference and adds that correction to the
    geometry polynomial.

    Returns
    -------
    absolute_data_hz
    data_polynomial
    geometry_polynomial
    ambiguity_number
    valid_mask
    rms_error_hz

    Implementation note
    -------------------
    [Inference] The public DAD text does not expose every operational rounding
    detail. Here the ambiguity number is selected as the nearest integer PRF
    multiple to the mean geometry-data difference.
    """
    tau = np.asarray(range_times_s, dtype=np.float64)
    fine = np.asarray(fine_unwrapped_hz, dtype=np.float64)
    geom = np.asarray(geometry_hz, dtype=np.float64)

    if tau.shape != fine.shape or tau.shape != geom.shape:
        raise ValueError("range_times_s, fine_unwrapped_hz and geometry_hz must match.")

    # Robust ambiguity estimate over all available range blocks rather than a
    # single noisy point.
    ambiguity_number = int(np.rint(np.median((geom - fine) / prf_hz)))
    absolute_data = fine + ambiguity_number * prf_hz

    geom_coeff, _, _ = robust_polynomial_fit(
        tau,
        geom,
        t0_s=t0_s,
        degree=degree,
        outlier_sigma=outlier_sigma,
        max_iterations=max_fit_iterations,
    )

    delta = absolute_data - geom

    delta_coeff, valid_mask, _ = robust_polynomial_fit(
        tau,
        delta,
        t0_s=t0_s,
        degree=degree,
        outlier_sigma=outlier_sigma,
        max_iterations=max_fit_iterations,
    )

    data_coeff = geom_coeff + delta_coeff

    x = tau - t0_s
    fitted = np.polynomial.polynomial.polyval(x, data_coeff)
    residual = absolute_data - fitted
    rms = float(np.sqrt(np.mean(residual[valid_mask] ** 2)))

    return (
        absolute_data,
        data_coeff,
        geom_coeff,
        ambiguity_number,
        valid_mask,
        rms,
    )


# ---------------------------------------------------------------------------
# High-level notebook API
# ---------------------------------------------------------------------------

class Sentinel1DCE:
    """High-level, notebook-friendly Sentinel-1 DCE estimator."""

    def __init__(
        self,
        prf_hz: float,
        config: DCEConfig,
    ) -> None:
        if prf_hz <= 0:
            raise ValueError("prf_hz must be positive.")

        self.prf_hz = float(prf_hz)
        self.config = config

    @classmethod
    def for_s6_research(cls, prf_hz: float) -> "Sentinel1DCE":
        """Convenience constructor for the reverse-engineered S6 profile."""
        return cls(
            prf_hz=prf_hz,
            config=DCEConfig.s6_research(),
        )

    def build_layout(
        self,
        *,
        n_azimuth_lines: int,
        n_range_samples: int,
        slant_range_times_s: ArrayLike,
        azimuth_times_s: Optional[ArrayLike] = None,
        azimuth_time_offset_s: float = 0.0,
        custom_azimuth_starts: Optional[Sequence[int]] = None,
        slice_start_times_s: Optional[Sequence[float]] = None,
        last_slice_stop_time_s: Optional[float] = None,
        product_start_time_s: Optional[float] = None,
        product_stop_time_s: Optional[float] = None,
        zero_dop_minus_acq_time_s: Optional[float] = None,
    ) -> tuple[list[AzimuthDCEBlock], list[RangeDCEBlock]]:
        """Build and inspect DCE block layout without estimating DC."""
        az_blocks = build_azimuth_blocks(
            n_lines=n_azimuth_lines,
            prf_hz=self.prf_hz,
            config=self.config,
            azimuth_times_s=azimuth_times_s,
            azimuth_time_offset_s=azimuth_time_offset_s,
            custom_starts=custom_azimuth_starts,
            slice_start_times_s=slice_start_times_s,
            last_slice_stop_time_s=last_slice_stop_time_s,
            product_start_time_s=product_start_time_s,
            product_stop_time_s=product_stop_time_s,
            zero_dop_minus_acq_time_s=zero_dop_minus_acq_time_s,
        )

        rg_blocks = build_range_blocks(
            n_range_samples=n_range_samples,
            slant_range_times_s=slant_range_times_s,
            config=self.config,
        )

        return az_blocks, rg_blocks

    def estimate_block(
        self,
        range_compressed_block: np.ndarray,
        azimuth_block: AzimuthDCEBlock,
        range_blocks: Sequence[RangeDCEBlock],
        *,
        t0_s: Optional[float] = None,
        geometry_dc_provider: Optional[GeometryDcProvider] = None,
    ) -> DCERecord:
        """Estimate one DCE record from one range-compressed azimuth block."""
        s = np.asarray(range_compressed_block)

        if s.ndim != 2:
            raise ValueError(
                "range_compressed_block must have shape (azimuth, range)."
            )
        if s.shape[0] != azimuth_block.num_lines:
            raise ValueError(
                f"Expected {azimuth_block.num_lines} azimuth lines, got {s.shape[0]}."
            )

        fine, accc, coherence = fine_dce_cdce(
            s,
            range_blocks,
            self.prf_hz,
        )

        range_times = np.asarray(
            [rb.center_slant_range_time_s for rb in range_blocks],
            dtype=np.float64,
        )

        unwrap_weights = (
            coherence
            if self.config.unwrap_weighting == "coherence"
            else None
        )

        fine_unwrapped = unwrap_fine_dce_dad(
            range_times,
            fine,
            self.prf_hz,
            fft_length=self.config.unwrap_fft_length,
            weights=unwrap_weights,
        )

        if t0_s is None:
            # A local reference close to the data minimizes polynomial
            # conditioning problems. For exact product reproduction the
            # annotation/product t0 may be supplied explicitly.
            t0 = float(range_times[0])
        else:
            t0 = float(t0_s)

        geometry_poly: Optional[np.ndarray] = None
        ambiguity_number: Optional[int] = None
        ambiguity_resolved = False

        if geometry_dc_provider is not None:
            geometry = np.asarray(
                geometry_dc_provider(
                    azimuth_block.azimuth_time_s,
                    range_times,
                ),
                dtype=np.float64,
            )

            if geometry.shape != range_times.shape:
                raise ValueError(
                    "geometry_dc_provider must return one DC value per range block."
                )

            (
                absolute,
                data_poly,
                geometry_poly,
                ambiguity_number,
                valid_mask,
                rms,
            ) = resolve_absolute_dce_with_geometry(
                range_times,
                fine_unwrapped,
                geometry,
                self.prf_hz,
                t0_s=t0,
                degree=self.config.polynomial_degree,
                outlier_sigma=self.config.outlier_sigma,
                max_fit_iterations=self.config.max_fit_iterations,
            )

            ambiguity_resolved = True

        else:
            # Without geometry only the range-unwrapped data DC is known; its
            # absolute PRF ambiguity cannot be guaranteed.
            absolute = fine_unwrapped.copy()

            data_poly, valid_mask, rms = robust_polynomial_fit(
                range_times,
                absolute,
                t0_s=t0,
                degree=self.config.polynomial_degree,
                outlier_sigma=self.config.outlier_sigma,
                max_iterations=self.config.max_fit_iterations,
            )

        return DCERecord(
            block=azimuth_block,
            t0_s=t0,
            range_blocks=list(range_blocks),
            fine_baseband_hz=fine,
            fine_unwrapped_hz=fine_unwrapped,
            fine_absolute_hz=absolute,
            data_dc_polynomial=data_poly,
            geometry_dc_polynomial=geometry_poly,
            accc=accc,
            coherence=coherence,
            valid_mask=valid_mask,
            rms_error_hz=rms,
            rms_above_threshold=bool(rms > self.config.rms_threshold_hz),
            ambiguity_number=ambiguity_number,
            absolute_ambiguity_resolved=ambiguity_resolved,
        )

    def estimate_scene(
        self,
        range_compressed: np.ndarray,
        slant_range_times_s: ArrayLike,
        *,
        azimuth_times_s: Optional[ArrayLike] = None,
        azimuth_time_offset_s: float = 0.0,
        t0_s: Optional[float] = None,
        geometry_dc_provider: Optional[GeometryDcProvider] = None,
        custom_azimuth_starts: Optional[Sequence[int]] = None,
        slice_start_times_s: Optional[Sequence[float]] = None,
        last_slice_stop_time_s: Optional[float] = None,
        product_start_time_s: Optional[float] = None,
        product_stop_time_s: Optional[float] = None,
        zero_dop_minus_acq_time_s: Optional[float] = None,
    ) -> list[DCERecord]:
        """Estimate DCE records for a complete internal signal-data extent.

        Parameters
        ----------
        range_compressed:
            Complex array with shape (azimuth_lines, range_samples).
            To reproduce the supplied S6 product layout, provide the complete
            internal 49890-line stream, not chunk 13 or chunk 14 separately.

        slant_range_times_s:
            Fast-time coordinate of the supplied range-compressed columns.

        azimuth_times_s:
            Optional line-time coordinate. If these are packet/acquisition
            times, ``azimuth_time_offset_s`` can shift them to the DCE time
            convention used for annotation.

        azimuth_time_offset_s:
            Optional constant timing offset, e.g. a known zero-Doppler minus
            acquisition-time offset.

        t0_s:
            Polynomial reference range time. For exact comparison with an ESA
            product, pass the product/reference t0 explicitly.

        geometry_dc_provider:
            Optional callback:
                geometry_dc_provider(azimuth_time_s, range_times_s) -> Hz array
            Supplying this enables DAD §5.4 absolute ambiguity resolution.
            Without geometry, the returned polynomial is range-unwrapped but
            not guaranteed to have the correct integer-PRF ambiguity.
        """
        s = np.asarray(range_compressed)

        if s.ndim != 2:
            raise ValueError(
                "range_compressed must have shape (azimuth_lines, range_samples)."
            )

        n_az, n_rg = s.shape

        az_blocks, rg_blocks = self.build_layout(
            n_azimuth_lines=n_az,
            n_range_samples=n_rg,
            slant_range_times_s=slant_range_times_s,
            azimuth_times_s=azimuth_times_s,
            azimuth_time_offset_s=azimuth_time_offset_s,
            custom_azimuth_starts=custom_azimuth_starts,
            slice_start_times_s=slice_start_times_s,
            last_slice_stop_time_s=last_slice_stop_time_s,
            product_start_time_s=product_start_time_s,
            product_stop_time_s=product_stop_time_s,
            zero_dop_minus_acq_time_s=zero_dop_minus_acq_time_s,
        )

        records: list[DCERecord] = []

        for block in az_blocks:
            data = s[block.start_line:block.stop_line, :]

            record = self.estimate_block(
                data,
                block,
                rg_blocks,
                t0_s=t0_s,
                geometry_dc_provider=geometry_dc_provider,
            )
            records.append(record)

        return records


    def prepare_segments(
        self,
        segments: Sequence[DCESegment],
        *,
        lanczos_radius: int = 8,
        azimuth_gap_tolerance_s: Optional[float] = None,
    ) -> PreparedSegmentScene:
        """Prepare multiple acquisition chunks with different SWST/range grids."""
        return prepare_dce_segments(
            segments,
            self.prf_hz,
            lanczos_radius=lanczos_radius,
            azimuth_gap_tolerance_s=azimuth_gap_tolerance_s,
        )

    def estimate_segments(
        self,
        segments: Sequence[DCESegment],
        *,
        azimuth_time_offset_s: float = 0.0,
        t0_s: Optional[float] = None,
        geometry_dc_provider: Optional[GeometryDcProvider] = None,
        custom_azimuth_starts: Optional[Sequence[int]] = None,
        slice_start_times_s: Optional[Sequence[float]] = None,
        last_slice_stop_time_s: Optional[float] = None,
        product_start_time_s: Optional[float] = None,
        product_stop_time_s: Optional[float] = None,
        zero_dop_minus_acq_time_s: Optional[float] = None,
        lanczos_radius: int = 8,
        azimuth_gap_tolerance_s: Optional[float] = None,
        batch_lines: int = 256,
        return_prepared_scene: bool = False,
    ) -> list[DCERecord] | tuple[list[DCERecord], PreparedSegmentScene]:
        """Estimate DCE over multiple acquisition segments/chunks.

        Unlike :meth:`estimate_scene`, the input chunks do not need the same
        SWST or range-vector length.  They are aligned lazily to a common
        slant-range grid using a Lanczos-windowed sinc fractional delay.

        This is intended for cases such as the supplied S6 acquisition where
        DCE2 crosses the chunk-13/chunk-14 boundary and chunk 14 is displaced
        by about 81.25 range samples relative to chunk 13.
        """
        prepared = self.prepare_segments(
            segments,
            lanczos_radius=lanczos_radius,
            azimuth_gap_tolerance_s=azimuth_gap_tolerance_s,
        )

        az_blocks = build_azimuth_blocks(
            n_lines=prepared.num_azimuth_lines,
            prf_hz=self.prf_hz,
            config=self.config,
            azimuth_times_s=prepared.azimuth_times_s,
            azimuth_time_offset_s=azimuth_time_offset_s,
            custom_starts=custom_azimuth_starts,
            slice_start_times_s=slice_start_times_s,
            last_slice_stop_time_s=last_slice_stop_time_s,
            product_start_time_s=product_start_time_s,
            product_stop_time_s=product_stop_time_s,
            zero_dop_minus_acq_time_s=zero_dop_minus_acq_time_s,
        )
        rg_blocks = build_range_blocks(
            n_range_samples=prepared.num_range_samples,
            slant_range_times_s=prepared.common_slant_range_times_s,
            config=self.config,
        )

        range_times = np.asarray(
            [rb.center_slant_range_time_s for rb in rg_blocks],
            dtype=np.float64,
        )

        records: list[DCERecord] = []

        for block in az_blocks:
            c_range, p0_range, p1_range, _ = _stream_accc_from_segments(
                prepared,
                block,
                batch_lines=batch_lines,
            )
            fine, accc, coherence = _fine_dce_from_accumulators(
                c_range,
                p0_range,
                p1_range,
                rg_blocks,
                self.prf_hz,
            )

            unwrap_weights = (
                coherence
                if self.config.unwrap_weighting == "coherence"
                else None
            )
            fine_unwrapped = unwrap_fine_dce_dad(
                range_times,
                fine,
                self.prf_hz,
                fft_length=self.config.unwrap_fft_length,
                weights=unwrap_weights,
            )

            t0 = float(range_times[0]) if t0_s is None else float(t0_s)
            geometry_poly: Optional[np.ndarray] = None
            ambiguity_number: Optional[int] = None
            ambiguity_resolved = False

            if geometry_dc_provider is not None:
                geometry = np.asarray(
                    geometry_dc_provider(block.azimuth_time_s, range_times),
                    dtype=np.float64,
                )
                if geometry.shape != range_times.shape:
                    raise ValueError(
                        "geometry_dc_provider must return one DC value per range block."
                    )

                (
                    absolute,
                    data_poly,
                    geometry_poly,
                    ambiguity_number,
                    valid_mask,
                    rms,
                ) = resolve_absolute_dce_with_geometry(
                    range_times,
                    fine_unwrapped,
                    geometry,
                    self.prf_hz,
                    t0_s=t0,
                    degree=self.config.polynomial_degree,
                    outlier_sigma=self.config.outlier_sigma,
                    max_fit_iterations=self.config.max_fit_iterations,
                )
                ambiguity_resolved = True
            else:
                absolute = fine_unwrapped.copy()
                data_poly, valid_mask, rms = robust_polynomial_fit(
                    range_times,
                    absolute,
                    t0_s=t0,
                    degree=self.config.polynomial_degree,
                    outlier_sigma=self.config.outlier_sigma,
                    max_iterations=self.config.max_fit_iterations,
                )

            records.append(
                DCERecord(
                    block=block,
                    t0_s=t0,
                    range_blocks=list(rg_blocks),
                    fine_baseband_hz=fine,
                    fine_unwrapped_hz=fine_unwrapped,
                    fine_absolute_hz=absolute,
                    data_dc_polynomial=data_poly,
                    geometry_dc_polynomial=geometry_poly,
                    accc=accc,
                    coherence=coherence,
                    valid_mask=valid_mask,
                    rms_error_hz=rms,
                    rms_above_threshold=bool(rms > self.config.rms_threshold_hz),
                    ambiguity_number=ambiguity_number,
                    absolute_ambiguity_resolved=ambiguity_resolved,
                )
            )

        if return_prepared_scene:
            return records, prepared
        return records

    @staticmethod
    def evaluate_records(
        records: Sequence[DCERecord],
        *,
        azimuth_time_s: float,
        slant_range_times_s: ArrayLike,
        interpolation: Literal["linear", "nearest"] = "linear",
    ) -> np.ndarray:
        """Evaluate DCE at an arbitrary azimuth time and range vector.

        The DAD produces one polynomial per DCE azimuth block. This helper
        provides a convenient notebook interface between those records.

        [Inference]
        Linear interpolation between neighboring DCE records is a research
        convenience and is NOT claimed here as an explicitly published DAD
        requirement.
        """
        if not records:
            raise ValueError("records must not be empty.")

        recs = sorted(records, key=lambda r: r.block.azimuth_time_s)
        times = np.asarray(
            [r.block.azimuth_time_s for r in recs],
            dtype=np.float64,
        )
        t = float(azimuth_time_s)

        if interpolation == "nearest":
            idx = int(np.argmin(np.abs(times - t)))
            return recs[idx].evaluate(slant_range_times_s)

        if interpolation != "linear":
            raise ValueError("interpolation must be 'linear' or 'nearest'.")

        if t <= times[0]:
            return recs[0].evaluate(slant_range_times_s)
        if t >= times[-1]:
            return recs[-1].evaluate(slant_range_times_s)

        hi = int(np.searchsorted(times, t, side="right"))
        lo = hi - 1

        alpha = (t - times[lo]) / (times[hi] - times[lo])

        # Evaluate each polynomial first, then interpolate the frequencies.
        # This remains correct even if the records use different t0 values.
        f0 = recs[lo].evaluate(slant_range_times_s)
        f1 = recs[hi].evaluate(slant_range_times_s)

        return (1.0 - alpha) * f0 + alpha * f1

    @staticmethod
    def evaluate_for_line(
        records: Sequence[DCERecord],
        *,
        line_index: int,
        azimuth_times_s: ArrayLike,
        slant_range_times_s: ArrayLike,
        azimuth_time_offset_s: float = 0.0,
        interpolation: Literal["linear", "nearest"] = "linear",
    ) -> np.ndarray:
        """Notebook convenience wrapper analogous to ``dce_for_block(index)``."""
        az = np.asarray(azimuth_times_s, dtype=np.float64)

        if line_index < 0 or line_index >= az.size:
            raise IndexError("line_index is outside azimuth_times_s.")

        t = float(az[line_index] + azimuth_time_offset_s)

        return Sentinel1DCE.evaluate_records(
            records,
            azimuth_time_s=t,
            slant_range_times_s=slant_range_times_s,
            interpolation=interpolation,
        )


# ---------------------------------------------------------------------------
# Small diagnostic helpers
# ---------------------------------------------------------------------------

GPS_EPOCH_UTC = np.datetime64("1980-01-06T00:00:00", "us")


def utc_iso_to_gps_seconds(text: str, gps_utc_offset_s: float = 18.0) -> float:
    """Convert a UTC ISO timestamp to GPS seconds since 1980-01-06."""
    utc_s = (
        np.datetime64(text, "us") - GPS_EPOCH_UTC
    ) / np.timedelta64(1, "s")
    return float(utc_s + gps_utc_offset_s)


def prepare_annotation_records(
    records: Sequence[dict], gps_utc_offset_s: float = 18.0
) -> list[dict]:
    """Normalize L1 annotation DCE dictionaries for numerical evaluation."""
    prepared = []
    for record in records:
        item = dict(record)
        item["dataDcPolynomial"] = np.asarray(
            item["dataDcPolynomial"], dtype=np.float64
        )
        item["azimuth_s"] = utc_iso_to_gps_seconds(
            item["azimuthTime"], gps_utc_offset_s
        )
        for source, target in (("fineStart", "fine_start_s"),
                               ("fineStop", "fine_stop_s")):
            if source in item:
                item[target] = utc_iso_to_gps_seconds(
                    item[source], gps_utc_offset_s
                )
        prepared.append(item)
    return sorted(prepared, key=lambda item: item["azimuth_s"])


def evaluate_annotation_dce(
    records: Sequence[dict],
    azimuth_time_s: float,
    slant_range_times_s: ArrayLike,
) -> np.ndarray:
    """Evaluate and linearly interpolate annotation DCE polynomials."""
    if not records:
        raise ValueError("records must not be empty.")

    def evaluate(record):
        dt = np.asarray(slant_range_times_s) - record["t0"]
        return np.polynomial.polynomial.polyval(
            dt, record["dataDcPolynomial"]
        )

    times = np.asarray([record["azimuth_s"] for record in records])
    time_s = float(azimuth_time_s)
    if time_s <= times[0]:
        return evaluate(records[0])
    if time_s >= times[-1]:
        return evaluate(records[-1])

    hi = int(np.searchsorted(times, time_s, side="right"))
    lo = hi - 1
    alpha = (time_s - times[lo]) / (times[hi] - times[lo])
    return (1.0 - alpha) * evaluate(records[lo]) + alpha * evaluate(records[hi])


def compare_annotation_dce(
    annotation_records: Sequence[dict],
    estimated_records: Sequence[DCERecord],
    slant_range_times_s: ArrayLike,
    *,
    prf_hz: Optional[float] = None,
) -> list[dict]:
    """Compare each annotation polynomial with the nearest estimated record."""
    if not annotation_records or not estimated_records:
        raise ValueError("annotation_records and estimated_records must not be empty.")

    tau = np.asarray(slant_range_times_s, dtype=np.float64)
    comparisons = []
    for index, annotation in enumerate(annotation_records, start=1):
        estimated = min(
            estimated_records,
            key=lambda record: abs(
                record.block.azimuth_time_s - annotation["azimuth_s"]
            ),
        )
        reference_hz = evaluate_annotation_dce(
            [annotation], annotation["azimuth_s"], tau
        )
        estimated_hz = estimated.evaluate(tau)
        error_hz = estimated_hz - reference_hz

        ambiguity_hz = 0.0
        if prf_hz is not None:
            ambiguity_hz = float(
                np.rint(-np.median(error_hz) / prf_hz) * prf_hz
            )
        adjusted_error_hz = error_hz + ambiguity_hz

        comparisons.append({
            "record": index,
            "range_times_s": tau,
            "annotation_hz": reference_hz,
            "estimated_hz": estimated_hz,
            "annotation_coefficients": np.asarray(
                annotation["dataDcPolynomial"], dtype=np.float64
            ),
            "estimated_coefficients": estimated.data_dc_polynomial.copy(),
            "azimuth_time_error_ms": 1e3 * (
                estimated.block.azimuth_time_s - annotation["azimuth_s"]
            ),
            "bias_hz": float(np.mean(error_hz)),
            "mae_hz": float(np.mean(np.abs(error_hz))),
            "rmse_hz": float(np.sqrt(np.mean(error_hz**2))),
            "max_abs_error_hz": float(np.max(np.abs(error_hz))),
            "integer_prf_adjustment_hz": ambiguity_hz,
            "ambiguity_adjusted_rmse_hz": float(
                np.sqrt(np.mean(adjusted_error_hz**2))
            ),
            "fit_rms_hz": estimated.rms_error_hz,
            "mean_coherence": float(np.mean(estimated.coherence)),
            "absolute_ambiguity_resolved": estimated.absolute_ambiguity_resolved,
        })

    return comparisons

def records_summary(records: Sequence[DCERecord]) -> list[dict]:
    """Return a notebook-friendly list of dictionaries."""
    out: list[dict] = []

    for i, r in enumerate(records):
        out.append(
            {
                "index": i,
                "start_line": r.block.start_line,
                "stop_line": r.block.stop_line,
                "num_lines": r.block.num_lines,
                "azimuth_start_s": r.block.start_time_s,
                "azimuth_time_s": r.block.azimuth_time_s,
                "azimuth_stop_s": r.block.stop_time_s,
                "t0_s": r.t0_s,
                "rms_error_hz": r.rms_error_hz,
                "rms_above_threshold": r.rms_above_threshold,
                "ambiguity_number": r.ambiguity_number,
                "absolute_ambiguity_resolved": r.absolute_ambiguity_resolved,
                "data_dc_polynomial": r.data_dc_polynomial.copy(),
            }
        )

    return out


def _self_test() -> None:
    """Synthetic tests for CDCE and multi-segment fractional alignment."""
    # --- Global sliced-product scheduler: 3/2/2 records over three slices ---
    schedule_prf = 1000.0
    schedule_lines = np.arange(3000, dtype=np.float64) / schedule_prf
    schedule_config = DCEConfig(
        azimuth_block_size_lines=100,
        num_range_blocks=1,
        azimuth_placement="slice_timeline",
    )
    schedule_kwargs = {
        "slice_start_times_s": [0.0, 1.0, 2.0],
        "last_slice_stop_time_s": 3.0,
        "zero_dop_minus_acq_time_s": 0.0,
    }
    for product_times, expected_starts in (
        ((0.0, 1.1), [0, 500, 1000]),
        ((1.1, 2.1), [1500, 2000]),
        ((2.1, 3.0), [2500, 2900]),
    ):
        blocks = build_azimuth_blocks(
            n_lines=schedule_lines.size,
            prf_hz=schedule_prf,
            config=schedule_config,
            azimuth_times_s=schedule_lines,
            product_start_time_s=product_times[0],
            product_stop_time_s=product_times[1],
            **schedule_kwargs,
        )
        if [block.start_line for block in blocks] != expected_starts:
            raise RuntimeError("Slice-timeline scheduler self-test failed.")

    # --- Original single-array CDCE test ---
    prf = 1000.0
    n_az = 256
    n_rg = 120

    r = np.arange(n_rg)
    true_f = 60.0 + 0.15 * r

    eta = np.arange(n_az)[:, None]
    s = np.exp(1j * 2.0 * np.pi * true_f[None, :] * eta / prf)

    tau = 0.006 + np.arange(n_rg) / 20e6

    cfg = DCEConfig(
        azimuth_block_size_lines=n_az,
        num_range_blocks=6,
        range_block_size_samples=20,
        azimuth_placement="custom",
        unwrap_fft_length=1024,
    )

    rg_blocks = build_range_blocks(n_rg, tau, cfg)
    fine, _, _ = fine_dce_cdce(s, rg_blocks, prf)

    expected = np.array(
        [np.mean(true_f[b.start_sample:b.stop_sample]) for b in rg_blocks]
    )
    if np.max(np.abs(fine - expected)) > 0.2:
        raise RuntimeError("Synthetic CDCE self-test failed.")

    # --- Multi-segment test: segment #2 starts 0.25 range sample earlier ---
    fs = 20e6
    dt = 1.0 / fs
    n1 = 160
    n2 = 160
    nr1 = 140
    nr2 = 150
    fdc = 82.0

    tau1 = 0.006 + np.arange(nr1) * dt
    tau2 = (0.006 - 0.25 * dt) + np.arange(nr2) * dt

    eta1 = np.arange(n1) / prf
    eta2 = np.arange(n1, n1 + n2) / prf

    # A continuous range phase makes fractional-grid alignment testable.
    fr = 1.7e6
    data1 = np.exp(
        1j * (
            2.0 * np.pi * fdc * eta1[:, None]
            + 2.0 * np.pi * fr * tau1[None, :]
        )
    )
    data2 = np.exp(
        1j * (
            2.0 * np.pi * fdc * eta2[:, None]
            + 2.0 * np.pi * fr * tau2[None, :]
        )
    )

    seg1 = DCESegment(data1, tau1, eta1, name="seg1")
    seg2 = DCESegment(data2, tau2, eta2, name="seg2")

    cfg2 = DCEConfig(
        azimuth_block_size_lines=n1 + n2,
        num_range_blocks=5,
        range_block_size_samples=20,
        azimuth_placement="custom",
        unwrap_fft_length=1024,
    )
    est = Sentinel1DCE(prf, cfg2)
    recs, prepared = est.estimate_segments(
        [seg1, seg2],
        custom_azimuth_starts=[0],
        return_prepared_scene=True,
        batch_lines=64,
    )

    shift = prepared.alignment_summary()[1]["source_start_index"]
    if abs(shift - 0.25) > 1e-8:
        raise RuntimeError(
            f"Fractional alignment self-test failed: expected 0.25, got {shift}."
        )
    if prepared.num_range_samples != nr1:
        raise RuntimeError("Reference range-grid self-test failed.")

    if np.max(np.abs(recs[0].fine_baseband_hz - fdc)) > 0.25:
        raise RuntimeError("Multi-segment DCE self-test failed.")

    # A missing azimuth line between segments must not become a lag-one pair.
    seg2_gap = DCESegment(data2, tau2, eta2 + 1.0 / prf, name="seg2_gap")
    prepared_gap = est.prepare_segments([seg1, seg2_gap])
    az_blocks, _ = est.build_layout(
        n_azimuth_lines=n1 + n2,
        n_range_samples=prepared_gap.num_range_samples,
        slant_range_times_s=prepared_gap.common_slant_range_times_s,
        azimuth_times_s=prepared_gap.azimuth_times_s,
        custom_azimuth_starts=[0],
    )
    *_, pair_count = _stream_accc_from_segments(
        prepared_gap,
        az_blocks[0],
        batch_lines=64,
    )
    if pair_count != n1 + n2 - 2:
        raise RuntimeError("Azimuth-gap self-test failed.")


__all__ = [
    "DCEConfig",
    "DCESegment",
    "SegmentAlignment",
    "PreparedSegmentScene",
    "AzimuthDCEBlock",
    "RangeDCEBlock",
    "DCERecord",
    "Sentinel1DCE",
    "build_azimuth_blocks",
    "build_range_blocks",
    "lag1_accc",
    "fine_dce_cdce",
    "prepare_dce_segments",
    "unwrap_fine_dce_dad",
    "robust_polynomial_fit",
    "resolve_absolute_dce_with_geometry",
    "utc_iso_to_gps_seconds",
    "prepare_annotation_records",
    "evaluate_annotation_dce",
    "compare_annotation_dce",
    "records_summary",
]


if __name__ == "__main__":
    _self_test()
    print("sentinel1_processing.dce self-test: PASS")
