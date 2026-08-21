"""
sentinel1_dce.py
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
- DCE block starts approximately at:
      0, floor(N_input/2), N_input - 6000
- nominal range block size = 1000 samples as a research starting point.
  The exact IPF internal range-block configuration is not public.

The module separates these assumptions from the core DAD algorithms so that
they can be changed without touching the estimator itself.
"""

from __future__ import annotations

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
    # "edge_mid_edge": [Inference] reverse-engineered S6 placement.
    # "custom"   : use explicit starts passed to build_azimuth_blocks().
    azimuth_placement: Literal["spacing", "edge_mid_edge", "custom"] = "spacing"

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
        - DCE block starts follow edge / middle / edge behavior for N=49890.
        - 1000 range samples is a starting estimate suggested by the first
          fine-DCE center being ~500 samples from t0. It is NOT claimed as a
          published IPF parameter.
        """
        return cls(
            azimuth_block_size_lines=6000,
            num_range_blocks=20,
            range_block_size_samples=1000,
            azimuth_placement="edge_mid_edge",
            azimuth_spacing_lines=None,
            unwrap_fft_length=4096,
            polynomial_degree=2,
            rms_threshold_hz=20.0,
        )


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
) -> list[AzimuthDCEBlock]:
    """Build DCE azimuth blocks.

    DAD §5.6 states that block size and estimate spacing are configurable.

    If ``azimuth_times_s`` is supplied, it is interpreted as the time attached
    to each input line. ``azimuth_time_offset_s`` can convert packet/acquisition
    time to the desired DCE time convention (e.g. zero-Doppler time).

    For the supplied S6 scene, the observed product is reproduced at the level
    of line placement by ``azimuth_placement="edge_mid_edge"``:
        [0, floor(N/2), N-B]
    where B is the DCE block length. This placement rule is [Inference].
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

    placement = config.azimuth_placement

    if placement == "edge_mid_edge":
        starts = [0, n_lines // 2, n_lines - B]

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

    elif placement == "custom":
        if not custom_starts:
            raise ValueError(
                "custom_starts must be supplied for placement='custom'."
            )
        starts = [int(s) for s in custom_starts]

    else:
        raise ValueError(f"Unsupported azimuth placement: {placement}")

    # Clip, sort and remove duplicates.
    last_start = n_lines - B
    starts = sorted({min(max(0, int(s)), last_start) for s in starts})

    if azimuth_times_s is not None:
        line_times = np.asarray(azimuth_times_s, dtype=np.float64)
        if line_times.ndim != 1 or line_times.size != n_lines:
            raise ValueError(
                "azimuth_times_s must be a 1-D array with one value per input line."
            )
        if np.any(np.diff(line_times) <= 0):
            raise ValueError("azimuth_times_s must be strictly increasing.")
        line_times = line_times + float(azimuth_time_offset_s)
    else:
        line_times = None

    blocks: list[AzimuthDCEBlock] = []

    for start in starts:
        stop = start + B

        if line_times is None:
            t_start = start / prf_hz + azimuth_time_offset_s
            t_stop = stop / prf_hz + azimuth_time_offset_s
        else:
            t_start = float(line_times[start])
            if stop < n_lines:
                # Right-hand boundary is exactly the next line time.
                t_stop = float(line_times[stop])
            else:
                # Extrapolate one PRI after the final line.
                t_stop = float(line_times[-1] + 1.0 / prf_hz)

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
    ) -> tuple[list[AzimuthDCEBlock], list[RangeDCEBlock]]:
        """Build and inspect DCE block layout without estimating DC."""
        az_blocks = build_azimuth_blocks(
            n_lines=n_azimuth_lines,
            prf_hz=self.prf_hz,
            config=self.config,
            azimuth_times_s=azimuth_times_s,
            azimuth_time_offset_s=azimuth_time_offset_s,
            custom_starts=custom_azimuth_starts,
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
    """Minimal synthetic CDCE test; no external Sentinel-1 data required."""
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


__all__ = [
    "DCEConfig",
    "AzimuthDCEBlock",
    "RangeDCEBlock",
    "DCERecord",
    "Sentinel1DCE",
    "build_azimuth_blocks",
    "build_range_blocks",
    "lag1_accc",
    "fine_dce_cdce",
    "unwrap_fine_dce_dad",
    "robust_polynomial_fit",
    "resolve_absolute_dce_with_geometry",
    "records_summary",
]


if __name__ == "__main__":
    _self_test()
    print("sentinel1_dce.py self-test: PASS")
