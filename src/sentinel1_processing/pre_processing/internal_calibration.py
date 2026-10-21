"""Pure calibration/PG operations: DAD §4.2.1.2–4.2.1.8 and §9.4."""

import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.signal import resample


def _complex(data, ndim):
    array = np.asarray(data)
    if (array.ndim != ndim or not array.size or not np.iscomplexobj(array)
            or not np.isfinite(array).all()):
        raise ValueError("Calibration cần array complex hữu hạn, đúng số chiều và không rỗng.")
    return array.astype(np.complex128)


def _nonnegative_real(value):
    a = np.asarray(value)
    return a.ndim == 0 and a.dtype.kind in "fiu" and np.isfinite(a) and a >= 0


def extract_pulse(decoded, *, order, method):
    """AUX pulse order is 1-based; PCC2=(x-y)/2, Average=mean(selected)."""
    data = _complex(decoded, 2)
    indices = np.asarray(order)
    if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != len(indices) or indices.min() < 1 or indices.max() > len(data)):
        raise ValueError("Pulse order phải là vị trí 1-based hợp lệ, không lặp.")
    selected = data[indices-1]
    if method == "PCC2" and len(indices) == 2:
        return (selected[0]-selected[1])/2
    if method == "Average":
        return selected.mean(axis=0)
    raise ValueError("Chỉ hỗ trợ PCC2 với hai pulse hoặc Average.")


def reconstruct_replica(tx, rx, epdn, ta, apdn, *, denominator_relative_floor):
    """DAD Eq. 4-32/4-33; reject singular spectra, never silently regularize."""
    pulses = [_complex(x, 1) for x in (tx, rx, epdn, ta, apdn)]
    if (len({p.shape for p in pulses}) != 1 or not _nonnegative_real(denominator_relative_floor)
            or denominator_relative_floor >= 1):
        raise ValueError("Calibration pulse phải cùng shape; spectral floor trong [0,1).")
    tx, rx, epdn, ta, apdn = [fft(x) for x in pulses]
    denominator = epdn*apdn
    if np.any(abs(denominator) <= denominator_relative_floor*abs(denominator).max()):
        raise ValueError("Không đủ spectral support EPDN/APDN để tái tạo replica.")
    replica = ifft(tx*rx*ta/denominator)
    if not np.isfinite(replica).all():
        raise ValueError("Replica reconstruction không hữu hạn.")
    return replica


def correlation_peak(signal, reference, *, oversampling_factor):
    """FFT zoom plus three-point parabolic refinement (§4.2.1.4); complex peak."""
    signal, reference = _complex(signal, 1), _complex(reference, 1)
    if (len(reference) > len(signal) or not isinstance(oversampling_factor, (int, np.integer))
            or isinstance(oversampling_factor, bool) or oversampling_factor < 2):
        raise ValueError("Reference dài hơn signal hoặc oversampling factor không phải số nguyên >=2.")
    correlation = ifft(fft(signal)*fft(reference, n=len(signal)).conj())
    zoomed = resample(correlation, len(signal)*oversampling_factor)
    index = int(np.argmax(abs(zoomed)))
    left, center, right = zoomed[(index-1) % len(zoomed)], zoomed[index], zoomed[(index+1) % len(zoomed)]
    a, b, c = abs(left), abs(center), abs(right)
    curvature = a-2*b+c
    delta = 0. if curvature >= 0 else float(np.clip(.5*(a-c)/curvature, -.5, .5))
    peak = center + .5*delta*(right-left) + .5*delta**2*(left-2*center+right)
    lag = (index+delta)/oversampling_factor
    if lag > len(signal)/2:
        lag -= len(signal)
    if not np.isfinite(peak) or abs(peak) == 0:
        raise ValueError("Không có correlation peak hữu hạn khác zero.")
    return complex(peak), float(lag)


def nominal_pg_replica(*, sample_rate_hz, pulse_length_s, start_frequency_hz, ramp_rate_hz_per_s):
    """DAD Eq. 4-25 / Calibration Plan 8.1 Eq. 9.2-14: amplitude 1/nsamples."""
    values = (sample_rate_hz, pulse_length_s, start_frequency_hz, ramp_rate_hz_per_s)
    if (any(np.asarray(v).ndim != 0 or np.asarray(v).dtype.kind not in "fiu"
            or not np.isfinite(v) for v in values) or sample_rate_hz <= 0 or pulse_length_s <= 0):
        raise ValueError("Chirp cần sample rate/pulse length dương và tham số thực hữu hạn.")
    with np.errstate(over="ignore", under="ignore"):
        samples = np.float64(sample_rate_hz)*pulse_length_s
    if not np.isfinite(samples) or samples <= 0:
        raise ValueError("Số mẫu nominal chirp không hữu hạn hoặc bằng zero.")
    count = int(np.ceil(samples))
    time = np.arange(count)/sample_rate_hz
    return np.exp(2j*np.pi*(start_frequency_hz*time + .5*ramp_rate_hz_per_s*time**2))/count


def pg_products(replicas, *, reference, pg_reference, oversampling_factor):
    """PGref/PGcal before swath ratio; caller supplies the calibrated reference.

    Plan 8.1 §9.2.4–6 uses the nominal 1/N chirp. Never substitute a raw
    measured replica: its amplitude and energy change the units of PGcal.
    """
    replicas = _complex(replicas, 2)
    ref = np.asarray(pg_reference)
    if ref.ndim != 0 or ref.dtype.kind not in "cfiu" or not np.isfinite(ref) or abs(ref) == 0:
        raise ValueError("PG reference phải hữu hạn khác zero.")
    reference = _complex(reference, 1)
    peaks, lags = zip(*(correlation_peak(r, reference, oversampling_factor=oversampling_factor) for r in replicas))
    with np.errstate(over="ignore", invalid="ignore"):
        products = pg_reference/np.array(peaks)
    if not np.isfinite(products).all() or np.any(abs(products) == 0):
        raise ValueError("PG product không hữu hạn hoặc bằng zero.")
    return products, np.array(peaks), np.array(lags)


def validate_pg(
    products, model, *, pg_amp_std_fraction, pg_phase_std_fraction,
    max_pg_amp_error, max_pg_phase_error, invalid_pg_max_fraction, use_model,
):
    """§4.2.1.8: QC precedes swath gain ratio; explicit model fallback policy."""
    products, model = _complex(products, 1), _complex(model, 1)
    limits = (pg_amp_std_fraction, pg_phase_std_fraction, max_pg_amp_error, max_pg_phase_error)
    if (products.shape != model.shape or len(products) < 2 or np.any(abs(products) == 0)
            or np.any(abs(model) == 0) or any(not _nonnegative_real(v) for v in limits)
            or not _nonnegative_real(invalid_pg_max_fraction) or invalid_pg_max_fraction > 1
            or not isinstance(use_model, (bool, np.bool_))):
        raise ValueError("PG/model hoặc configurable thresholds không hợp lệ.")
    amplitude, phase = abs(products), np.unwrap(np.angle(products))
    relative = ((abs(amplitude-amplitude.mean()) <= pg_amp_std_fraction*amplitude.std()) &
                (abs(phase-phase.mean()) <= pg_phase_std_fraction*phase.std()))
    absolute = ((abs(amplitude-abs(model)) < max_pg_amp_error) &
                (abs(np.angle(products/model)) < max_pg_phase_error))
    valid = relative & absolute
    invalid_fraction = float(np.mean(~valid))
    fallback = bool(use_model or invalid_fraction > invalid_pg_max_fraction)
    selected = np.ones(len(products), dtype=bool) if fallback else valid
    if selected.sum() < 2:
        raise ValueError("Không đủ hai PG valid; AUX không yêu cầu fallback theo invalid fraction hiện tại.")
    return {"relative_valid": relative.tolist(), "absolute_valid": absolute.tolist(),
            "valid": valid.tolist(), "invalid_fraction": invalid_fraction,
            "source": "model" if fallback else "extracted", "selected": selected.tolist()}, model if fallback else products


def mean_internal_delay(peak_lags_samples, *, sample_rate_hz, delta_t_guard1_s, delta_t_suppr_s, valid):
    """DAD Eq. 4-34: mean delay of valid correlation peaks, independent of PG QC."""
    lags, mask = np.asarray(peak_lags_samples), np.asarray(valid)
    if (lags.ndim != 1 or lags.dtype.kind not in "fiu" or mask.shape != lags.shape
            or mask.dtype.kind != "b" or not mask.any() or not np.isfinite(lags[mask]).all()
            or not _nonnegative_real(sample_rate_hz) or sample_rate_hz == 0
            or not _nonnegative_real(delta_t_guard1_s) or not _nonnegative_real(delta_t_suppr_s)):
        raise ValueError("Delay cần peak valid hữu hạn, mask bool, sample rate dương và guard/suppression hợp lệ.")
    delay = np.mean(lags[mask]/sample_rate_hz - (delta_t_guard1_s-delta_t_suppr_s))
    if not np.isfinite(delay):
        raise ValueError("Internal delay trung bình không hữu hạn.")
    return float(delay)


def interpolate_pg(times_s, corrections, query_times_s):
    """Eq. 9-17: interpolate/extrapolate complex values, never wrapped phase."""
    times = np.asarray(times_s)
    values = _complex(corrections, 1)
    query = np.asarray(query_times_s)
    if (times.ndim != 1 or times.dtype.kind not in "fiu" or query.dtype.kind not in "fiu"
            or len(times) != len(values) or len(times) < 2
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(query).all() or not query.size):
        raise ValueError("PG time phải hữu hạn, tăng nghiêm ngặt, đủ hai node.")
    result = interp1d(times, values, bounds_error=False, fill_value="extrapolate")(query)
    if not np.isfinite(result).all() or np.any(abs(result) == 0):
        raise ValueError("PG nội/ngoại suy không hữu hạn hoặc bằng zero.")
    return result


__all__ = ["extract_pulse", "reconstruct_replica", "correlation_peak", "nominal_pg_replica", "pg_products", "validate_pg", "mean_internal_delay", "interpolate_pg"]
