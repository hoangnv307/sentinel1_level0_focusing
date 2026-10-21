"""Receiver gain vector from DAD §9.2.3, Eq. 9-9–9-16."""

import numpy as np


def calculate_gain(
    n_samples, *, sample_rate_hz, swst_s, pulse_length_s, suppressed_s,
    gain_trend_coefficients, gain_overshoot_coefficients, enabled,
):
    """Return power gain GV; divide complex samples by sqrt(GV), Eq. 9-16.

    SWST is the downlink value, before adding suppressed time. Trend time is
    relative to TX end; overshoot time starts at the first retained sample.
    API times are seconds; both AUX models use microseconds internally.
    Disabled correction returns identity without evaluating the coefficients.
    """
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("Cờ receiver correction phải là boolean.")
    if (not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool)
            or n_samples <= 0):
        raise ValueError("n_samples phải là số nguyên dương.")
    timing = (sample_rate_hz, swst_s, pulse_length_s, suppressed_s)
    if any(np.asarray(v).ndim != 0 or np.asarray(v).dtype.kind not in "fiu"
           or not np.isfinite(v) for v in timing):
        raise ValueError("Timing receiver phải là scalar thực hữu hạn.")
    if sample_rate_hz <= 0 or pulse_length_s <= 0 or swst_s < 0 or suppressed_s < 0:
        raise ValueError("Sample rate/pulse length phải dương; SWST/suppressed time không âm.")
    if not enabled:
        return np.ones(n_samples)
    trend_coefficients = np.asarray(gain_trend_coefficients)
    overshoot_coefficients = np.asarray(gain_overshoot_coefficients)
    if (trend_coefficients.shape != (8,) or overshoot_coefficients.shape != (5,)
            or any(c.dtype.kind not in "fiu" or not np.isfinite(c).all()
                   for c in (trend_coefficients, overshoot_coefficients))):
        raise ValueError("Receiver cần 8 hệ số trend và 5 hệ số overshoot hữu hạn.")
    trend_coefficients = trend_coefficients.astype(float)
    overshoot_coefficients = overshoot_coefficients.astype(float)
    ts = np.arange(n_samples) / sample_rate_hz * 1e6
    t1 = ts + (swst_s + suppressed_s - pulse_length_s) * 1e6
    p, q = trend_coefficients[:5], trend_coefficients[5:]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        denominator = np.polynomial.polynomial.polyval(t1, q)
        if not np.isfinite(denominator).all() or np.any(denominator == 0):
            raise ValueError("Receiver gain có mẫu số bằng 0 hoặc không hữu hạn.")
        trend = np.polynomial.polynomial.polyval(t1, p) / denominator
        a, b, c, d, phi = overshoot_coefficients
        overshoot = a * np.exp(-b * ts) * np.sin(d * ts + phi) + c
        gain = 10.0 ** ((trend + overshoot) / 10.0)
    if not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("Receiver gain không hữu hạn hoặc không dương.")
    return gain


__all__ = ["calculate_gain"]
