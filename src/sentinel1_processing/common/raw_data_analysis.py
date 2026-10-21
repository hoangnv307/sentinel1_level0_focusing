"""Raw I/Q statistics, DAD §4.1–4.1.1 (population variances, Fisher transform)."""

import numpy as np


def analyze_batches(batches):
    """Analyze selected valid echo rows; batch boundaries do not define analysis scopes.

    Each batch is (lines, range samples). Selection belongs to the caller.
    ``outside_null_bounds`` means significant departure from the ideal value;
    it is deliberately not the ambiguous IPF ``*SignificanceFlag`` name.
    """
    count = lines = 0
    width = None
    mean = np.zeros(2)
    m2 = np.zeros(2)
    correlations = []
    excluded_constant = excluded_perfect = 0
    for batch in batches:
        raw = np.asarray(batch)
        if (raw.ndim != 2 or not np.iscomplexobj(raw) or raw.shape[0] == 0
                or raw.shape[1] < 2 or not np.isfinite(raw).all()):
            raise ValueError("Mỗi batch phải là echo complex 2D hữu hạn, không rỗng, có ít nhất 2 range samples.")
        if width is not None and raw.shape[1] != width:
            raise ValueError("Một analysis scope phải có cùng số range samples mỗi dòng.")
        width = raw.shape[1]
        i = raw.real.astype(np.float64)
        q = raw.imag.astype(np.float64)
        batch_mean = np.array([i.mean(), q.mean()])
        batch_m2 = np.array([np.var(i), np.var(q)]) * raw.size
        total = count + raw.size
        delta = batch_mean - mean
        m2 += batch_m2 + delta**2 * (count * raw.size / total)
        mean += delta * (raw.size / total)
        count = total
        lines += raw.shape[0]
        # Center per line: Eq. 4-12's printed parentheses conflict with covariance.
        i -= i.mean(axis=1, keepdims=True)
        q -= q.mean(axis=1, keepdims=True)
        sii = np.einsum("ij,ij->i", i, i)
        sqq = np.einsum("ij,ij->i", q, q)
        siq = np.einsum("ij,ij->i", i, q)
        valid = (sii > 0) & (sqq > 0)
        excluded_constant += int((~valid).sum())
        corr = siq[valid] / np.sqrt(sii[valid] * sqq[valid])
        if not np.isfinite(corr).all() or np.any(abs(corr) > 1 + 64*np.finfo(float).eps):
            raise ValueError("Correlation I/Q không hợp lệ.")
        # Fisher z is undefined at ±1; report exclusions, do not silently clip infinity.
        perfect = abs(corr) >= 1 - 8*np.finfo(float).eps
        excluded_perfect += int(perfect.sum())
        correlations.extend(corr[~perfect].tolist())
    if count == 0:
        raise ValueError("Không có echo hợp lệ để phân tích.")
    std = np.sqrt(m2/count)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Moments I/Q không hữu hạn.")
    bias_bound = 3 * std/np.sqrt(count)
    gain_bounds = [float(1-3/np.sqrt(count)), float(1+3/np.sqrt(count))]
    gain = float(std[0]/std[1]) if np.all(std > 0) else None
    theta = theta_bounds = null_corr_bounds = z_mean = z_std = None
    if len(correlations) >= 2:
        z = np.arctanh(correlations)
        z_mean, z_std = float(np.mean(z)), float(np.std(z))
        c = np.tanh(z_mean)
        sigma_minus = c-np.tanh(z_mean-z_std)
        sigma_plus = np.tanh(z_mean+z_std)-c
        theta = float(np.arcsin(c))
        theta_bounds = [float(np.arcsin(np.tanh(z_mean-z_std))),
                        float(np.arcsin(np.tanh(z_mean+z_std)))]
        # Symmetric zero-null interpretation; the printed lower sign in §4.1 is inconsistent.
        null_corr_bounds = [float(-3*sigma_minus), float(3*sigma_plus)]
    outside = lambda value, bounds: None if value is None else bool(value < bounds[0] or value > bounds[1])
    return {
        "line_count": int(lines), "range_sample_count": int(width), "sample_count": int(count),
        "mean_i": float(mean[0]), "mean_q": float(mean[1]),
        "std_i": float(std[0]), "std_q": float(std[1]),
        "iq_gain_imbalance": gain, "iq_quadrature_departure_rad": theta,
        "bounds": {"i_bias": [-float(bias_bound[0]), float(bias_bound[0])],
                   "q_bias": [-float(bias_bound[1]), float(bias_bound[1])],
                   "iq_gain_imbalance": gain_bounds,
                   "iq_quadrature_departure_rad": theta_bounds,
                   "null_quadrature_correlation": null_corr_bounds},
        "outside_null_bounds": {
            "i_bias": bool(abs(mean[0]) > bias_bound[0]),
            "q_bias": bool(abs(mean[1]) > bias_bound[1]),
            "iq_gain_imbalance": outside(gain, gain_bounds),
            "iq_quadrature_departure": outside(None if theta is None else float(np.sin(theta)), null_corr_bounds),
        },
        "quadrature": {"valid_lines": len(correlations),
                       "excluded_constant_lines": excluded_constant,
                       "excluded_perfect_correlation_lines": excluded_perfect,
                       "fisher_z_mean": z_mean, "fisher_z_std": z_std,
                       "status": "valid" if theta is not None else "insufficient_valid_lines"},
    }


def correction_parameters(
    analysis, *, correct_iq_bias, correct_iq_gain, correct_iq_quadrature,
    predefined_i_bias=None, predefined_q_bias=None,
    predefined_gain_imbalance=None, predefined_quadrature_rad=None,
):
    """Select explicitly supplied predefined statistics or measured ones, then apply AUX flags.

    Significance is reported separately: it does not silently replace statistics.
    No AUX default is invented when the supplied AUX has none.
    """
    flags = (correct_iq_bias, correct_iq_gain, correct_iq_quadrature)
    if any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
        raise ValueError("Cờ correction phải là boolean.")
    measured = (analysis["mean_i"], analysis["mean_q"], analysis["iq_gain_imbalance"],
                analysis["iq_quadrature_departure_rad"])
    predefined = (predefined_i_bias, predefined_q_bias, predefined_gain_imbalance, predefined_quadrature_rad)
    chosen = [m if p is None else p for m, p in zip(measured, predefined)]
    enabled = (correct_iq_bias, correct_iq_bias, correct_iq_gain, correct_iq_quadrature)
    for index, (value, active) in enumerate(zip(chosen, enabled)):
        if value is not None:
            if np.asarray(value).ndim != 0 or not np.isrealobj(value):
                raise ValueError("Statistic correction phải là scalar thực.")
            chosen[index] = value = float(value)
        if (active and value is None) or (value is not None and (not np.isfinite(value)
                or index == 2 and value <= 0 or index == 3 and abs(value) >= np.pi/2)):
            raise ValueError("Statistic correction không hợp lệ; cần giá trị predefined tường minh nếu phép đo không xác định.")
    names = ("i_bias", "q_bias", "gain_imbalance", "quadrature_rad")
    return {"selected_statistics": dict(zip(names, chosen)),
            "sources": dict(zip(names, ["measured" if p is None else "predefined" for p in predefined])),
            "applied_parameters": dict(zip(names, [v if on else neutral for v, on, neutral in
                                                     zip(chosen, enabled, (0., 0., 1., 0.))])),
            "enabled": dict(zip(("iq_bias", "iq_gain", "iq_quadrature"), map(bool, flags)))}


__all__ = ["analyze_batches", "correction_parameters"]
