"""Raw-data corrections from Sentinel-1 L1 DAD Section 9.2."""

import numpy as np


def estimate_iq_bias(radar_data):
    """Estimate the constant I/Q mean used by DAD Section 9.2.1."""
    data = np.asarray(radar_data)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("radar_data must be a non-empty 2-D array.")
    return complex(np.mean(data, dtype=np.complex128))


def apply(
    radar_data, *, i_bias, q_bias, gain_imbalance, quadrature_rad,
    correct_iq_bias, correct_iq_gain, correct_iq_quadrature,
):
    """Correct decoded complex samples in DAD §9.2.1 order, without mutating input.

    Bias inputs are measured means (§4.1), so removal subtracts them despite
    the printed plus signs in Eq. 9-1/9-2. Gain is sigma_I/sigma_Q;
    theta is radians: Q_out = rho*(Q-mu_Q)/cos(theta) - (I-mu_I)*tan(theta).
    Disabled steps use identity and do not require a defined statistic.
    """
    data = np.asarray(radar_data)
    if (data.ndim != 2 or not data.size or not np.iscomplexobj(data)
            or not np.isfinite(data).all()):
        raise ValueError("radar_data phải là ma trận complex 2D hữu hạn, không rỗng.")
    flags = (correct_iq_bias, correct_iq_gain, correct_iq_quadrature)
    if any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
        raise ValueError("Cờ correction phải là boolean.")
    values = (i_bias, q_bias, gain_imbalance, quadrature_rad)
    enabled = (correct_iq_bias, correct_iq_bias, correct_iq_gain, correct_iq_quadrature)
    for index, (value, active) in enumerate(zip(values, enabled)):
        if value is None and not active:
            continue
        if (value is None or np.asarray(value).ndim != 0 or np.asarray(value).dtype.kind not in "fiu"
                or not np.isfinite(value) or index == 2 and value <= 0
                or index == 3 and abs(value) >= np.pi / 2):
            raise ValueError("Statistic I/Q correction không hợp lệ (gain > 0, |theta| < pi/2 radians).")
    corrected = data.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        if correct_iq_bias:
            corrected.real -= i_bias
            corrected.imag -= q_bias
        if correct_iq_gain:
            corrected.imag *= gain_imbalance
        if correct_iq_quadrature:
            corrected.imag /= np.cos(quadrature_rad)
            corrected.imag -= corrected.real * np.tan(quadrature_rad)
    if not np.isfinite(corrected).all():
        raise ValueError("I/Q correction tạo giá trị không hữu hạn.")
    return corrected


__all__ = ["estimate_iq_bias", "apply"]
