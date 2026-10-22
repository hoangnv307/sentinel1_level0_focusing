"""Range Dependent Gain Correction from DAD Section 6.1.2."""

import numpy as np


def apply(range_compressed, gain):
    """Multiply each range sample by its gain value."""
    gain = np.asarray(gain)
    if gain.ndim != 1 or gain.size != range_compressed.shape[1]:
        raise ValueError("gain must contain one value per range sample.")
    range_compressed *= gain[None, :]
    return range_compressed


__all__ = ["apply", "eap_correction", "rsl_correction", "bank_times", "interpolate_bank", "build_eap_bank"]


def eap_correction(elevation_angles_deg, angular_grid_deg, complex_lut, *, minimum_magnitude):
    """DAD §9.5: nội suy complex LUT rồi sqrt(|EAP|)/EAP, giữ nguyên phase."""
    grid=np.asarray(angular_grid_deg,dtype=float)
    lut=np.asarray(complex_lut,dtype=complex)
    angles=np.asarray(elevation_angles_deg,dtype=float)
    if (grid.ndim!=1 or grid.size<2 or lut.shape!=grid.shape or not np.isfinite(grid).all()
            or not np.isfinite(lut).all() or np.any(np.diff(grid)<=0)
            or not np.isfinite(angles).all() or not np.isfinite(minimum_magnitude) or minimum_magnitude<0):
        raise ValueError('EAP grid/LUT/angles/floor không hợp lệ.')
    if np.any(angles<grid[0]) or np.any(angles>grid[-1]):
        raise ValueError('Elevation angle ngoài support AUX_CAL LUT.')
    value=np.interp(angles,grid,lut)
    if np.any(abs(value)<=minimum_magnitude):
        raise ValueError('EAP bằng zero hoặc dưới minimum_magnitude.')
    return np.sqrt(abs(value))/value


def rsl_correction(slant_range_m, *, reference_range_m):
    """DAD §9.6: amplitude gain, không dùng power gain mũ 3 cho I/Q."""
    ranges=np.asarray(slant_range_m,dtype=float)
    if (not np.isfinite(ranges).all() or np.any(ranges<=0)
            or not np.isfinite(reference_range_m) or reference_range_m<=0):
        raise ValueError('RSL ranges/reference range phải hữu hạn, dương.')
    with np.errstate(over='ignore',invalid='ignore'):
        correction=(ranges/reference_range_m)**1.5
    if not np.isfinite(correction).all():
        raise ValueError('RSL amplitude correction không hữu hạn.')
    return correction


def bank_times(start_time_s, stop_time_s, *, update_interval_s):
    if (not np.isfinite([start_time_s,stop_time_s,update_interval_s]).all()
            or update_interval_s<=0 or stop_time_s<=start_time_s):
        raise ValueError('Bank time interval không hợp lệ.')
    # Always include both support endpoints; never extrapolate a bank at the last echo.
    relative=np.arange(0.,stop_time_s-start_time_s,update_interval_s)
    return np.r_[relative,stop_time_s-start_time_s]+start_time_s


def interpolate_bank(query_times_s, node_times_s, complex_vectors):
    times=np.asarray(node_times_s,dtype=float);query=np.asarray(query_times_s,dtype=float)
    bank=np.asarray(complex_vectors)
    if (times.ndim!=1 or times.size<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or query.ndim!=1 or not np.isfinite(query).all() or bank.ndim!=2 or bank.shape[0]!=times.size
            or not np.isfinite(bank).all()):
        raise ValueError('EAP bank không hợp lệ.')
    if np.any(query<times[0]) or np.any(query>times[-1]):
        raise ValueError('Query ngoài time support của EAP bank.')
    high=np.clip(np.searchsorted(times,query,side='right'),1,times.size-1)
    low=high-1
    fraction=(query-times[low])/(times[high]-times[low])
    return bank[low]*(1-fraction[:,None])+bank[high]*fraction[:,None]


def build_eap_bank(node_times_s, slant_range_m, *, look_angle_provider, boresight_angle_provider,
                   angular_grid_deg, complex_lut, minimum_magnitude):
    """Provider tính geometry/terrain độc lập; LUT và mọi tolerance là input."""
    times=np.asarray(node_times_s,dtype=float);ranges=np.asarray(slant_range_m,dtype=float)
    if (times.ndim!=1 or times.size<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or ranges.ndim!=1 or not ranges.size or not np.isfinite(ranges).all() or np.any(ranges<=0)):
        raise ValueError('EAP bank time/range grid không hợp lệ.')
    vectors=[]
    for time in times:
        look=np.asarray(look_angle_provider(float(time),ranges),dtype=float)
        boresight=boresight_angle_provider(float(time))
        if look.shape!=ranges.shape or np.asarray(boresight).ndim!=0:
            raise ValueError('EAP geometry provider trả shape không hợp lệ.')
        vectors.append(eap_correction(look-boresight,angular_grid_deg,complex_lut,
                                      minimum_magnitude=minimum_magnitude))
    return np.asarray(vectors)
