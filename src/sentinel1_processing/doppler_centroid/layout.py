"""Lịch DCE theo acquisition context; khoảng H/K là suy luận từ metadata."""

import numpy as np


def context_starts(*, slice_start_times_s, last_slice_stop_time_s, current_slice_index,
                   block_size_lines, prf_hz, nominal_spacing_s, tail_stop_convention):
    """Trả start theo giây trong cùng epoch input; stop là biên B dòng.

    `last_line` coi mốc context là biên stop; `after_last_line` thêm một PRI
    sau dòng cuối. Không tự chọn convention từ satellite hoặc L1 residual.
    """
    starts = np.asarray(slice_start_times_s, dtype=float)
    if (starts.ndim != 1 or not starts.size or not np.isfinite(starts).all()
            or np.any(np.diff(starts) <= 0)):
        raise ValueError("Slice starts phải hữu hạn, tăng nghiêm ngặt.")
    if (isinstance(current_slice_index, bool) or not isinstance(current_slice_index, (int, np.integer))
            or not 0 <= current_slice_index < starts.size):
        raise ValueError("current_slice_index nằm ngoài context.")
    if (isinstance(block_size_lines, bool) or not isinstance(block_size_lines, (int, np.integer))
            or block_size_lines <= 1 or not np.isfinite(prf_hz) or prf_hz <= 0
            or not np.isfinite(nominal_spacing_s) or nominal_spacing_s <= 0):
        raise ValueError("Block size, PRF và nominal spacing không hợp lệ.")
    end = float(last_slice_stop_time_s)
    if not np.isfinite(end) or end <= starts[-1]:
        raise ValueError("Last slice stop phải sau last slice start.")
    if tail_stop_convention not in ("last_line", "after_last_line"):
        raise ValueError("Cần tail_stop_convention tường minh.")
    # Work relative to the first start; avoid floor/round on large GPS epochs.
    origin = starts[0]
    relative = starts-origin
    index = current_slice_index
    a = relative[index]
    final = index == len(starts)-1
    stop = end-origin if final else relative[index+1]
    span = stop-a
    duration = block_size_lines/prf_hz
    if final:
        tail = (stop-duration if tail_stop_convention == "last_line" else
                a+(round(span*prf_hz)+1-block_size_lines)/prf_hz)
        if tail < a:
            raise ValueError("Slice cuối ngắn hơn một block DCE.")
    else:
        tail = stop
    count = max(1, int(np.floor(span/nominal_spacing_s)))
    result = sorted(set([*(a+j*span/count for j in range(count)), tail]))
    return np.asarray(result)+origin


def map_starts(start_times_s, line_times_s, *, block_size_lines, prf_hz,
               mapping_tolerance_s, continuity_tolerance_s, missing_support_policy):
    """Giữ báo cáo mọi cửa sổ; không clamp, extrapolate hoặc lấp gap bằng zero."""
    times = np.asarray(line_times_s, dtype=float)
    requested = np.asarray(start_times_s, dtype=float)
    if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or requested.ndim != 1 or not requested.size
            or not np.isfinite(requested).all() or np.any(np.diff(requested) <= 0)):
        raise ValueError("Line times và requested starts phải hữu hạn, tăng nghiêm ngặt.")
    if (not isinstance(block_size_lines, (int, np.integer)) or isinstance(block_size_lines, bool)
            or block_size_lines <= 1 or not np.isfinite(prf_hz) or prf_hz <= 0
            or not np.isfinite(mapping_tolerance_s) or mapping_tolerance_s < 0
            or not np.isfinite(continuity_tolerance_s) or continuity_tolerance_s < 0
            or mapping_tolerance_s >= 0.5/prf_hz or continuity_tolerance_s >= 0.5/prf_hz):
        raise ValueError("Block/PRF/tolerance không hợp lệ; tolerance phải nhỏ hơn nửa PRI.")
    if missing_support_policy not in ("error", "report"):
        raise ValueError("missing_support_policy phải là error hoặc report.")
    origin = times[0]
    relative = times-origin
    reports = []
    previous_index = None
    for desired in requested:
        delta = desired-origin
        index = int(np.argmin(np.abs(relative-delta)))
        error = float(relative[index]-delta)
        reason = None
        # Outside the declared input interval is missing support, not nearest-line rounding.
        if delta < -mapping_tolerance_s or delta > relative[-1]+mapping_tolerance_s:
            reason = "start_outside_input"
        elif abs(error) > mapping_tolerance_s:
            raise ValueError("Context start không khớp line grid trong mapping tolerance.")
        elif index == previous_index:
            raise ValueError("Hai context windows ánh xạ tới cùng một dòng.")
        elif index+block_size_lines > len(times):
            reason = "incomplete_block"
        elif np.any(np.abs(np.diff(relative[index:index+block_size_lines])-1/prf_hz) > continuity_tolerance_s):
            reason = "packet_gap_or_pri_change"
        stop = index+block_size_lines
        reports.append({"requested_start_time_s": float(desired), "start_line": index if reason is None else None,
            "stop_line_exclusive": stop if reason is None else None,
            "start_time_s": float(times[index]) if reason is None else None,
            "stop_time_s": float(times[index]+block_size_lines/prf_hz) if reason is None else None,
            "center_time_s": float(times[index]+block_size_lines/(2*prf_hz)) if reason is None else None,
            "mapping_error_s": error, "available": reason is None, "unavailable_reason": reason,
            "window_id": f"{float(desired).hex()}:{block_size_lines}:{float(prf_hz).hex()}"})
        if reason is not None and missing_support_policy == "error":
            raise ValueError(f"DCE window thiếu support: {reason}.")
        previous_index = index
    return reports
