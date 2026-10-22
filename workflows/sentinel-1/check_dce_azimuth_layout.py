"""Đối chiếu giả thuyết lịch DCE với slice context trong annotation Stripmap."""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


def candidate_starts(duration_s, block_duration_s, *, nominal_spacing_s):
    """Suy luận nghiên cứu: chia sensing span rồi thêm block cuối đủ support.

    duration_s là last-line minus first-line, không cộng PRI. Thời điểm
    block cuối được caller tính riêng theo biên exclusive của dữ liệu.
    """
    if not all(math.isfinite(x) and x > 0 for x in
               (duration_s, block_duration_s, nominal_spacing_s)):
        raise ValueError("Duration, block duration và spacing phải hữu hạn, dương.")
    divisions = max(1, math.floor(duration_s / nominal_spacing_s))
    return [j * duration_s / divisions for j in range(divisions)
            if j * duration_s / divisions + block_duration_s <= duration_s]


def compare_starts(predicted, observed, *, tolerance_s):
    """Kiểm cả số lượng lẫn vị trí, tránh bỏ qua estimate thừa/thiếu."""
    same_count = len(predicted) == len(observed)
    residuals = [a - b for a, b in zip(predicted, observed)] if same_count else None
    maximum = max(map(abs, residuals)) if residuals else None
    return {"predicted_starts_s": predicted, "count_matches": same_count,
            "residuals_s": residuals, "max_abs_residual_s": maximum,
            "matches": same_count and maximum is not None and maximum <= tolerance_s}


def analyze_annotation(path, *, azimuth_block_size_lines, nominal_spacing_s,
                       time_tolerance_s, tail_stop_convention="after_last_line"):
    """Chỉ dùng timing/layout metadata, không đọc giá trị Doppler để fit lịch."""
    if (azimuth_block_size_lines <= 1 or not math.isfinite(time_tolerance_s)
            or time_tolerance_s <= 0):
        raise ValueError("Block size phải >1 và tolerance phải dương.")
    if tail_stop_convention not in ("last_line", "after_last_line"):
        raise ValueError("Tail convention phải là last_line hoặc after_last_line.")
    root = ET.parse(path).getroot()

    def required(parent, name):
        value = parent.findtext(name)
        if value is None:
            raise ValueError(f"{path}: thiếu {name}.")
        return value

    def time(parent, name):
        return datetime.fromisoformat(required(parent, name).removesuffix("Z"))

    swath = required(root, "adsHeader/swath")
    if swath not in {f"S{i}" for i in range(1, 7)}:
        raise ValueError(f"{path}: chỉ hỗ trợ Stripmap S1–S6.")
    downlinks = root.findall("generalAnnotation/downlinkInformationList/downlinkInformation")
    slices = root.findall("imageAnnotation/imageInformation/sliceList/slice")
    records = root.findall("dopplerCentroid/dcEstimateList/dcEstimate")
    if len(downlinks) != 1 or not slices or not records:
        raise ValueError(f"{path}: cần một downlink, slice context và DC estimate.")
    slice_numbers = [int(required(s, "sliceNumber")) for s in slices]
    slice_starts = [time(s, "sensingStartTime") for s in slices]
    slice_stops = [time(s, "sensingStopTime") for s in slices]
    current_number = int(required(root, "imageAnnotation/imageInformation/sliceNumber"))
    if (slice_numbers != sorted(set(slice_numbers))
            or slice_starts != sorted(set(slice_starts))
            or any(b <= a for a, b in zip(slice_starts, slice_stops))
            or current_number not in slice_numbers):
        raise ValueError(f"{path}: slice context không hợp lệ.")
    current = slice_numbers.index(current_number)
    if (current + 1 < len(slices)
            and slice_numbers[current + 1] != current_number + 1):
        raise ValueError(f"{path}: thiếu slice kế tiếp trong context.")
    dl = downlinks[0]
    pri = float(required(dl, "downlinkValues/pri"))
    offset = float(required(root, "imageAnnotation/imageInformation/zeroDopMinusAcqTime"))
    if not math.isfinite(pri) or pri <= 0 or not math.isfinite(offset):
        raise ValueError(f"{path}: PRI hoặc time offset không hợp lệ.")
    first = time(dl, "firstLineSensingTime")
    last = time(dl, "lastLineSensingTime")
    duration = (last - first).total_seconds()
    if (abs((slice_starts[current] - first).total_seconds()) > time_tolerance_s
            or abs((slice_stops[current] - last).total_seconds()) > time_tolerance_s):
        raise ValueError(f"{path}: slice và downlink có extent khác nhau.")
    line_count = round(duration / pri) + 1
    if abs(duration - (line_count - 1) * pri) > time_tolerance_s:
        raise ValueError(f"{path}: sensing span không khớp lưới PRI.")
    if line_count < azimuth_block_size_lines:
        raise ValueError(f"{path}: thiếu support cho một block DCE.")
    block_duration = azimuth_block_size_lines * pri
    regular = candidate_starts(duration, block_duration, nominal_spacing_s=nominal_spacing_s)
    tail = (line_count - azimuth_block_size_lines) * pri
    previous_proposal = sorted(set([*regular, tail]))
    has_next = current + 1 < len(slices)
    schedule_span = ((slice_starts[current + 1] - first).total_seconds()
                     if has_next else duration)
    context_regular = candidate_starts(schedule_span, block_duration,
                                       nominal_spacing_s=nominal_spacing_s)
    final_tail = duration - block_duration if tail_stop_convention == "last_line" else tail
    context_end = schedule_span if has_next else final_tail
    proposed = sorted(set([*context_regular, context_end]))
    if (proposed[0] < -time_tolerance_s
            or proposed[-1] + block_duration > duration + pri + time_tolerance_s):
        raise ValueError(f"{path}: lịch giả thuyết vượt support downlink.")
    starts, widths, centers, range_counts, acquisition_windows = [], [], [], [], []
    for record in records:
        start = time(record, "fineDceAzimuthStartTime")
        stop = time(record, "fineDceAzimuthStopTime")
        center = time(record, "azimuthTime")
        starts.append((start - first).total_seconds() - offset)
        widths.append((stop - start).total_seconds() / pri)
        centers.append((center - start).total_seconds() - (stop - start).total_seconds() / 2)
        range_counts.append(len(record.findall("fineDceList/fineDce")))
        acquisition_windows.append({key: (time(record, tag) - timedelta(seconds=offset)).isoformat()
                                    for key, tag in (("start", "fineDceAzimuthStartTime"),
                                                     ("stop", "fineDceAzimuthStopTime"),
                                                     ("center", "azimuthTime"))})
    if starts != sorted(set(starts)) or any(w <= 0 for w in widths):
        raise ValueError(f"{path}: timing DCE không tăng hoặc width không dương.")
    fixed_lines = [i * pri for i in range(0, line_count - azimuth_block_size_lines + 1, 3000)]
    fixed_time = [i * nominal_spacing_s for i in range(math.floor(tail / nominal_spacing_s) + 1)]
    midpoint = [0.0] + ([duration / 2] if duration / 2 <= tail else [])
    models = {name: compare_starts(sorted(set([*values, tail])), starts,
                                  tolerance_s=time_tolerance_s)
              for name, values in {"adaptive_span": previous_proposal, "fixed_3000_lines": fixed_lines,
                                   "fixed_nominal_seconds": fixed_time,
                                   "single_slice_midpoint": midpoint}.items()}
    # Các control model phía trên vẫn dùng tail cũ; model slice context dùng biên riêng.
    models["slice_context"] = compare_starts(proposed, starts, tolerance_s=time_tolerance_s)
    hypothesis = models["slice_context"]
    boundary_error = None
    if hypothesis["count_matches"]:
        # Start, stop và center được kiểm riêng; offset không được fit theo annotation.
        errors = []
        for predicted, record in zip(proposed, records):
            for tag, relative in (("fineDceAzimuthStartTime", predicted),
                                  ("fineDceAzimuthStopTime", predicted + block_duration),
                                  ("azimuthTime", predicted + block_duration / 2)):
                errors.append((time(record, tag) - first).total_seconds() - offset - relative)
        boundary_error = max(map(abs, errors))
    return {"file": str(Path(path).resolve()),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "swath": swath, "polarisation": required(root, "adsHeader/polarisation"),
            "mission": root.findtext("adsHeader/missionId"),
            "absolute_orbit": root.findtext("adsHeader/absoluteOrbitNumber"),
            "datatake_id": root.findtext("adsHeader/missionDataTakeId"),
            "slice_number": current_number, "slice_count": len(slices),
            "slice_context": [{"number": n, "start": a.isoformat(), "stop": b.isoformat()}
                              for n, a, b in zip(slice_numbers, slice_starts, slice_stops)],
            "schedule_span_s": schedule_span, "has_next_slice": has_next,
            "tail_stop_convention": tail_stop_convention,
            "sensing_start": first.isoformat(), "sensing_stop": last.isoformat(),
            "sensing_span_s": duration, "pri_s": pri, "inferred_line_count": line_count,
            "block_widths_lines": widths, "range_counts": range_counts,
            "observed_starts_s": starts, "observed_start_offsets_lines": [s / pri for s in starts],
            "stride_s": [b - a for a, b in zip(starts, starts[1:])],
            "acquisition_windows": acquisition_windows,
            "divisions": max(1, math.floor(schedule_span / nominal_spacing_s)),
            "max_start_stop_center_error_s": boundary_error,
            "matches": (hypothesis["matches"] and boundary_error is not None
                        and boundary_error <= time_tolerance_s),
            "models": models}


def shared_windows(results, *, time_tolerance_s):
    """So cửa sổ acquisition giữa các sản phẩm kề nhau, không so timestamp ZD trực tiếp."""
    found = []
    for left in results:
        for right in results:
            identity = ("mission", "datatake_id", "swath", "polarisation")
            if (any(left[k] is None or left[k] != right[k] for k in identity)
                    or right["slice_number"] != left["slice_number"] + 1
                    or left['slice_context'][0]['start'] != right['slice_context'][0]['start']):
                continue
            if left["slice_context"] != right["slice_context"]:
                raise ValueError("Các sản phẩm cùng datatake có slice context khác nhau.")
            for i, a in enumerate(left["acquisition_windows"]):
                for j, b in enumerate(right["acquisition_windows"]):
                    error = max(abs((datetime.fromisoformat(a[k]) - datetime.fromisoformat(b[k]))
                                    .total_seconds()) for k in ("start", "stop", "center"))
                    if error <= time_tolerance_s:
                        found.append({"left": left["file"], "right": right["file"],
                                      "left_record": i, "right_record": j,
                                      "acquisition_window": a, "max_time_error_s": error})
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home() / "sar_data/focusing_data"
    parser.add_argument("inputs", nargs="*", type=Path,
                        default=[base / "l1_sm_annotation_metadata", base / "product_1/level-1"])
    parser.add_argument("--azimuth-block-size-lines", type=int, default=6000)
    parser.add_argument("--nominal-spacing-s", type=float, default=10.0)
    parser.add_argument("--time-tolerance-s", type=float, default=5e-6)
    parser.add_argument("--tail-stop-convention", choices=("last_line", "after_last_line"),
                        default="after_last_line")
    parser.add_argument("--output", type=Path, default=Path("output/sentinel-1/dce_azimuth_layout.json"))
    args = parser.parse_args()
    paths = set()
    for item in args.inputs:
        if not item.exists():
            parser.error(f"Không tìm thấy input: {item}")
        paths.update(item.rglob("*.xml") if item.is_dir() else [item])
    annotations = [p for p in sorted(paths) if ET.parse(p).getroot().find("dopplerCentroid") is not None]
    if not annotations:
        parser.error("Không tìm thấy annotation chứa dopplerCentroid.")
    results = [analyze_annotation(p, azimuth_block_size_lines=args.azimuth_block_size_lines,
                                 nominal_spacing_s=args.nominal_spacing_s,
                                 time_tolerance_s=args.time_tolerance_s,
                                 tail_stop_convention=args.tail_stop_convention) for p in annotations]
    report = {"status": "Giả thuyết từ annotation, chưa xác nhận cấu hình IPF",
              "config": {"azimuth_block_size_lines": args.azimuth_block_size_lines,
                         "nominal_spacing_s": args.nominal_spacing_s,
                         "tail_stop_convention": args.tail_stop_convention,
                         "time_tolerance_s": args.time_tolerance_s},
              "annotations": results,
              "shared_acquisition_windows": shared_windows(results, time_tolerance_s=args.time_tolerance_s)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for row in results:
        error = row['max_start_stop_center_error_s']
        error_text = f"{error * 1e6:.3f} µs" if error is not None else "khác số estimate"
        print(f"{Path(row['file']).name}: slice={row['slice_number']}/{row['slice_count']}, "
              f"K={row['divisions']}, n={len(row['observed_starts_s'])}, "
              f"sai số={error_text}, khớp={row['matches']}")
    if not all(row["matches"] for row in results):
        raise SystemExit("Có annotation không khớp giả thuyết; xem báo cáo JSON.")


if __name__ == "__main__":
    main()
