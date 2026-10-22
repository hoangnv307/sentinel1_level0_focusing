"""DCE S4/VV từ L0/AUX/job, với nhánh L1 regression được chọn tường minh."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import sentinel1decoder
from sentinel1decoder.enums import RxChannelId, SignalType

from sentinel1_processing.azimuth_pre_processing.range import compression
from sentinel1_processing.common.raw_data_analysis import analyze_batches, correction_parameters
from sentinel1_processing.common.raw_data_correction import apply as correct_raw_iq
from sentinel1_processing.common.receiver_gain_compensation import calculate_gain
from sentinel1_processing.doppler_centroid import estimation, geometry
from sentinel1_processing.range_processing.swst_bias import correct
from sentinel1_processing.range_processing.range_reference_function import create_freq_domain
from sentinel1_processing.utils.dce_reference_input import ReferenceInput
from sentinel1_processing.utils.pg_calibration_input import analyze_calibration
from sentinel1_processing.pre_processing.internal_calibration import interpolate_pg


BATCH_LINES = 256
RAW_ANALYSIS_SELECTION = {
    "azimuth_start_line": 0, "azimuth_step": 1, "max_azimuth_lines": None,
    "range_start_sample": 0, "range_stop_sample": None,
}


def one(items, description):
    items = list(items)
    if len(items) != 1:
        raise ValueError(f"Cần đúng một {description}; tìm thấy {len(items)}.")
    return items[0]


def read_auxiliary(product, selected_paths=None):
    ins_path = (Path(selected_paths["INS"]) if selected_paths is not None else
                one(product.glob("auxiliary/*/data/s1a-aux-ins.xml"), "AUX_INS"))
    pp1_path = (Path(selected_paths["PP1"]) if selected_paths is not None else
                one(product.glob("auxiliary/*/data/s1a-aux-pp1.xml"), "AUX_PP1"))
    ins = ET.parse(ins_path).getroot()
    pp1 = one(
        (p for p in ET.parse(pp1_path).getroot().findall("productList/product")
         if p.findtext("productId") == "SM_SL1__1"),
        "cấu hình AUX_PP1 SM_SL1__1",
    )
    calibration = one(
        (p for p in ins.findall("internalCalibrationParamsList/internalCalibrationParams")
         if p.findtext("swath") == "S4" and p.findtext("polarisation") == "VV"),
        "internal calibration S4/VV",
    )
    if pp1.findtext("dcProcParams/dcInputData") != "Range Compressed":
        raise ValueError("Demo yêu cầu DCE trên Range Compressed.")
    method = pp1.findtext("dcProcParams/dcMethod")
    if method not in ("Data Analysis", "Geometry", "Predefined"):
        raise ValueError("DC method AUX không hỗ trợ.")
    swath = one(
        (e for e in ins.findall("swathParamsList/swathParams") if e.findtext("swath") == "S4"),
        "swath parameters S4",
    )
    receiver = one(
        (e for e in swath.findall("rxVariationCorrectionParamsList/rxVariationCorrectionParams")
         if e.findtext("rxPolarisation") == "V"), "receiver parameters V",
    )
    nominal = one(
        (e for e in swath.findall("pulseParamsList/pulseParams") if e.findtext("polarisation") == "VV"),
        "nominal chirp S4/VV",
    )
    flags = {}
    for name, field in (("correct_iq_bias", "correctIQBiasFlag"),
                        ("correct_iq_gain", "correctIQGainImbalanceFlag"),
                        ("correct_iq_quadrature", "correctIQOrthogonalityFlag"),
                        ("correct_rx_variation", "correctRxVariationFlag")):
        value = pp1.findtext(f"commonProcParams/{field}")
        if value not in ("true", "false"):
            raise ValueError(f"Thiếu cờ cấu hình {field}.")
        flags[name] = value == "true"
    for key, path in (("apply_eap","slcProcParams/applyElevationAntennaPatternFlag"),
                      ("apply_rsl","slcProcParams/applyRangeSpreadingLossFlag"),
                      ("perform_internal_calibration","preProcParams/performInternalCalibrationFlag")):
        value=pp1.findtext(path)
        if value not in ("true","false"):
            raise ValueError(f"Thiếu hoặc sai cờ AUX {path}.")
        flags[key]=value=="true"
    return {
        "ellipsoid_a_m": float(pp1.findtext("commonProcParams/ellipsoidParams/ellipsoidSemiMajorAxis")),
        "ellipsoid_b_m": float(pp1.findtext("commonProcParams/ellipsoidParams/ellipsoidSemiMinorAxis")),
        "predefined_coefficients": ([float(v) for v in pp1.findtext("dcProcParams/dcPredefinedCoefficients").split()]
            if pp1.findtext("dcProcParams/dcPredefinedCoefficients") is not None else None),
        "dc_method": method,
        "rrf_spectrum": pp1.findtext("slcProcParams/rrfSpectrum"),
        "radar_frequency_hz": float(ins.findtext("radarFrequency")),
        "swst_bias_s": float(calibration.findtext("swstBias")),
        "rms_threshold_hz": float(pp1.findtext("dcProcParams/dcRmsErrorThreshold")),
        **flags,
        "rx_gain_trend": [float(x) for x in receiver.findtext("gainTrendCoefficients").split()],
        "rx_gain_overshoot": [float(x) for x in receiver.findtext("gainOvershootCoefficients").split()],
        "nominal_amplitude_coefficients": [float(x) for x in nominal.findtext("amplitudeCoefficients").split()],
        "nominal_phase_coefficients_cycles": [float(x) for x in nominal.findtext("phaseCoefficients").split()],
        "files": [str(ins_path), str(pp1_path)],
    }


def analyze_echoes(
    level0, *, batch_lines, azimuth_start_line, azimuth_step, max_azimuth_lines,
    range_start_sample, range_stop_sample, iq_correction=None,
):
    """Select valid S4/VV echoes in acquisition order; one statistics scope across chunks."""
    positive = (batch_lines, azimuth_step)
    nonnegative = (azimuth_start_line, range_start_sample)
    if (any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v <= 0 for v in positive)
            or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0 for v in nonnegative)
            or max_azimuth_lines is not None and (not isinstance(max_azimuth_lines, (int, np.integer))
                                                or isinstance(max_azimuth_lines, bool) or max_azimuth_lines <= 0)
            or range_stop_sample is not None and (not isinstance(range_stop_sample, (int, np.integer))
                                                 or isinstance(range_stop_sample, bool) or range_stop_sample <= range_start_sample)):
        raise ValueError("Cấu hình chọn mẫu raw analysis không hợp lệ.")
    decoder = sentinel1decoder.Level0Decoder(level0.filename)
    sources = []
    eligible = selected = flagged = 0
    first = None

    def batches():
        nonlocal eligible, selected, flagged, first
        for chunk in level0.acquisition_chunks:
            metadata = level0.get_acquisition_chunk_metadata(chunk)
            if not (metadata["Signal Type"] == SignalType.ECHO).any():
                continue
            if not (metadata["Signal Type"] == SignalType.ECHO).all():
                raise ValueError("Echo chunk chứa loại signal khác nhau.")
            if (not metadata.index.is_unique or not metadata.index.is_monotonic_increasing
                    or not (metadata["Swath Number"] == 3).all()
                    or not (metadata["Rx Channel ID"] == RxChannelId.RXV_POL_CHANNEL).all()):
                raise ValueError("Raw analysis yêu cầu echo S4/VV theo thứ tự packet, không lặp.")
            flagged += int(metadata["Error Flag"].sum())
            valid = metadata.loc[~metadata["Error Flag"].astype(bool)]
            if valid.empty:
                continue
            if first is None:
                first = valid.iloc[0]
            for field in ("Rx Gain", "Range Decimation", "Number of Quads", "Instrument Configuration ID"):
                if not (valid[field] == first[field]).all():
                    raise ValueError(f"{field} thay đổi; cần chia raw analysis thành scope riêng.")
            indices = np.arange(eligible, eligible + len(valid))
            chosen = (indices >= azimuth_start_line) & ((indices-azimuth_start_line) % azimuth_step == 0)
            eligible += len(valid)
            chosen_metadata = valid.iloc[np.flatnonzero(chosen)]
            if max_azimuth_lines is not None:
                chosen_metadata = chosen_metadata.iloc[:max(0, max_azimuth_lines-selected)]
            if chosen_metadata.empty:
                continue
            width = 2*int(first["Number of Quads"])
            stop = width if range_stop_sample is None else range_stop_sample
            if stop > width or stop-range_start_sample < 2:
                raise ValueError("Range ROI raw analysis nằm ngoài echo hoặc không đủ 2 mẫu.")
            sources.append({"chunk": int(chunk), "line_count": len(chosen_metadata),
                            "first_packet": int(chosen_metadata.index[0]),
                            "last_packet": int(chosen_metadata.index[-1])})
            selected += len(chosen_metadata)
            for start in range(0, len(chosen_metadata), batch_lines):
                headers = chosen_metadata.iloc[start:start+batch_lines]
                raw = decoder.decode_packets(headers)
                if raw.shape != (len(headers), width):
                    raise ValueError("Shape decoded echo không khớp header.")
                raw = raw[:, range_start_sample:stop]
                if iq_correction is not None:
                    raw = correct_raw_iq(raw, **iq_correction["applied_parameters"], **{
                        "correct_iq_bias": iq_correction["enabled"]["iq_bias"],
                        "correct_iq_gain": iq_correction["enabled"]["iq_gain"],
                        "correct_iq_quadrature": iq_correction["enabled"]["iq_quadrature"],
                    })
                yield raw

    statistics = analyze_batches(batches())
    return {**statistics, "source_file": str(level0.filename), "scope": "valid echo S4/VV across acquisition chunks",
            "selection": {"batch_lines": int(batch_lines), "azimuth_start_line": int(azimuth_start_line),
                          "azimuth_step": int(azimuth_step), "max_azimuth_lines": max_azimuth_lines,
                          "range_start_sample": int(range_start_sample), "range_stop_sample": range_stop_sample},
            "sources": sources, "eligible_echo_lines": eligible, "excluded_error_flag_lines": flagged}


def compress_echoes(
    level0, aux, directory, reference_input=None, *, raw_analysis=None, batch_lines=BATCH_LINES,
    correct_iq_bias=None, correct_iq_gain=None, correct_iq_quadrature=None,
    predefined_i_bias=None, predefined_q_bias=None,
    predefined_gain_imbalance=None, predefined_quadrature_rad=None,
    correct_rx_variation=None,
    pg_correction_provider=None,
    range_gain_factory=None, range_reference_factory=None, continuity_tolerance_s=None,
    internal_delay_s=None, minimum_far_range_discard_samples=0,
):
    """Decode/compress theo batch vào memmap; không giữ toàn raw scene trong RAM."""
    if not isinstance(batch_lines, (int, np.integer)) or isinstance(batch_lines, bool) or batch_lines <= 0:
        raise ValueError("batch_lines phải là số nguyên dương.")
    delay = (0. if reference_input is None else reference_input.internal_delay_s) if internal_delay_s is None else internal_delay_s
    if (np.asarray(delay).ndim != 0 or np.asarray(delay).dtype.kind not in "fiu"
            or not np.isfinite(delay) or delay < 0):
        raise ValueError("internal_delay_s phải là scalar thực hữu hạn, không âm.")
    if (not isinstance(minimum_far_range_discard_samples, (int, np.integer))
            or isinstance(minimum_far_range_discard_samples, bool) or minimum_far_range_discard_samples < 0):
        raise ValueError("minimum_far_range_discard_samples phải là số nguyên không âm.")
    if reference_input is not None and (range_gain_factory is not None or range_reference_factory is not None):
        raise ValueError("Không áp chồng L1 reference corrections và independent factories.")
    chunks = []
    for chunk in level0.acquisition_chunks:
        metadata = level0.get_acquisition_chunk_metadata(chunk)
        if metadata["Signal Type"].iloc[0] == SignalType.ECHO:
            chunks.append((chunk, metadata))
    if not chunks:
        raise ValueError("L0 không có echo chunk.")
    flags = {name: aux[name] if value is None else value for name, value in (
        ("correct_iq_bias", correct_iq_bias), ("correct_iq_gain", correct_iq_gain),
        ("correct_iq_quadrature", correct_iq_quadrature))}
    use_reference_bias = raw_analysis is None and reference_input is not None
    if raw_analysis is None and (reference_input is None or flags["correct_iq_gain"] or flags["correct_iq_quadrature"]):
        raw_analysis = analyze_echoes(level0, batch_lines=batch_lines, **RAW_ANALYSIS_SELECTION)
    if raw_analysis is not None and raw_analysis.get("source_file") != str(level0.filename):
        raise ValueError("Raw analysis không thuộc file L0 đang xử lý.")
    bias_source = "L0 raw analysis, one acquisition scope"
    if predefined_i_bias is not None or predefined_q_bias is not None:
        bias_source = "explicit predefined bias; see per-component sources"
    elif use_reference_bias:
        bias_source = "L1 support/*BiasUsedForCorrection"
    if use_reference_bias:
        if predefined_i_bias is None:
            predefined_i_bias = reference_input.bias.real
        if predefined_q_bias is None:
            predefined_q_bias = reference_input.bias.imag
    parameters = correction_parameters(
        raw_analysis if raw_analysis is not None else {
            "mean_i": None, "mean_q": None, "iq_gain_imbalance": None,
            "iq_quadrature_departure_rad": None},
        **flags, predefined_i_bias=predefined_i_bias, predefined_q_bias=predefined_q_bias,
        predefined_gain_imbalance=predefined_gain_imbalance, predefined_quadrature_rad=predefined_quadrature_rad,
    )
    used = parameters["applied_parameters"]
    bias = complex(used["i_bias"], used["q_bias"])
    rx_enabled = aux["correct_rx_variation"] if correct_rx_variation is None else correct_rx_variation
    constant_fields = [
        "PRI", "Range Decimation", "Tx Pulse Length", "Tx Ramp Rate",
        "Tx Pulse Start Frequency", "Rank", "Swath Number", "Rx Channel ID",
    ]
    first = chunks[0][1].iloc[0]
    for chunk, metadata in chunks:
        if metadata["Error Flag"].any():
            raise ValueError("DCE demo chưa hỗ trợ echo bị Error Flag; raw analysis riêng loại các dòng đó.")
        for field in constant_fields:
            if not (metadata[field] == first[field]).all():
                raise ValueError(f"Chunk {chunk}: {field} thay đổi; ngoài phạm vi demo.")
        for field in ("SWST", "SWL", "Number of Quads"):
            if metadata[field].nunique() != 1:
                raise ValueError(f"Chunk {chunk}: {field} phải hằng trong chunk.")
    if int(first["Swath Number"]) != 3 or first["Rx Channel ID"] != RxChannelId.RXV_POL_CHANNEL:
        raise ValueError("Demo chỉ hỗ trợ raw S4/VV hiện tại (swath number 3, RX 0).")
    pri = float(first["PRI"])
    sample_rate = sentinel1decoder.utilities.range_dec_to_sample_rate(first["Range Decimation"])
    pulse_samples = int(np.ceil(float(first["Tx Pulse Length"]) * sample_rate))
    suppressed_time = 320.0 / (8.0 * sentinel1decoder.constants.F_REF)
    origins = [
        float(m["Rank"].iloc[0]) * pri + float(m["SWST"].iloc[0]) + suppressed_time
        for _, m in chunks
    ]
    decoder = sentinel1decoder.Level0Decoder(level0.filename)
    segments, diagnostics = [], []
    for (chunk, metadata), origin in zip(chunks, origins):
        n_range = 2 * int(metadata["Number of Quads"].iloc[0])
        offset = (origin - min(origins)) * sample_rate
        shift = (round(offset) - offset) / sample_rate
        support = compression.slc_support(n_range,sample_rate_hz=sample_rate,
            pulse_length_s=float(first["Tx Pulse Length"]),slant_range_origin_s=origin,swst_bias_s=aux["swst_bias_s"],
            range_time_shift_s=shift,internal_delay_s=delay,
            minimum_far_range_discard_samples=minimum_far_range_discard_samples)
        n_valid=support["n_valid"];tau=support["raw_times_s"];stop_sample=support["stop_sample"]
        far_throwaway=support["far_throwaway"];required_far_discard=support["required_far_discard"]
        eta = metadata["Coarse Time"].to_numpy(float) + metadata["Fine Time"].to_numpy(float)
        tolerance=0.05*pri if continuity_tolerance_s is None else continuity_tolerance_s
        if not np.isfinite(tolerance) or not 0<=tolerance<.5*pri:
            raise ValueError("Continuity tolerance phải hữu hạn và nhỏ hơn nửa PRI.")
        if not np.allclose(np.diff(eta), pri, rtol=0, atol=tolerance):
            raise ValueError(f"Chunk {chunk}: timeline có gap hoặc PRI không đều.")
        data = np.lib.format.open_memmap(
            Path(directory) / f"range-{chunk}.npy", mode="w+", dtype=np.complex64,
            shape=(len(metadata), n_valid),
        )
        range_gain_provider = (None if range_gain_factory is None else range_gain_factory(tau[:stop_sample]+shift))
        rx_gain = calculate_gain(
            n_range, sample_rate_hz=sample_rate, swst_s=float(metadata["SWST"].iloc[0]),
            pulse_length_s=float(first["Tx Pulse Length"]), suppressed_s=suppressed_time,
            gain_trend_coefficients=aux.get("rx_gain_trend"),
            gain_overshoot_coefficients=aux.get("rx_gain_overshoot"), enabled=rx_enabled,
        )
        fft_length = 1 << (n_range + pulse_samples - 2).bit_length()
        bank = None
        if reference_input is not None:
            if internal_delay_s is None:
                reference = reference_input.range_reference(sample_rate, float(first["Tx Pulse Length"]), fft_length)
            else:
                reference = reference_input.range_reference(sample_rate, float(first["Tx Pulse Length"]), fft_length,
                                                            internal_delay_s=delay)
            bank = reference_input.range_gain_bank(tau[:stop_sample] + shift)
        elif range_reference_factory is not None:
            reference = range_reference_factory(sample_rate_hz=sample_rate,
                pulse_length_s=float(first["Tx Pulse Length"]),fft_length=fft_length,internal_delay_s=delay)
        else:
            reference = create_freq_domain(sample_rate_hz=sample_rate,
                pulse_start_frequency_hz=float(first["Tx Pulse Start Frequency"]),
                pulse_ramp_rate_hz_per_s=float(first["Tx Ramp Rate"]),
                pulse_length_s=float(first["Tx Pulse Length"]), fft_length=fft_length, internal_delay_s=delay)
        raw = decoder.decode_packets(metadata.iloc[:batch_lines])
        print(f"Chunk {chunk}: {len(metadata)} echo × {n_range} mẫu, I/Q bias={bias}", flush=True)
        for start in range(0, len(metadata), batch_lines):
            stop = min(start + batch_lines, len(metadata))
            if start:
                raw = decoder.decode_packets(metadata.iloc[start:stop])
            if raw.shape != (stop - start, n_range) or not np.isfinite(raw).all():
                raise ValueError(f"Chunk {chunk}: dữ liệu decode không hợp lệ.")
            raw = correct_raw_iq(raw, **used, **flags)
            with np.errstate(over="ignore", invalid="ignore"):
                if rx_enabled:
                    # Eq. 9-16 uses amplitude sqrt(GV), not the power gain GV.
                    raw /= np.sqrt(rx_gain)[None, :]
                if pg_correction_provider is not None or reference_input is not None:
                    pg = np.asarray((pg_correction_provider if pg_correction_provider is not None else
                                     reference_input.pg_gain)(eta[start:stop]))
                    if pg.shape != (stop-start,) or not np.isfinite(pg).all() or np.any(abs(pg) == 0):
                        raise ValueError("PG correction provider phải trả vector theo echo hữu hạn, khác zero.")
                    raw *= pg[:, None]
            if not np.isfinite(raw).all():
                raise ValueError(f"Chunk {chunk}: dữ liệu sau receiver/PG correction không hữu hạn.")
            _, range_times = compression.compress(
                raw, tau, sample_rate_hz=sample_rate,
                pulse_start_frequency_hz=float(first["Tx Pulse Start Frequency"]),
                pulse_ramp_rate_hz_per_s=float(first["Tx Ramp Rate"]),
                pulse_length_s=float(first["Tx Pulse Length"]),
                iq_bias=0j,
                range_reference_function=reference, range_time_shift_s=shift, output="slc",
                output_array=data[start:stop],
            )
            if range_gain_provider is not None:
                gain = np.asarray(range_gain_provider(eta[start:stop]))
                if gain.shape not in ((stop_sample,),(stop-start,stop_sample)) or not np.isfinite(gain).all():
                    raise ValueError("Range gain provider phải trả vector/matrix hữu hạn trên range support thực.")
                data[start:stop,:stop_sample] *= gain
            if not np.isfinite(data[start:stop,:stop_sample]).all():
                raise ValueError("Input DCE không hữu hạn sau range-dependent gain.")
            if reference_input is not None:
                reference_input.apply_range_gain(data[start:stop, :stop_sample], bank, eta[start:stop])
                if not np.isfinite(data[start:stop, :stop_sample]).all():
                    raise ValueError(f"Chunk {chunk}: input DCE sau hiệu chỉnh không hữu hạn.")
        data.flush()
        segments.append(estimation.Segment(data[:, :stop_sample], range_times[:stop_sample], eta, name=f"chunk-{chunk}"))
        diagnostics.append({
            "chunk": int(chunk), "shape": list(segments[-1].range_compressed.shape),
            "hardware_delay_far_throwaway_samples": far_throwaway,
            "internal_delay_s": float(delay), "required_far_range_discard_samples": required_far_discard,
            "minimum_far_range_discard_samples": int(minimum_far_range_discard_samples),
            "iq_bias": [bias.real, bias.imag],
            "iq_bias_source": bias_source if flags["correct_iq_bias"] else "disabled",
            "iq_analysis_lines": raw_analysis["line_count"] if raw_analysis is not None else 0,
            "raw_data_analysis": raw_analysis if chunk == chunks[0][0] else None,
            "raw_correction_parameters": parameters,
            "pg_source": "caller-supplied correction" if pg_correction_provider is not None else (
                "L1 inverse-PG annotation" if reference_input is not None else "none"),
            "receiver_compensation": {
                "enabled": bool(rx_enabled),
                "flag_source": "AUX_PP1 commonProcParams/correctRxVariationFlag" if correct_rx_variation is None else "explicit override",
                "coefficients_source": "AUX_INS S4/RX V gainTrendCoefficients/gainOvershootCoefficients",
                "swst_downlink_s": float(metadata["SWST"].iloc[0]),
                "pulse_length_s": float(first["Tx Pulse Length"]), "suppressed_time_s": suppressed_time,
                "delta_from_tx_end_s": float(metadata["SWST"].iloc[0]) + suppressed_time - float(first["Tx Pulse Length"]),
                "power_gain_min": float(rx_gain.min()), "power_gain_max": float(rx_gain.max()),
                "power_gain_first": float(rx_gain[0]), "power_gain_last": float(rx_gain[-1]),
                "amplitude_divisor_min": float(np.sqrt(rx_gain.min())),
                "amplitude_divisor_max": float(np.sqrt(rx_gain.max())),
            },
            "range_gain_source": "L0/AUX/job provider" if range_gain_provider is not None else "none",
            "fractional_range_shift_s": shift,
        })
    return segments, pri, sample_rate, diagnostics


def infer_configuration(annotation, segments, pri, sample_rate, rms_threshold, far_range_throwaway_samples=0):
    """Suy layout L1; kiểm chứng trên lưới L0, không dùng tần số DC L1 làm input."""
    header = annotation.find("adsHeader")
    if [header.findtext(k) for k in ("missionId", "swath", "polarisation", "productType")] != [
        "S1A", "S4", "VV", "SLC"
    ]:
        raise ValueError("Annotation phải là S1A/S4/VV/SLC.")
    prepared = estimation.prepare_segments(segments, 1.0 / pri)
    if not np.allclose(np.diff(prepared.azimuth_times_s), pri, rtol=0, atol=0.05 * pri):
        raise ValueError("Demo yêu cầu các echo chunk liên tiếp, không có gap.")
    records = annotation.findall("dopplerCentroid/dcEstimateList/dcEstimate")
    if not records:
        raise ValueError("Annotation thiếu DC estimate.")
    range_start = max(float(s.slant_range_times_s[0]) for s in segments)
    range_stop = min(float(s.slant_range_times_s[-1]) for s in segments) + 1.0 / sample_rate
    roi_samples = int(round((range_stop - range_start) * sample_rate))
    points = records[0].findall("fineDceList/fineDce")
    centres = np.array([float(p.findtext("slantRangeTime")) for p in points])
    # Giả thuyết kiểm được: block đầu bắt đầu ở near range của phần support giao nhau.
    width = int(round(2.0 * (centres[0] - range_start) * sample_rate))
    if not 0 < width <= roi_samples or len(points) < 3:
        raise ValueError("Không suy được range block hợp lệ từ annotation.")
    # Kiểm layout gốc L1; ROI xử lý thực tế có thể hẹp hơn do bù hardware delay.
    annotated_roi_samples = roi_samples + far_range_throwaway_samples
    predicted = np.floor(np.linspace(0, annotated_roi_samples - width, len(points))) + width / 2.0
    offset = float(annotation.findtext("imageAnnotation/imageInformation/zeroDopMinusAcqTime"))
    starts, sizes = [], []
    for record in records:
        times = np.array([float(p.findtext("slantRangeTime"))
                          for p in record.findall("fineDceList/fineDce")])
        if times.shape != centres.shape or not np.allclose(
            (times - range_start) * sample_rate, predicted, rtol=0, atol=0.02
        ):
            raise ValueError("Range centres L1 không khớp layout suy từ support L0.")
        if abs(float(record.findtext("t0")) - range_start) * sample_rate > 0.02:
            raise ValueError("Origin polynomial L1 khác origin suy từ support L0.")
        begin = float(Time(record.findtext("fineDceAzimuthStartTime"), scale="utc").gps) - offset
        end = float(Time(record.findtext("fineDceAzimuthStopTime"), scale="utc").gps) - offset
        count = int(round((end - begin) / pri))
        index = int(np.argmin(np.abs(prepared.azimuth_times_s - begin)))
        if (abs(prepared.azimuth_times_s[index] - begin) > 0.05 * pri
                or abs((end - begin) / pri - count) > 0.05
                or index + count > prepared.num_azimuth_lines):
            raise ValueError("Azimuth support L1 không nằm trên timeline L0.")
        starts.append(index)
        sizes.append(count)
    if len(set(sizes)) != 1 or starts != sorted(set(starts)):
        raise ValueError("Demo yêu cầu DC block cùng chiều dài, tăng theo thời gian.")
    config = estimation.Config(
        azimuth_block_size_lines=sizes[0], num_range_blocks=len(points),
        range_block_size_samples=width, range_roi_stop=roi_samples,
        azimuth_placement="custom", rms_threshold_hz=rms_threshold,
        # Không thể suy duy nhất hai tham số này từ annotation L1.
        unwrap_fft_length=4096, outlier_sigma=3.5,
        fit_weighting="uniform", accc_range_weighting="power",
    )
    return config, starts, range_start, offset


def save_results(output, estimates, reference, provenance, time_offset):
    figure, axes = plt.subplots(len(estimates), 1, figsize=(11, 4 * len(estimates)), squeeze=False)
    from sentinel1_processing.utils.dce_job import real_values
    if len(estimates)!=len(reference):raise ValueError("Regression layout/reference record counts differ.")
    results = []
    for index, (record, annotated) in enumerate(zip(estimates, reference), start=1):
        tau = record.range_times_s
        reference_points = annotated.findall("fineDceList/fineDce")
        reference_tau = np.array([float(p.findtext("slantRangeTime")) for p in reference_points])
        reference_fine = np.array([float(p.findtext("frequency")) for p in reference_points])
        if tau[0] < reference_tau[0] - 1e-10 or tau[-1] > reference_tau[-1] + 1e-10:
            raise ValueError("Không so Fine DC ngoài range support của L1.")
        fine = np.interp(tau, reference_tau, reference_fine)
        coefficients = np.fromstring(annotated.findtext("dataDcPolynomial"), sep=" ")
        l1_fit = np.polynomial.polynomial.polyval(tau - float(annotated.findtext("t0")), coefficients)
        data_fit = record.evaluate(tau)
        geometry_fit = (None if record.geometry_dc_polynomial is None else
                        np.polynomial.polynomial.polyval(tau-record.t0_s,record.geometry_dc_polynomial))
        selected_source = record.selected_source
        fine_rmse=None
        if record.fine_baseband_hz.size==tau.size:
            valid=np.isfinite(record.fine_baseband_hz)
            if valid.any():fine_rmse=float(np.sqrt(np.mean((record.fine_baseband_hz[valid]-fine[valid])**2)))
        fit_rmse = float(np.sqrt(np.mean((data_fit - l1_fit) ** 2)))
        axes[index - 1, 0].plot(tau * 1e3, data_fit, label=f"DC được chọn: {selected_source}")
        if geometry_fit is not None:
            axes[index - 1, 0].plot(tau * 1e3, geometry_fit, label="Geometry DC từ L0", linestyle=":")
        axes[index - 1, 0].plot(tau * 1e3, l1_fit, label="Polynomial L1 tham chiếu", linestyle="--")
        if record.fine_absolute_hz.size:
            axes[index - 1, 0].scatter(tau * 1e3, record.fine_absolute_hz, label="Fine DC L0, đã resolve ambiguity", s=16)
        axes[index - 1, 0].scatter(reference_tau * 1e3, reference_fine, label="Fine DC L1", marker="x", s=24)
        axes[index - 1, 0].set(
            xlabel="Slant-range time hai chiều [ms]", ylabel="Doppler centroid [Hz]",
            title=f"Block {index}: fit RMS={record.rms_error_hz:.3f} Hz; nguồn chọn: {selected_source}",
        )
        axes[index - 1, 0].grid(alpha=0.3)
        axes[index - 1, 0].legend()
        results.append({
            "start_line": record.block.start_line, "stop_line_exclusive": record.block.stop_line,
            "acquisition_center_utc": Time(record.block.azimuth_time_s, format="gps").utc.isot,
            "l1_time_aligned_center_utc": Time(record.block.azimuth_time_s + time_offset, format="gps").utc.isot,
            "t0_s": record.t0_s, "range_times_s": tau.tolist(),
            "fine_baseband_hz": real_values(record.fine_baseband_hz),
            "fine_unwrapped_hz": real_values(record.fine_unwrapped_hz),
            "fine_absolute_hz": real_values(record.fine_absolute_hz),
            "l1_reference_range_times_s": reference_tau.tolist(),
            "l1_fine_interpolated_hz": fine.tolist(),
            "data_dc_polynomial": real_values(record.data_dc_polynomial),
            "geometry_dc_polynomial": real_values(record.geometry_dc_polynomial),
            "selected_source": selected_source,
            "selected_dc_hz": data_fit.tolist(),"selection_reason":record.selection_reason,
            "ambiguity_number": record.ambiguity_number,
            "rms_error_hz": float(record.rms_error_hz) if np.isfinite(record.rms_error_hz) else None, "rms_above_threshold": record.rms_above_threshold,
            "coherence": record.coherence.tolist(), "valid_mask": record.valid_mask.tolist(),
            "fine_vs_l1_rmse_hz": fine_rmse, "polynomial_vs_l1_rmse_hz": fit_rmse,
        })
        print(f"Block {index}: fit RMS={record.rms_error_hz:.3f} Hz, "
              f"Fine DC/L1 RMSE={fine_rmse} Hz, polynomial/L1 RMSE={fit_rmse:.3f} Hz")
    figure.tight_layout()
    figure.savefig(output / "doppler_centroid.png", dpi=160)
    plt.close(figure)
    (output / "doppler_centroid.json").write_text(
        json.dumps({**provenance, "estimates": results}, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def regression_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, default=Path.home() / "sar_data/focusing_data/product_1")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-lines", type=int, default=BATCH_LINES)
    parser.add_argument("--analysis-start-line", type=int, default=0)
    parser.add_argument("--analysis-step", type=int, default=1)
    parser.add_argument("--analysis-lines", type=int, default=None)
    parser.add_argument("--analysis-range-start", type=int, default=0)
    parser.add_argument("--analysis-range-stop", type=int, default=None)
    for name in ("bias", "gain", "quadrature"):
        parser.add_argument(f"--correct-iq-{name}", action=argparse.BooleanOptionalAction, default=None,
                            help="Ghi đè cờ AUX_PP1 cho thử nghiệm; mặc định giữ AUX.")
    for name in ("i-bias", "q-bias", "gain-imbalance", "quadrature-rad"):
        parser.add_argument(f"--predefined-{name}", type=float, default=None,
                            help="Statistic tường minh thay số đo; quadrature tính bằng radians.")
    parser.add_argument("--correct-rx-variation", action=argparse.BooleanOptionalAction, default=None,
                        help="Ghi đè receiver gain compensation; mặc định theo AUX_PP1.")
    parser.add_argument("--l0-pg-drift", action="store_true",
                        help="Dựng PG drift tương đối từ calibration L0/AUX; chế độ nghiên cứu, chưa phục hồi absolute scale.")
    parser.add_argument("--pg-oversampling-factor", type=int, default=16)
    parser.add_argument("--pg-denominator-relative-floor", type=float, default=0.)
    parser.add_argument("--internal-delay-source", choices=("l0", "l1", "none"), default="l0",
                        help="Mặc định đo delay từ PG L0/AUX; l1 chỉ cùng --l1-input-corrections; none để đối chứng.")
    parser.add_argument("--minimum-far-range-discard-samples", type=int, default=0,
                        help="Cắt tối thiểu mép xa để đối chứng cùng ROI; không giảm guard bắt buộc theo delay.")
    parser.add_argument("--l0-raw-analysis", action="store_true",
                        help="Thay bias L1 bằng raw analysis L0 khi dùng --l1-input-corrections.")
    parser.add_argument(
        "--l1-input-corrections", action="store_true",
        help="Dùng I/Q bias, replica, PG và EAP từ L1 để chuẩn bị input DCE chẩn đoán.",
    )
    args = parser.parse_args(argv)
    if args.internal_delay_source == "l1" and not args.l1_input_corrections:
        parser.error("--internal-delay-source l1 cần --l1-input-corrections.")
    product = args.product.expanduser().resolve()
    raw = one(product.glob("level-0/s1a-s4-raw-s-vv-*.dat"), "raw S4/VV")
    annotation_path = one(product.glob("level-1/s1a-s4-slc-vv-*.xml"), "annotation S4/VV")
    annotation = ET.parse(annotation_path).getroot()
    aux = read_auxiliary(product)
    reference_input = ReferenceInput(annotation, aux) if args.l1_input_corrections else None
    if args.output is None:
        suffix = "_l1_input" if args.l1_input_corrections else ""
        if args.l1_input_corrections and args.l0_raw_analysis:
            suffix += "_l0_raw_analysis"
        if args.l0_pg_drift:
            suffix += "_l0_pg_drift"
        suffix += f"_delay_{args.internal_delay_source}"
        args.output = Path(f"output/sentinel-1/product_1_dce{suffix}")
    level0 = sentinel1decoder.Level0File(str(raw))
    calibration_report = pg_report = pg_provider = None
    if args.l0_pg_drift or args.internal_delay_source == "l0":
        pg_times, pg_values, calibration_report = analyze_calibration(
            level0, product, oversampling_factor=args.pg_oversampling_factor,
            denominator_relative_floor=args.pg_denominator_relative_floor)
        args.output.mkdir(parents=True, exist_ok=True)
        calibration_path = args.output/"calibration.json"
        calibration_path.write_text(json.dumps(calibration_report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        if args.internal_delay_source == "l0" and not calibration_report["internal_delay"]["available"]:
            raise ValueError(f"Không có peak hợp lệ để đo delay L0; xem {calibration_path}.")
        if args.l0_pg_drift and not calibration_report["quality"]["interpolation_available"]:
            raise ValueError(f"Không đủ hai node PG để nội suy; xem {calibration_path}.")
    if args.l0_pg_drift:
        pg_report = calibration_report
        pg_provider = lambda query: interpolate_pg(pg_times, pg_values, query)
    internal_delay_s = (calibration_report["internal_delay"]["mean_s"] if args.internal_delay_source == "l0" else
                        reference_input.internal_delay_s if args.internal_delay_source == "l1" else 0.)
    raw_analysis = analyze_echoes(
        level0, batch_lines=args.batch_lines, azimuth_start_line=args.analysis_start_line,
        azimuth_step=args.analysis_step, max_azimuth_lines=args.analysis_lines,
        range_start_sample=args.analysis_range_start, range_stop_sample=args.analysis_range_stop,
    )
    correction_inputs = {key: getattr(args, key) for key in (
        "correct_iq_bias", "correct_iq_gain", "correct_iq_quadrature", "predefined_i_bias",
        "predefined_q_bias", "predefined_gain_imbalance", "predefined_quadrature_rad")}
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dce-", dir=args.output) as directory:
        segments, pri, sample_rate, diagnostics = compress_echoes(
            level0, aux, directory, reference_input,
            raw_analysis=raw_analysis if reference_input is None or args.l0_raw_analysis else None,
            batch_lines=args.batch_lines,
            **correction_inputs,
            correct_rx_variation=args.correct_rx_variation,
            pg_correction_provider=pg_provider,
            internal_delay_s=internal_delay_s,
            minimum_far_range_discard_samples=args.minimum_far_range_discard_samples,
        )
        corrected_analysis = analyze_echoes(
            level0, **raw_analysis["selection"], iq_correction=diagnostics[0]["raw_correction_parameters"],
        )
        far_throwaway = diagnostics[0]["hardware_delay_far_throwaway_samples"]
        config, starts, t0, offset = infer_configuration(
            annotation, segments, pri, sample_rate, aux["rms_threshold_hz"], far_throwaway
        )
        print(f"Cấu hình: {asdict(config)}; azimuth starts={starts}", flush=True)
        estimator = estimation.Estimator(1.0 / pri, config)
        geometry_estimator = geometry.Estimator.from_level0_product(
            level0, sentinel1decoder.constants.SPEED_OF_LIGHT_MPS / aux["radar_frequency_hz"]
        )
        estimates = estimator.estimate_segments(
            segments, custom_azimuth_starts=starts, dce_range_start_s=t0, t0_s=t0,
            geometry_dc_provider=lambda time, tau: geometry_estimator.estimate(
                time, tau * sentinel1decoder.constants.SPEED_OF_LIGHT_MPS / 2.0
            ),
        )
        provenance = {
            "raw_file": str(raw), "annotation_file": str(annotation_path), "auxiliary": aux,
            "auxiliary_generation_note": "AUX_INS khác generation manifest L1 là provenance; không suy ra lỗi validity hay nguyên nhân sai số DCE.",
            "prf_hz": 1.0 / pri, "range_sample_rate_hz": sample_rate,
            "config": asdict(config), "azimuth_start_lines": starts,
            "range_origin_s_from_l0": t0, "l1_zero_dop_minus_acq_time_s": offset,
            "configuration_source": "Layout suy từ L1 và kiểm trên lưới L0 theo yêu cầu người dùng; DC tính từ L0.",
            "raw_data_analysis": raw_analysis,
            "post_iq_correction_analysis": corrected_analysis,
            "iq_correction_overrides": correction_inputs,
            "receiver_compensation_override": args.correct_rx_variation,
            "l0_pg_calibration": pg_report,
            "internal_delay_compensation": {"source": args.internal_delay_source, "applied_s": internal_delay_s,
                "l0_estimate": None if calibration_report is None else calibration_report["internal_delay"],
                "l1_reference_s": None if reference_input is None else reference_input.internal_delay_s,
                "minimum_far_range_discard_samples": args.minimum_far_range_discard_samples},
            "input_corrections": {**(reference_input.summary() if reference_input else {"source": "L0 raw analysis + nominal chirp từ header + receiver compensation L0/AUX"}),
                                  "rx_variation_applied": diagnostics[0]["receiver_compensation"]["enabled"],
                                  "rx_variation_flag_source": diagnostics[0]["receiver_compensation"]["flag_source"],
                                  "pg_correction_source": f"L0/AUX PG (relative, {pg_report['quality']['source']})" if pg_report is not None else (
                                      "L1 inverse-PG annotation" if reference_input is not None else "none"),
                                  "internal_delay_compensation_s": internal_delay_s,
                                  "internal_delay_source": args.internal_delay_source,
                                  "internal_delay_reference_l1_s": None if reference_input is None else reference_input.internal_delay_s,
                                  "iq_bias_reference_l1": None if reference_input is None else [reference_input.bias.real, reference_input.bias.imag],
                                  "iq_bias": diagnostics[0]["iq_bias"],
                                  "iq_bias_actually_applied": diagnostics[0]["iq_bias"],
                                  "iq_bias_actual_source": diagnostics[0]["iq_bias_source"]},
            "polynomial_convention": "Hệ số tăng dần: f(tau)=c0+c1*(tau-t0)+c2*(tau-t0)^2; Hz, s.",
            "assumed_internal_parameters": {"unwrap_fft_length": 4096, "outlier_sigma": 3.5},
            "limitations": [
                *([
                    "PG chỉ khôi phục drift tương đối; không phục hồi absolute radiometric scale.",
                    "L1 input corrections không còn là kiểm chứng L0 độc lập với L1.",
                    "Bỏ mép xa thiếu support khi advance hardware delay; Fine DC L1 nội suy tới tâm block mới để so sánh.",
                    "RRF nominal theo polynomial/delay chưa chứng minh tái tạo chính xác spectrum Extended Tapered của IPF.",
                ] if reference_input else [f"Demo chưa áp {('PG drift, ' if pg_report is None else '')}EAP/RSL và RFI mitigation."]),
                "Geometry dùng ephemeris L0 và ellipsoid, không dùng AUX_RESORB/DEM hay orbit L1.",
                "L1 không xác định duy nhất range window/weighting, unwrap FFT hay ngưỡng outlier.",
            ],
            "segments": diagnostics,
        }
        save_results(args.output, estimates, annotation.findall("dopplerCentroid/dcEstimateList/dcEstimate"), provenance, offset)
    print(f"Đã ghi biểu đồ và JSON vào {args.output}")


def prepare_range_gain(job, aux, inventory, headers, geom, resources, *, ledger=None):
    """Resolve AUX flags and independent providers; no L1 input."""
    from sentinel1_processing.range_processing import dependent_gain as gain
    from sentinel1_processing.utils.eap_input import read_eap_lut, read_roll_law
    from sentinel1_processing.utils.terrain import RasterTerrain
    corrections=job["corrections"]
    if set(corrections)-{"apply_eap","apply_rsl","rrf_spectrum","reference_range_m","eap", "correct_iq_bias",
        "correct_iq_gain","correct_iq_quadrature","correct_rx_variation","perform_internal_calibration",
        "internal_delay_source","predefined_i_bias","predefined_q_bias","predefined_gain_imbalance","predefined_quadrature_rad"}:
        raise ValueError("Corrections có key không hỗ trợ.")
    flags={key:corrections.get(key,aux[key]) for key in ("apply_eap","apply_rsl")}
    if ledger is None:ledger=[]
    for key,value in flags.items():
        if not isinstance(value,bool):raise ValueError(f"{key} phải là boolean.")
        if value!=aux[key] and job["mode"]!="research":
            raise ValueError(f"Override {key} khác AUX cần research mode.")
        ledger.append({"name":key,"requested_by_aux":aux[key],"requested":value,
            "supported":True,"applied":False,"source":"job override" if key in corrections else "AUX_PP1"})
    reference=corrections.get("reference_range_m")
    if flags["apply_rsl"]:
        if isinstance(reference,bool) or not isinstance(reference,(int,float)):
            raise ValueError("RSL yêu cầu reference_range_m từ job/AUX, không lấy L1.")
        gain.rsl_correction([1.],reference_range_m=reference)
    if "eap" in corrections and not flags["apply_eap"]:
        raise ValueError("Job có EAP config nhưng apply_eap=false; không bỏ qua config ngầm.")
    bank_time=boresight_time=boresight_values=lut=grid=None
    eap=corrections.get("eap")
    if flags["apply_eap"]:
        keys={"update_interval_s","boresight_update_interval_s","antenna_axis_body","boresight_source",
              "minimum_magnitude","terrain","solver"}
        if not isinstance(eap,dict) or set(eap)!=keys:
            raise ValueError("EAP cần update intervals/antenna axis/boresight source/floor/terrain/solver tường minh.")
        paths={item["kind"]:item["path"] for item in inventory}
        grid,lut=read_eap_lut(paths["CAL"],swath="S4",polarisation="VV")
        terrain=eap["terrain"]
        if not isinstance(terrain,dict):raise ValueError("EAP terrain cần object.")
        if terrain.get("kind")=="constant":
            if job["mode"]!="research" or set(terrain)!={"kind","height_m"} or not np.isfinite(terrain["height_m"]):
                raise ValueError("Terrain hằng chỉ cho research, với height_m hữu hạn.")
            terrain_provider=lambda lat,lon:float(terrain["height_m"])
        elif terrain.get("kind")=="dem":
            if set(terrain)!={"kind","path","height_datum","height_unit"}:
                raise ValueError("DEM cần path/height_datum/height_unit.")
            terrain_provider=resources.enter_context(RasterTerrain(Path(terrain["path"]).expanduser(),
                height_datum=terrain["height_datum"],height_unit=terrain["height_unit"]))
        else:raise ValueError("EAP terrain.kind phải là dem hoặc research constant.")
        solver=eap["solver"]
        if not isinstance(solver,dict) or set(solver)!={"range_tolerance_m","height_tolerance_m","doppler_tolerance_mps","max_nfev","look_side"}:
            raise ValueError("EAP terrain solver thiếu/có tham số không hỗ trợ.")
        for key in ("range_tolerance_m","height_tolerance_m","doppler_tolerance_mps"):
            value=solver[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError(f"EAP solver {key} phải là số hữu hạn dương.")
        if isinstance(solver["max_nfev"],bool) or not isinstance(solver["max_nfev"],int) or solver["max_nfev"]<1:
            raise ValueError("EAP solver max_nfev phải là số nguyên dương.")
        if solver["look_side"] not in ("right","left"):
            raise ValueError("EAP solver look_side không hỗ trợ.")
        if (not isinstance(eap["minimum_magnitude"],(int,float)) or isinstance(eap["minimum_magnitude"],bool)
                or not np.isfinite(eap["minimum_magnitude"]) or eap["minimum_magnitude"]<0):
            raise ValueError("EAP minimum_magnitude không hợp lệ.")
        if eap["boresight_source"] not in ("attitude","attitude_or_nominal"):
            raise ValueError("EAP boresight source không hỗ trợ.")
        axis=np.asarray(eap["antenna_axis_body"],dtype=float)
        if axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1.,atol=1e-12,rtol=0):
            raise ValueError("EAP antenna_axis_body cần unit vector hữu hạn.")
        times=headers["line_times_s"]
        bank_time=gain.bank_times(times[0],times[-1],update_interval_s=eap["update_interval_s"])
        boresight_time=gain.bank_times(times[0],times[-1],update_interval_s=eap["boresight_update_interval_s"])
        geom.orbit.validate_time_coverage(np.unique(np.r_[bank_time,boresight_time]))
        roll=read_roll_law(paths["INS"])
        boresight_values=[];boresight_sources=[]
        for time in boresight_time:
            if geom.attitude_times_s[0]<=time<=geom.attitude_times_s[-1]:
                boresight_values.append(geom.boresight_angle_deg(time,antenna_axis_body=eap["antenna_axis_body"]))
                boresight_sources.append("L0 attitude")
            elif eap["boresight_source"]=="attitude_or_nominal":
                from sentinel1_processing.doppler_centroid.geometry import geodetic_position, nominal_boresight_angle_deg
                altitude=geodetic_position(geom.orbit.position(time),ellipsoid_a_m=geom.orbit.a_m,
                    ellipsoid_b_m=geom.orbit.b_m)[2]
                boresight_values.append(nominal_boresight_angle_deg(altitude,**roll))
                boresight_sources.append("AUX_INS nominal roll; missing attitude coverage")
            else:raise ValueError("EAP boresight query ngoài attitude coverage.")
        ledger[0].update({"bank_times_gps_s":bank_time.tolist(),"boresight_times_gps_s":boresight_time.tolist(),
            "boresight_angles_deg":boresight_values,"boresight_sources":boresight_sources,
            "terrain":terrain,"lut_source":paths["CAL"]})

    def factory(tau):
        ranges=np.asarray(tau)*sentinel1decoder.constants.SPEED_OF_LIGHT_MPS/2
        rsl=gain.rsl_correction(ranges,reference_range_m=reference) if flags["apply_rsl"] else np.ones(len(ranges))
        if not flags["apply_eap"]:
            return lambda times:rsl
        bank=gain.build_eap_bank(bank_time,ranges,
            look_angle_provider=lambda time,r:geom.terrain_look_angles_deg(time,r,terrain_provider=terrain_provider,**solver),
            boresight_angle_provider=lambda time:float(np.interp(time,boresight_time,boresight_values)),
            angular_grid_deg=grid,complex_lut=lut,minimum_magnitude=eap["minimum_magnitude"])
        bank*=rsl[None,:]
        if not np.isfinite(bank).all():raise ValueError("EAP/RSL bank không hữu hạn.")
        return lambda times:gain.interpolate_bank(times,bank_time,bank)
    return (factory if any(flags.values()) else None),ledger


def run_job(args):
    from contextlib import ExitStack
    with ExitStack() as resources:
        return _run_job(args, resources)


def _run_job(args, resources):
    """Luồng L0/AUX độc lập; preflight luôn được ghi trước decode I/Q."""
    from sentinel1_processing.utils import dce_job
    from dataclasses import replace

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    report = {"ready": False, "errors": [], "processing_profile": "l0_aux"}
    try:
        job = dce_job.read_job(args.config)
        report["job"] = job
        product = args.product.expanduser().resolve()
        raw = one(product.glob("level-0/s1a-s4-raw-s-vv-*.dat"), "raw S4/VV")
        level0 = sentinel1decoder.Level0File(str(raw))
        headers = dce_job.header_inventory(level0,
            continuity_tolerance_s=job["layout"]["continuity_tolerance_s"])
        inventory = dce_job.auxiliary_inventory(product, selection_time_gps=headers["line_times_s"][0])
        aux = read_auxiliary(product, {entry["kind"]: entry["path"] for entry in inventory})
        plans = dce_job.resolve_processing_scopes(level0,headers,job,aux)
        windows = [w for plan in plans for w in plan["windows"]]
        layout_report = plans[0]["layout"] if len(plans)==1 else {"policy":job["layout"]["policy"],"scopes":len(plans)}
        report["processing_sources"]=[dce_job.fingerprint(Path(__file__)),
            *[dce_job.fingerprint(p) for p in sorted(Path(estimation.__file__).parents[1].rglob("*.py"))]]
        report.update({"headers": {k:v for k,v in headers.items() if k not in ("line_times_s","_scope_views")},
            "auxiliary_inventory":inventory,"resolved_auxiliary":aux,
            "layout":layout_report,"windows":windows,
            "inputs":[dce_job.fingerprint(raw), dce_job.fingerprint(args.config)],
            "parameter_sources":{"rms_threshold_hz":"AUX_PP1 SM_SL1__1/dcProcParams/dcRmsErrorThreshold",
                "layout":"job layout/context; inferred spacing profile", "estimator":"job estimator + recorded Config defaults"}})
        terrain=job["corrections"].get("eap",{}).get("terrain",{})
        if terrain.get("kind")=="dem":
            report["inputs"].append(dce_job.fingerprint(Path(terrain["path"]).expanduser()))
        if job["layout"].get("missing_support_policy","error")=="error" and any(not w["available"] for w in windows):
            report["errors"].append("DCE windows thiếu support hoặc cắt qua scope boundary; xem windows.")
        spectrum=job["corrections"].get("rrf_spectrum",aux["rrf_spectrum"])
        if spectrum not in ("header_chirp","aux_nominal_fft") or job["mode"]!="research":
            report["errors"].append("RRF chưa xác nhận cho strict; header_chirp/aux_nominal_fft chỉ dùng research.")
        geometry_config=job.get("geometry",{})
        if set(geometry_config)!={"n_control_points","quaternion_norm_tolerance"}:
            raise ValueError("Geometry cần n_control_points và quaternion_norm_tolerance tường minh.")
        orbit=None
        orbit_path=getattr(args,"orbit_eof",None)
        if orbit_path:
            from sentinel1_processing.utils.orbit_input import read_orbit
            orbit=read_orbit(orbit_path,mission="Sentinel-1A",
                wavelength_m=sentinel1decoder.constants.SPEED_OF_LIGHT_MPS/aux["radar_frequency_hz"],
                ellipsoid_a_m=aux["ellipsoid_a_m"],ellipsoid_b_m=aux["ellipsoid_b_m"])
            report["inputs"].append(dce_job.fingerprint(orbit_path))
        needs_geometry=(aux["dc_method"]!="Predefined" or job["corrections"].get("apply_eap",aux["apply_eap"]))
        geom=None
        centers=np.array(sorted({w["center_time_s"] for w in windows if w["available"]}))
        if needs_geometry:
            geom=geometry.Estimator.from_level0_product(level0,
                sentinel1decoder.constants.SPEED_OF_LIGHT_MPS/aux["radar_frequency_hz"],
                orbit_provider=orbit,ellipsoid_a_m=aux["ellipsoid_a_m"],ellipsoid_b_m=aux["ellipsoid_b_m"],
                quaternion_norm_tolerance=geometry_config["quaternion_norm_tolerance"])
            geom.orbit.validate_time_coverage(centers)
            if aux["dc_method"]!="Predefined" and (centers[0]<geom.attitude_times_s[0] or centers[-1]>geom.attitude_times_s[-1]):
                report["errors"].append("Attitude không phủ mọi DCE center.")
            report["geometry"]={"source":"EOF orbit + L0 attitude" if orbit_path else "L0 PVT/attitude", "query_times_gps_s":centers.tolist(),
                "orbit_interval_gps_s":[float(geom.orbit.orbit_times_s[0]),float(geom.orbit.orbit_times_s[-1])],
                "attitude_interval_gps_s":[float(geom.attitude_times_s[0]),float(geom.attitude_times_s[-1])],
                "attitude_frame_validation":"EOCFI-NOTE-052 2.1 §3.5 numerical example; GM2000 via mean J2000 frame"}
        else:
            if orbit is not None:orbit.validate_time_coverage(centers)
            report["geometry"]={"source":"not required for predefined DC without EAP"}
        ledger=[]
        report["correction_ledger"]=ledger
        ledger.append({"name":"rrf_spectrum","requested_by_aux":aux["rrf_spectrum"],"requested":spectrum,
            "supported":spectrum in ("header_chirp","aux_nominal_fft") and job["mode"]=="research","applied":False})
        report["processing_scopes"]=[]
        for plan in plans:
            range_ledger=[]
            scope_report={"scope_id":plan["scope_id"],"global_start_line":plan["global_start_line"],
                "line_count":len(plan["headers"]["line_times_s"]),"pri_s":plan["headers"]["pri_s"],
                "split_reasons":plan["headers"].get("split_reasons",[]),
                "layout":plan["layout"],"windows":plan["windows"],"correction_ledger":range_ledger,
                "parameters":dce_job.parameter_inventory(job,plan["config"],plan["headers"])}
            plan["report"]=scope_report
            report["processing_scopes"].append(scope_report)
            try:
                factory=None
                if any(w["available"] for w in plan["windows"]):
                    factory,_=prepare_range_gain(job,aux,inventory,plan["headers"],geom,resources,ledger=range_ledger)
                plan["range_gain_factory"]=factory
            finally:
                if len(report["processing_scopes"])==1:ledger.extend(range_ledger)
        correction_overrides={key:job["corrections"].get(key) for key in (
            "correct_iq_bias","correct_iq_gain","correct_iq_quadrature","correct_rx_variation",
            "predefined_i_bias","predefined_q_bias","predefined_gain_imbalance","predefined_quadrature_rad")}
        for key in ("correct_iq_bias","correct_iq_gain","correct_iq_quadrature","correct_rx_variation","perform_internal_calibration"):
            requested=job["corrections"].get(key,aux[key])
            if not isinstance(requested,bool):raise ValueError(f"{key} cần boolean.")
            if requested!=aux[key] and job["mode"]!="research":
                raise ValueError(f"Override {key} khác AUX chỉ dùng research.")
            ledger.append({"name":key,"requested_by_aux":aux[key],"requested":requested,"supported":True,"applied":False,
                "source":"job override" if key in job["corrections"] else "AUX_PP1"})
        for key,value in correction_overrides.items():
            if key.startswith("predefined_") and value is not None and job["mode"]!="research":
                raise ValueError("Predefined raw statistic override chỉ dùng research.")
        delay_source=job["corrections"].get("internal_delay_source","l0")
        if delay_source not in ("l0","none") or delay_source=="none" and job["mode"]!="research":
            raise ValueError("Independent delay source cần l0; none chỉ dùng research, không nhận L1.")
        ledger.append({"name":"internal_delay","requested":delay_source=="l0","source":delay_source,
            "supported":True,"applied":False})
        report["correction_ledger"]=ledger
        report["ready"]=not report["errors"]
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ET.ParseError) as error:
        report["errors"].append(str(error))
    (output/"preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    if not report["ready"]:
        raise ValueError(f"Preflight chưa đạt: {report['errors']}; xem {output/'preflight.json'}")
    if args.preflight_only:
        print(f"Preflight đạt: {output/'preflight.json'}")
        return
    calibration_report=None;pg_provider=None
    report["range_support_validated"]=False
    try:
        needs_pg=job["corrections"].get("perform_internal_calibration",aux["perform_internal_calibration"])
        if delay_source=="l0" or needs_pg:
            pg_times,pg_values,calibration_report=analyze_calibration(level0,product,
                oversampling_factor=job["calibration"]["oversampling_factor"],
                denominator_relative_floor=job["calibration"]["denominator_relative_floor"],
                selected_auxiliary_paths={entry["kind"]:entry["path"] for entry in inventory},
                output_normalization=job["calibration"]["output_normalization"])
            (output/"calibration.json").write_text(json.dumps(calibration_report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        if delay_source=="l0" and not calibration_report["internal_delay"]["available"]:
            raise ValueError("Không có internal delay L0 hợp lệ; xem calibration.json.")
        if needs_pg:
            if not calibration_report["quality"]["interpolation_available"]:
                raise ValueError("Không đủ PG nodes theo AUX; xem calibration.json.")
            pg_provider=lambda times:interpolate_pg(pg_times,pg_values,times)
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as error:
        report["ready"]=False
        report["errors"].append(str(error))
        (output/"preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        raise
    delay=calibration_report["internal_delay"]["mean_s"] if delay_source=="l0" else 0.
    reference_factory=None
    if spectrum=="aux_nominal_fft":
        from sentinel1_processing.range_processing.range_reference_function import from_coefficients
        reference_factory=lambda **parameters:from_coefficients(**parameters,
            amplitude_coefficients=aux["nominal_amplitude_coefficients"],
            phase_coefficients_cycles=aux["nominal_phase_coefficients_cycles"])
    # Resolve the measured-delay range support and all gain banks before echo statistics/decode.
    report["range_support_validated"]=False
    try:
        for plan in plans:
            if not any(w["available"] for w in plan["windows"]):
                plan["report"]["skipped_reason"]="no_supported_dce_window"
                continue
            config,t0,support=dce_job.range_support(plan["level0"],aux,plan["config"],internal_delay_s=delay,
                minimum_far_range_discard_samples=job.get("minimum_far_range_discard_samples",0))
            plan["config"]=config;plan["t0"]=t0
            plan["report"]["range_support"]={k:v for k,v in support.items() if not k.startswith("_")}
            plan["report"]["resolved_config"]=asdict(config)
            plan["report"]["parameters"]=dce_job.parameter_inventory(job,config,plan["headers"])
            original=plan["range_gain_factory"]
            if original is not None:
                gain_providers={(len(tau),float(tau[0]),float(tau[-1])):original(tau) for tau in support["_grids"]}
                plan["range_gain_factory"]=lambda tau,providers=gain_providers:providers[(len(tau),float(tau[0]),float(tau[-1]))]
        report["range_support_validated"]=True
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as error:
        report["ready"]=False
        report["errors"].append(str(error))
        raise
    finally:
        (output/"preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    estimates=[];diagnostics=[]
    for plan in plans:
        active=[w for w in plan["windows"] if w["available"]]
        if not active:
            continue
        view=plan["level0"];config=plan["config"];t0=plan["t0"]
        analysis=analyze_echoes(view,batch_lines=job.get("batch_lines",BATCH_LINES),**RAW_ANALYSIS_SELECTION)
        analysis["processing_scope_id"]=plan["scope_id"]
        with TemporaryDirectory(prefix="dce-",dir=output) as directory:
            segments,pri,rate,details=compress_echoes(view,aux,directory,raw_analysis=analysis,
                batch_lines=job.get("batch_lines",BATCH_LINES),pg_correction_provider=pg_provider,
                **correction_overrides,range_gain_factory=plan["range_gain_factory"],range_reference_factory=reference_factory,
                internal_delay_s=delay,minimum_far_range_discard_samples=job.get("minimum_far_range_discard_samples",0),
                continuity_tolerance_s=job["layout"]["continuity_tolerance_s"])
            actual_start=max(float(segment.slant_range_times_s[0]) for segment in segments)
            actual_end=min(float(segment.slant_range_times_s[-1]) for segment in segments)+1/rate
            if (abs(actual_start-t0)*rate>1e-5
                    or int(round((actual_end-actual_start)*rate))!=plan["report"]["range_support"]["available_range_samples"]):
                raise RuntimeError("Compressed range support khác preflight; không tiếp tục với ROI sai.")
            records=estimation.Estimator(1/pri,config).estimate_segments(segments,
                custom_azimuth_starts=[w["start_line"] for w in active],dce_range_start_s=t0,t0_s=t0,
                azimuth_gap_tolerance_s=job["layout"]["continuity_tolerance_s"],
                geometry_dc_provider=(None if geom is None else lambda time,tau:geom.estimate(
                    time,tau*sentinel1decoder.constants.SPEED_OF_LIGHT_MPS/2,
                    n_control_points=geometry_config["n_control_points"])))
            for record,window in zip(records,active):
                record.processing_scope_id=plan["scope_id"]
                record.acquisition_window_id=window["window_id"]
                offset=plan["global_start_line"]
                record.block=replace(record.block,start_line=record.block.start_line+offset,
                    stop_line=record.block.stop_line+offset,center_line=record.block.center_line+offset)
            estimates.extend(records)
            diagnostics.extend({**item,"processing_scope_id":plan["scope_id"]} for item in details)
    all_ledgers=[report["correction_ledger"],*[p["report"]["correction_ledger"] for p in plans]]
    for correction in (item for ledger in all_ledgers for item in ledger):
        correction["applied"]=(correction["requested"] if isinstance(correction["requested"],bool) else correction["supported"])
    provenance={"job":job,"resolved_config":asdict(plans[0]["config"]) if len(plans)==1 else None,
        "resolved_configs_by_scope":{p["scope_id"]:asdict(p["config"]) for p in plans},"preflight":report,
        "calibration":calibration_report,"segments":diagnostics,"processing_sources":report["processing_sources"],
        "timeline":"acquisition GPS seconds", "l1_used_for_processing":False}
    dce_job.write_estimates(output/"doppler_centroid.json",estimates,provenance)
    if args.compare_annotation:
        if args.comparison_max_time_delta_s is None:
            raise ValueError("Comparison cần --comparison-max-time-delta-s tường minh; processing result đã lưu.")
        comparison=dce_job.compare_annotations(estimates,args.compare_annotation,
            max_time_delta_s=args.comparison_max_time_delta_s)
        (output/"comparison.json").write_text(json.dumps(comparison,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(f"Đã lưu DCE độc lập L1: {output/'doppler_centroid.json'}")


def main():
    parser=argparse.ArgumentParser(description="Ước lượng DCE S4 từ L0/AUX/job; L1 chỉ để comparison.")
    parser.add_argument("--profile",choices=("l0-aux","l1-regression"),default="l0-aux")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--product",type=Path,default=Path.home()/"sar_data/focusing_data/product_1")
    parser.add_argument("--output",type=Path,default=Path("output/sentinel-1/s4_dce_job"))
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--orbit-eof",type=Path,help="Orbit EOF thay PVT L0, không fallback nếu file chỉ định lỗi.")
    parser.add_argument("--compare-annotation",type=Path)
    parser.add_argument("--comparison-max-time-delta-s",type=float)
    args,extra=parser.parse_known_args()
    if args.profile=="l1-regression":
        if args.config or args.preflight_only or args.compare_annotation:
            parser.error("Regression dùng CLI cũ; không nhận config/preflight/comparison mới.")
        import sys
        argv=list(sys.argv[1:])
        if "--profile" in argv:
            index=argv.index("--profile");del argv[index:index+2]
        else:
            argv.remove("--profile=l1-regression")
        return regression_main(argv)
    if extra:parser.error(f"Tham số không hỗ trợ cho l0-aux: {extra}")
    if not args.config:parser.error("Luồng L0/AUX cần --config JSON; dùng --profile l1-regression để chạy nhánh cũ.")
    run_job(args)


if __name__ == "__main__":
    main()
