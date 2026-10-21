"""Demo DCE S4/VV từ L0; suy layout từ annotation L1 theo yêu cầu người dùng."""

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
from sentinel1_processing.utils.dce_reference_input import ReferenceInput


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


def read_auxiliary(product):
    ins_path = one(product.glob("auxiliary/*/data/s1a-aux-ins.xml"), "AUX_INS")
    pp1_path = one(product.glob("auxiliary/*/data/s1a-aux-pp1.xml"), "AUX_PP1")
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
    if (pp1.findtext("dcProcParams/dcMethod") != "Data Analysis"
            or pp1.findtext("dcProcParams/dcInputData") != "Range Compressed"):
        raise ValueError("Demo yêu cầu DCE Data Analysis trên Range Compressed.")
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
    return {
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
):
    """Decode/compress theo batch vào memmap; không giữ toàn raw scene trong RAM."""
    if not isinstance(batch_lines, (int, np.integer)) or isinstance(batch_lines, bool) or batch_lines <= 0:
        raise ValueError("batch_lines phải là số nguyên dương.")
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
        n_valid = n_range - pulse_samples
        # Advance do hardware delay cần mẫu tương lai; bỏ mép xa thiếu full chirp support.
        far_throwaway = 0 if reference_input is None else int(np.ceil(reference_input.internal_delay_s * sample_rate))
        stop_sample = n_valid - far_throwaway
        if stop_sample < 2:
            raise ValueError(f"Chunk {chunk} không đủ range support sau compression.")
        tau = correct(origin + np.arange(n_range) / sample_rate, aux["swst_bias_s"])
        offset = (origin - min(origins)) * sample_rate
        shift = (round(offset) - offset) / sample_rate
        eta = metadata["Coarse Time"].to_numpy(float) + metadata["Fine Time"].to_numpy(float)
        if not np.allclose(np.diff(eta), pri, rtol=0, atol=0.05 * pri):
            raise ValueError(f"Chunk {chunk}: timeline có gap hoặc PRI không đều.")
        data = np.lib.format.open_memmap(
            Path(directory) / f"range-{chunk}.npy", mode="w+", dtype=np.complex64,
            shape=(len(metadata), n_valid),
        )
        raw = decoder.decode_packets(metadata.iloc[:batch_lines])
        rx_gain = calculate_gain(
            n_range, sample_rate_hz=sample_rate, swst_s=float(metadata["SWST"].iloc[0]),
            pulse_length_s=float(first["Tx Pulse Length"]), suppressed_s=suppressed_time,
            gain_trend_coefficients=aux.get("rx_gain_trend"),
            gain_overshoot_coefficients=aux.get("rx_gain_overshoot"), enabled=rx_enabled,
        )
        reference = bank = None
        if reference_input is not None:
            fft_length = 1 << (n_range + pulse_samples - 2).bit_length()
            reference = reference_input.range_reference(sample_rate, float(first["Tx Pulse Length"]), fft_length)
            bank = reference_input.range_gain_bank(tau[:stop_sample] + shift)
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
                if reference_input is not None:
                    raw *= reference_input.pg_gain(eta[start:stop])[:, None]
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
            if reference_input is not None:
                reference_input.apply_range_gain(data[start:stop, :stop_sample], bank, eta[start:stop])
                if not np.isfinite(data[start:stop, :stop_sample]).all():
                    raise ValueError(f"Chunk {chunk}: input DCE sau hiệu chỉnh không hữu hạn.")
        data.flush()
        segments.append(estimation.Segment(data[:, :stop_sample], range_times[:stop_sample], eta, name=f"chunk-{chunk}"))
        diagnostics.append({
            "chunk": int(chunk), "shape": list(segments[-1].range_compressed.shape),
            "hardware_delay_far_throwaway_samples": far_throwaway,
            "iq_bias": [bias.real, bias.imag],
            "iq_bias_source": bias_source if flags["correct_iq_bias"] else "disabled",
            "iq_analysis_lines": raw_analysis["line_count"] if raw_analysis is not None else 0,
            "raw_data_analysis": raw_analysis if chunk == chunks[0][0] else None,
            "raw_correction_parameters": parameters,
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
        geometry_fit = np.polynomial.polynomial.polyval(tau - record.t0_s, record.geometry_dc_polynomial)
        selected_source = "geometry" if record.rms_above_threshold else "data"
        fine_rmse = float(np.sqrt(np.mean((record.fine_baseband_hz - fine) ** 2)))
        fit_rmse = float(np.sqrt(np.mean((data_fit - l1_fit) ** 2)))
        axes[index - 1, 0].plot(tau * 1e3, data_fit, label="Data DC từ L0")
        axes[index - 1, 0].plot(tau * 1e3, geometry_fit, label="Geometry DC từ L0", linestyle=":")
        axes[index - 1, 0].plot(tau * 1e3, l1_fit, label="Polynomial L1 tham chiếu", linestyle="--")
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
            "fine_baseband_hz": record.fine_baseband_hz.tolist(),
            "fine_unwrapped_hz": record.fine_unwrapped_hz.tolist(),
            "fine_absolute_hz": record.fine_absolute_hz.tolist(),
            "l1_reference_range_times_s": reference_tau.tolist(),
            "l1_fine_interpolated_hz": fine.tolist(),
            "data_dc_polynomial": record.data_dc_polynomial.tolist(),
            "geometry_dc_polynomial": record.geometry_dc_polynomial.tolist(),
            "selected_source": selected_source,
            "selected_dc_hz": (geometry_fit if record.rms_above_threshold else data_fit).tolist(),
            "ambiguity_number": record.ambiguity_number,
            "rms_error_hz": record.rms_error_hz, "rms_above_threshold": record.rms_above_threshold,
            "coherence": record.coherence.tolist(), "valid_mask": record.valid_mask.tolist(),
            "fine_vs_l1_rmse_hz": fine_rmse, "polynomial_vs_l1_rmse_hz": fit_rmse,
        })
        print(f"Block {index}: fit RMS={record.rms_error_hz:.3f} Hz, "
              f"Fine DC/L1 RMSE={fine_rmse:.3f} Hz, polynomial/L1 RMSE={fit_rmse:.3f} Hz")
    figure.tight_layout()
    figure.savefig(output / "doppler_centroid.png", dpi=160)
    plt.close(figure)
    (output / "doppler_centroid.json").write_text(
        json.dumps({**provenance, "estimates": results}, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main():
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
    parser.add_argument("--l0-raw-analysis", action="store_true",
                        help="Thay bias L1 bằng raw analysis L0 khi dùng --l1-input-corrections.")
    parser.add_argument(
        "--l1-input-corrections", action="store_true",
        help="Dùng I/Q bias, replica, PG và EAP từ L1 để chuẩn bị input DCE chẩn đoán.",
    )
    args = parser.parse_args()
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
        args.output = Path(f"output/sentinel-1/product_1_dce{suffix}")
    level0 = sentinel1decoder.Level0File(str(raw))
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
            "input_corrections": {**(reference_input.summary() if reference_input else {"source": "L0 raw analysis + nominal chirp từ header + receiver compensation L0/AUX"}),
                                  "rx_variation_applied": diagnostics[0]["receiver_compensation"]["enabled"],
                                  "rx_variation_flag_source": diagnostics[0]["receiver_compensation"]["flag_source"],
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
                ] if reference_input else ["Demo chưa áp PG drift, EAP/RSL và RFI mitigation."]),
                "Geometry dùng ephemeris L0 và ellipsoid, không dùng AUX_RESORB/DEM hay orbit L1.",
                "L1 không xác định duy nhất range window/weighting, unwrap FFT hay ngưỡng outlier.",
            ],
            "segments": diagnostics,
        }
        save_results(args.output, estimates, annotation.findall("dopplerCentroid/dcEstimateList/dcEstimate"), provenance, offset)
    print(f"Đã ghi biểu đồ và JSON vào {args.output}")


if __name__ == "__main__":
    main()
