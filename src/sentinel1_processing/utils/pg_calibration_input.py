"""Inventory and research PG drift for the local S1A/S4/VV acquisition."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import sentinel1decoder

from ..pre_processing.internal_calibration import (
    extract_pulse, reconstruct_replica, nominal_pg_replica, pg_products, validate_pg, mean_internal_delay, correlation_peak,
)


SIGNALS = {8: "TxCal", 9: "RxCal", 10: "EpdnCal", 11: "TaCal", 12: "ApdnCal", 15: "TxHCalIso"}


def read_configuration(product):
    roots, paths = [], []
    for kind in ("ins", "pp1"):
        matches = list(Path(product).glob(f"auxiliary/*/data/s1a-aux-{kind}.xml"))
        if len(matches) != 1:
            raise ValueError(f"Cần đúng một AUX_{kind.upper()}.")
        paths.append(str(matches[0]))
        roots.append(ET.parse(matches[0]).getroot())
    ins, pp1 = roots
    # Bundled 3.14 XSD has stale dB/degrees; this reader supports the 3.15/3.16 contract.
    if pp1.get("schemaVersion") not in ("3.15", "3.16"):
        raise ValueError("Cần AUX_PP1 3.15/3.16 để xác định PG amplitude tuyến tính và phase radian.")
    records = [r for r in ins.findall("internalCalibrationParamsList/internalCalibrationParams")
               if r.findtext("swath") == "S4" and r.findtext("polarisation") == "VV"]
    products = [p for p in pp1.findall("productList/product") if p.findtext("productId") == "SM_SL1__1"]
    if len(records) != 1 or len(products) != 1:
        raise ValueError("Không tìm thấy đúng calibration S4/VV và processor SM_SL1__1.")
    calibration, processor = records[0], products[0].find("preProcParams")
    flag = processor.findtext("performInternalCalibrationFlag")
    source = processor.findtext("pgSource")
    if flag not in ("true", "false") or source not in ("Extracted", "Model"):
        raise ValueError("Cấu hình performInternalCalibrationFlag/pgSource không hợp lệ.")
    orders = {}
    for kind in ("replica", "pg"):
        orders[kind] = {p.findtext("signal"): {"order": [int(x) for x in p.findtext("order").split()],
                                              "method": p.findtext("method")}
                        for p in calibration.findall(f"{kind}PccParamsList/pccParams")}
    model = np.fromstring(calibration.findtext("pgProductModel/values"), sep=" ").reshape(-1, 2)
    model = model[:, 0]+1j*model[:, 1]
    # Nonconstant LUTs need an ascending-node epoch; never invent that epoch from L1.
    if len(model) < 2 or not np.isfinite(model).all() or not np.all(model == model[0]):
        raise ValueError("Adapter S4 hiện yêu cầu AUX PG model hằng; LUT khác cần mốc ascending node tường minh.")
    reference = complex(float(calibration.findtext("pgReference/re")), float(calibration.findtext("pgReference/im")))
    names = {"pg_amp_std_fraction": "maxPgAmpStdFraction", "pg_phase_std_fraction": "maxPgPhaseStdFraction",
             "max_pg_amp_error": "maxPgAmpError", "max_pg_phase_error": "maxPgPhaseError",
             "invalid_pg_max_fraction": "maxNumInvalidPgValFraction"}
    thresholds = {key: float(processor.findtext(f"replicaThresholds/{field}")) for key, field in names.items()}
    return {"files": paths, "orders": orders, "perform_internal_calibration": flag == "true",
            "pg_source": source, "pg_reference": [reference.real, reference.imag],
            "constant_pg_model": [model[0].real, model[0].imag], "thresholds": thresholds,
            "pg_threshold_units": {"amplitude": "linear", "phase": "radian", "pp1_schema_version": pp1.get("schemaVersion")},
            "pg_model_interval_s": float(calibration.findtext("pgProductModel/pgModelInterval")),
            "delta_t_guard1_s": float(ins.findtext("deltaTGuard1")),
            "delta_t_suppr_s": float(ins.findtext("deltaTSuppr"))}


def analyze_calibration(level0, product, *, oversampling_factor, denominator_relative_floor):
    """No L1 input; return selected acquisition times, relative PG correction, report.

    PGref/PGcal uses the nominal 1/N reference (Calibration Plan 8.1).
    Reports delay and PG availability independently; consumers enforce their needs.
    Returned PG nodes can contain fewer than two values and cannot then be interpolated.
    """
    if not Path(level0.filename).name.startswith("s1a-s4-raw-s-vv-"):
        raise ValueError("Adapter calibration này chỉ hỗ trợ sản phẩm S1A/S4/VV hiện tại.")
    configuration = read_configuration(product)
    if (not isinstance(oversampling_factor, (int, np.integer)) or isinstance(oversampling_factor, bool)
            or oversampling_factor < 2):
        raise ValueError("oversampling_factor phải là số nguyên >=2.")
    floor = np.asarray(denominator_relative_floor)
    if floor.ndim != 0 or floor.dtype.kind not in "fiu" or not np.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("denominator_relative_floor phải trong [0,1).")
    pg_reference = complex(*configuration["pg_reference"])
    if not np.isfinite(pg_reference) or abs(pg_reference) == 0:
        raise ValueError("PG reference phải hữu hạn khác zero.")
    inventory, groups, current = [], [], []
    decoder = sentinel1decoder.Level0Decoder(level0.filename)
    image_rx_powers = []

    def flush():
        if current:
            groups.append(current.copy())
            current.clear()

    for chunk in level0.acquisition_chunks:
        metadata = level0.get_acquisition_chunk_metadata(chunk)
        if metadata["Signal Type"].nunique() != 1:
            raise ValueError("Một acquisition chunk có nhiều Signal Type.")
        signal = int(metadata["Signal Type"].iloc[0].value)
        times = metadata["Coarse Time"].to_numpy(float)+metadata["Fine Time"].to_numpy(float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Timeline calibration/header không hữu hạn hoặc không tăng.")
        record = {"chunk": int(chunk), "packet_count": len(metadata), "signal_code": signal,
                  "signal": SIGNALS.get(signal, str(metadata["Signal Type"].iloc[0])),
                  "first_packet": int(metadata.index[0]), "last_packet": int(metadata.index[-1]),
                  "start_gps_s": float(times[0]), "last_gps_s": float(times[-1]),
                  "error_flag_count": int(metadata["Error Flag"].sum())}
        if signal not in SIGNALS:
            flush()
            inventory.append(record)
            continue
        fields = ("BAQ Mode", "Range Decimation", "Swath Number", "Cal Type", "Calibration Beam Address",
                  "Rx Gain", "Rx Channel ID", "Tx Pulse Number", "Tx Pulse Length", "Tx Pulse Start Frequency", "Tx Ramp Rate")
        record["headers"] = {f: [str(v) for v in metadata[f].unique()] for f in fields}
        rate = sentinel1decoder.utilities.range_dec_to_sample_rate(metadata["Range Decimation"].iloc[0])
        is_pg = metadata["Range Decimation"].iloc[0].value == 0
        record.update(kind="pg" if is_pg else "image", sample_rate_hz=float(rate),
                      range_samples=2*int(metadata["Number of Quads"].iloc[0]))
        if (metadata["Error Flag"].any() or metadata["BAQ Mode"].iloc[0].value != 0
                or metadata["Rx Channel ID"].iloc[0].value != 0
                or np.any(np.diff(metadata["Space Packet Count"].to_numpy()) != 1)):
            raise ValueError(f"Calibration chunk {chunk}: lỗi header, thiếu packet hoặc ngoài phạm vi VV/BYPASS.")
        if signal in (8, 9, 10, 11, 12):
            parameters = configuration["orders"]["pg" if is_pg else "replica"][SIGNALS[signal]]
            raw = decoder.decode_packets(metadata)
            if raw.shape != (len(metadata), record["range_samples"]):
                raise ValueError("Calibration decoded shape không khớp header.")
            pulse = extract_pulse(raw, **parameters)
            selected = metadata.iloc[np.array(parameters["order"])-1]
            record["selection"] = {**parameters, "packets": [int(x) for x in selected.index],
                                    "beam_addresses": selected["Calibration Beam Address"].astype(int).tolist()}
            record["extracted_mean_power"] = float(np.mean(abs(pulse)**2))
            if not is_pg and signal == 9:
                image_rx_powers.append({"chunk": int(chunk), "power": record["extracted_mean_power"]})
        else:
            pulse = None
        inventory.append(record)
        if is_pg:
            if signal == 8:
                flush()
            current.append((record, metadata, pulse))
        else:
            flush()
    flush()
    if not configuration["perform_internal_calibration"]:
        raise ValueError("AUX tắt internal calibration; không tự dựng PG extracted.")
    if not groups:
        raise ValueError("Không có PG calibration sequence.")
    replicas, node_times, nodes, reference, chirp_parameters = [], [], [], None, None
    for group in groups:
        signals = {r["signal_code"]: p for r, _, p in group if p is not None}
        if set(signals) != {8, 9, 10, 11, 12} or len(signals) != sum(p is not None for _, _, p in group):
            raise ValueError("PG sequence thiếu hoặc không đúng bộ Tx/Rx/EPDN/TA/APDN.")
        records = [r for r, _, _ in group]
        first = group[0][1].iloc[0]
        for _, metadata, _ in group:
            for field in ("Range Decimation", "Number of Quads", "Tx Pulse Length", "Tx Pulse Start Frequency", "Tx Ramp Rate"):
                if not (metadata[field] == first[field]).all():
                    raise ValueError(f"PG sequence thay đổi {field}.")
        rate = records[0]["sample_rate_hz"]
        parameters = dict(sample_rate_hz=rate, pulse_length_s=float(first["Tx Pulse Length"]),
                          start_frequency_hz=float(first["Tx Pulse Start Frequency"]),
                          ramp_rate_hz_per_s=float(first["Tx Ramp Rate"]))
        nominal = nominal_pg_replica(**parameters)
        if reference is None:
            reference = nominal
            chirp_parameters = parameters
        elif parameters != chirp_parameters:
            raise ValueError("PG sequences thay đổi nominal chirp; không dùng chung reference khác grid/chirp.")
        node_times.append((records[0]["start_gps_s"]+records[-1]["last_gps_s"])/2)
        node = {"chunks": [r["chunk"] for r in records], "time_gps_s": node_times[-1],
                "sample_rate_hz": rate, "reconstructed_replica_valid": False,
                "replica_rejection_reason": None, "pg_measurement_valid": False,
                "pg_measurement_rejection_reason": None,
                "replica_mean_amplitude_dn": None, "replica_energy_dn_squared": None,
                "pgcal_peak_dn": None, "inverse_pgcal": None, "normalized_pg_correction": None,
                "nominal_peak_lag_samples": None, "internal_delay_diagnostic_s": None}
        nodes.append(node)
        replica = None
        try:
            replica = reconstruct_replica(*(signals[c] for c in (8, 9, 10, 11, 12)),
                                          denominator_relative_floor=denominator_relative_floor)
            node.update(replica_mean_amplitude_dn=float(np.mean(abs(replica))),
                        replica_energy_dn_squared=float(np.vdot(replica, replica).real))
            peak, lag = correlation_peak(replica, reference, oversampling_factor=oversampling_factor)
            if lag < 0 or lag+len(reference) > len(replica):
                raise ValueError("PG correlation peak nằm ngoài cửa sổ chứa trọn nominal chirp.")
        except ValueError as error:
            node["replica_rejection_reason"] = str(error)
            node["pg_measurement_rejection_reason"] = str(error)
        else:
            node.update(reconstructed_replica_valid=True,
                        pgcal_peak_dn=[peak.real, peak.imag], nominal_peak_lag_samples=lag,
                        internal_delay_diagnostic_s=lag/rate - (
                            configuration["delta_t_guard1_s"]-configuration["delta_t_suppr_s"]))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                inverse = np.complex128(1)/peak
                product = pg_reference/np.complex128(peak)
            if np.isfinite(inverse) and np.isfinite(product) and abs(product) > 0:
                node.update(inverse_pgcal=[float(inverse.real), float(inverse.imag)],
                            normalized_pg_correction=[float(product.real), float(product.imag)],
                            pg_measurement_valid=True)
            else:
                node["pg_measurement_rejection_reason"] = "Inverse PGcal hoặc PGref/PGcal không hữu hạn/khác zero."
        replicas.append(replica)
    if np.any(np.diff(node_times) <= 0):
        raise ValueError("PG sequence timestamps phải tăng nghiêm ngặt.")
    delay_valid = [n["reconstructed_replica_valid"] for n in nodes]
    lags = np.array([np.nan if n["nominal_peak_lag_samples"] is None else n["nominal_peak_lag_samples"] for n in nodes])
    delay = mean_internal_delay(lags, sample_rate_hz=rate,
                                delta_t_guard1_s=configuration["delta_t_guard1_s"],
                                delta_t_suppr_s=configuration["delta_t_suppr_s"], valid=delay_valid) if any(delay_valid) else None
    products = np.array([complex(*n["normalized_pg_correction"]) if n["pg_measurement_valid"] else np.nan+0j for n in nodes])
    model = np.full(len(products), complex(*configuration["constant_pg_model"]))
    quality, chosen = validate_pg(products, model, **configuration["thresholds"],
                                  use_model=configuration["pg_source"] == "Model",
                                  measurement_valid=[n["pg_measurement_valid"] for n in nodes])
    quality["absolute_scale_verified"] = any(n["pg_measurement_valid"] for n in nodes)
    quality["absolute_scale_verification_scope"] = "PGref/PGcal model units; not end-to-end IPF radiometric parity"
    quality["model_amplitude_residual"] = [float(abs(p)-abs(m)) if n["pg_measurement_valid"] else None
                                          for p, m, n in zip(products, model, nodes)]
    quality["model_phase_residual_rad"] = [float(np.angle(p/m)) if n["pg_measurement_valid"] else None
                                          for p, m, n in zip(products, model, nodes)]
    selected = np.array(quality["selected"])
    relative = chosen[selected]/chosen[selected][0] if selected.any() else np.array([], complex)
    for node, reasons, selected_node in zip(nodes, quality["rejection_reasons"], selected):
        node.update(pg_rejection_reasons=reasons, pg_selected=bool(selected_node))
    audit = {"legacy_raw_first_replica_products": None, "legacy_to_nominal_peak_magnitude_ratio": None}
    # A historical diagnostic must never gate either delay or PG correction.
    if all(delay_valid):
        try:
            legacy, legacy_peaks, _ = pg_products(replicas, reference=replicas[0], pg_reference=pg_reference,
                                                  oversampling_factor=oversampling_factor)
            peaks = np.array([complex(*n["pgcal_peak_dn"]) for n in nodes])
            audit.update(legacy_raw_first_replica_products=np.column_stack((legacy.real, legacy.imag)).tolist(),
                         legacy_to_nominal_peak_magnitude_ratio=(abs(legacy_peaks)/abs(peaks)).tolist())
        except ValueError as error:
            audit["legacy_unavailable_reason"] = str(error)
    report = {"raw_file": str(level0.filename), "configuration": configuration, "inventory": inventory,
              "internal_delay": {"available": delay is not None, "unavailable_reason": None if delay is not None else "Không có correlation peak hợp lệ để đo delay.",
                                 "mean_s": delay, "source": "L0 PG correlation + AUX_INS guard/suppression (DAD 4-34)",
                                 "valid_sequence_indices": np.flatnonzero(delay_valid).tolist(),
                                 "per_sequence_delay_s": [n["internal_delay_diagnostic_s"] for n in nodes],
                                 "sample_rate_hz": rate, "peak_lags_samples": [n["nominal_peak_lag_samples"] for n in nodes],
                                 "delta_t_guard1_s": configuration["delta_t_guard1_s"],
                                 "delta_t_suppr_s": configuration["delta_t_suppr_s"],
                                 "oversampling_factor": oversampling_factor,
                                 "denominator_relative_floor": denominator_relative_floor,
                                 "selection": "valid reconstructed correlation peaks; independent of PG amplitude/phase QC"},
              "nodes": nodes, "quality": quality, "image_rx_mean_powers": image_rx_powers,
              "selected_times_gps_s": np.array(node_times)[selected].tolist(),
              "relative_corrections": np.column_stack((relative.real, relative.imag)).tolist(),
              "pg_reference_function": {"source": "nominal PG chirp from L0, DAD 4-25 / Calibration Plan 8.1 9.2-14",
                                        "chirp_parameters_from_l0": chirp_parameters,
                                        "sample_count": len(reference), "amplitude": float(abs(reference[0])),
                                        "energy": float(np.vdot(reference, reference).real),
                                        "fft_convention": "forward unscaled, inverse 1/N (DAD 9.20)",
                                        "peak_interpolation": "FFT resample with amplitude preservation, three-point parabola"},
              "normalization_audit": {
                  **audit,
                  "legacy_peak_units": "DN squared * samples; incompatible with PGref/PGcal model units",
                  "stripmap_receive_gain_ratio": 1.,
                  "stripmap_receive_gain_ratio_source": "Calibration Plan 8.1 9.2.6: single-swath RX settings; no IW/EW swath-1 ratio"},
              "research_parameters": {"oversampling_factor": oversampling_factor,
                                      "denominator_relative_floor": denominator_relative_floor},
              "absolute_radiometric_scale_recovered": False,
              "limitations": ["PG model units follow the nominal 1/N calibration reference; end-to-end radiometric parity is not verified against IPF.",
                              "DCE still applies relative drift only; no absolute radiometric correction is enabled.",
                              "Only constant AUX PG model is supported without an L0-derived ascending-node epoch."]}
    echoes = [r for r in inventory if r["signal_code"] == 0]
    if echoes:
        span = [echoes[0]["start_gps_s"], echoes[-1]["last_gps_s"]]
        report["echo_time_span_gps_s"] = span
        selected_times = np.array(node_times)[selected]
        report["echo_coverage_without_extrapolation"] = bool(quality["interpolation_available"] and selected_times[0] <= span[0] and selected_times[-1] >= span[1])
    return np.array(node_times)[selected], relative, report
