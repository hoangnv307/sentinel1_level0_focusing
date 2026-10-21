"""Inventory and research PG drift for the local S1A/S4/VV acquisition."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import sentinel1decoder

from ..pre_processing.internal_calibration import (
    extract_pulse, reconstruct_replica, correlation_peak, pg_products, validate_pg,
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
            "pg_model_interval_s": float(calibration.findtext("pgProductModel/pgModelInterval")),
            "delta_t_guard1_s": float(ins.findtext("deltaTGuard1")),
            "delta_t_suppr_s": float(ins.findtext("deltaTSuppr"))}


def analyze_calibration(level0, product, *, oversampling_factor, denominator_relative_floor):
    """No L1 input; return selected acquisition times, relative PG correction, report.

    Absolute DN/PG scale and inter-swath gain ratio are not recovered here.
    The opt-in DCE correction is normalized to its first selected node.
    """
    if not Path(level0.filename).name.startswith("s1a-s4-raw-s-vv-"):
        raise ValueError("Adapter calibration này chỉ hỗ trợ sản phẩm S1A/S4/VV hiện tại.")
    configuration = read_configuration(product)
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
    if len(groups) < 2:
        raise ValueError("Không đủ hai PG calibration sequence.")
    replicas, node_times, nodes = [], [], []
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
        replica = reconstruct_replica(*(signals[c] for c in (8, 9, 10, 11, 12)),
                                      denominator_relative_floor=denominator_relative_floor)
        rate = records[0]["sample_rate_hz"]
        n = int(np.ceil(float(first["Tx Pulse Length"])*rate))
        t = np.arange(n)/rate
        nominal = np.exp(2j*np.pi*(float(first["Tx Pulse Start Frequency"])*t +
                                  .5*float(first["Tx Ramp Rate"])*t*t))/n
        peak, lag = correlation_peak(replica, nominal, oversampling_factor=oversampling_factor)
        node_times.append((records[0]["start_gps_s"]+records[-1]["last_gps_s"])/2)
        replicas.append(replica)
        nodes.append({"chunks": [r["chunk"] for r in records], "time_gps_s": node_times[-1],
                      "nominal_mean_correlation_peak": [peak.real, peak.imag],
                      "inverse_nominal_peak_diagnostic": [float((1/peak).real), float((1/peak).imag)],
                      "nominal_peak_lag_samples": lag, "sample_rate_hz": rate,
                      "internal_delay_diagnostic_s": lag/rate - (
                          configuration["delta_t_guard1_s"]-configuration["delta_t_suppr_s"])})
    products, peaks, lags = pg_products(replicas, pg_reference=complex(*configuration["pg_reference"]),
                                       oversampling_factor=oversampling_factor)
    model = np.full(len(products), complex(*configuration["constant_pg_model"]))
    quality, chosen = validate_pg(products, model, **configuration["thresholds"],
                                  use_model=configuration["pg_source"] == "Model")
    quality["absolute_scale_verified"] = False
    selected = np.array(quality["selected"])
    relative = chosen[selected]/chosen[selected][0]
    for index, node in enumerate(nodes):
        node.update(compressed_first_replica_peak=[float(peaks[index].real), float(peaks[index].imag)],
                    first_replica_lag_samples=float(lags[index]),
                    pg_product_raw_fft_units=[float(products[index].real), float(products[index].imag)])
    report = {"raw_file": str(level0.filename), "configuration": configuration, "inventory": inventory,
              "nodes": nodes, "quality": quality, "image_rx_mean_powers": image_rx_powers,
              "selected_times_gps_s": np.array(node_times)[selected].tolist(),
              "relative_corrections": np.column_stack((relative.real, relative.imag)).tolist(),
              "research_parameters": {"oversampling_factor": oversampling_factor,
                                      "denominator_relative_floor": denominator_relative_floor},
              "absolute_radiometric_scale_recovered": False,
              "limitations": ["PG products use raw FFT/DN units; absolute calibration/normalization is not verified against IPF.",
                              "Relative normalization cancels a constant PG reference and swath gain ratio; no absolute ratio is invented.",
                              "Only constant AUX PG model is supported without an L0-derived ascending-node epoch."]}
    echoes = [r for r in inventory if r["signal_code"] == 0]
    if echoes:
        span = [echoes[0]["start_gps_s"], echoes[-1]["last_gps_s"]]
        report["echo_time_span_gps_s"] = span
        selected_times = np.array(node_times)[selected]
        report["echo_coverage_without_extrapolation"] = bool(selected_times[0] <= span[0] and selected_times[-1] >= span[1])
    return np.array(node_times)[selected], relative, report
