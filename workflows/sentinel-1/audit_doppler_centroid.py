"""Kiểm tra tham số và độ nhạy DCE S4/VV; không tối ưu cấu hình theo L1."""

import argparse
import copy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from xml.etree import ElementTree as ET

import numpy as np
import sentinel1decoder
from sentinel1decoder.enums import SignalType

import estimate_doppler_centroid as demo


def compare(record, annotated):
    """Chỉ so trên giao range support; không ngoại suy Fine DC của L1."""
    points = annotated.findall("fineDceList/fineDce")
    ref_tau = np.array([float(p.findtext("slantRangeTime")) for p in points])
    ref_fine = np.array([float(p.findtext("frequency")) for p in points])
    tau = record.range_times_s
    common = (tau >= ref_tau[0] - 1e-10) & (tau <= ref_tau[-1] + 1e-10)
    if common.sum() < 3:
        raise ValueError("Không đủ giao range support để đối chiếu L1.")
    ref = np.interp(tau[common], ref_tau, ref_fine)
    polynomial = np.fromstring(annotated.findtext("dataDcPolynomial"), sep=" ")
    geom = np.fromstring(annotated.findtext("geometryDcPolynomial"), sep=" ")
    x = tau[common] - float(annotated.findtext("t0"))
    residual=np.empty(0)
    if record.fine_baseband_hz.size==tau.size:
        fine=record.fine_baseband_hz[common]
        finite=np.isfinite(fine)&np.isfinite(ref)
        residual=fine[finite]-ref[finite]
    fitted = record.evaluate(tau[common]) - np.polynomial.polynomial.polyval(x, polynomial)
    geom_residual=(None if record.geometry_dc_polynomial is None else np.polynomial.polynomial.polyval(
        tau[common]-record.t0_s,record.geometry_dc_polynomial)-np.polynomial.polynomial.polyval(x,geom))
    from sentinel1_processing.utils.dce_job import real_values
    coherence=record.coherence[np.isfinite(record.coherence)]
    return {
        "start_line": record.block.start_line, "stop_line": record.block.stop_line,
        "acquisition_center_gps_s": record.block.azimuth_time_s,
        "compared_range_points": int(common.sum()),
        "fine_rmse_hz": float(np.sqrt(np.mean(residual**2))) if residual.size else None,
        "fine_bias_hz": float(np.mean(residual)) if residual.size else None,
        "fine_max_abs_hz": float(np.max(np.abs(residual))) if residual.size else None,
        "polynomial_rmse_hz": float(np.sqrt(np.mean(fitted**2))),
        "geometry_rmse_hz": None if geom_residual is None else float(np.sqrt(np.mean(geom_residual**2))),
        "fit_rms_hz": float(record.rms_error_hz) if np.isfinite(record.rms_error_hz) else None,
        "valid_points": int(record.valid_mask.sum()),
        "ambiguity": record.ambiguity_number,
        "selected_source": record.selected_source,
        "selection_reason": record.selection_reason,
        "coherence_min": float(coherence.min()) if coherence.size else None,
        "coherence_max": float(coherence.max()) if coherence.size else None,
        "range_times_s": tau.tolist(),
        "fine_baseband_hz": real_values(record.fine_baseband_hz),
        "data_dc_polynomial": None if record.data_dc_polynomial is None else record.data_dc_polynomial.tolist(),
    }


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inventory(level0, annotation, aux, reference):
    chunks = []
    times = []
    for chunk in level0.acquisition_chunks:
        metadata = level0.get_acquisition_chunk_metadata(chunk)
        if metadata["Signal Type"].iloc[0] != SignalType.ECHO:
            continue
        eta = metadata["Coarse Time"].to_numpy(float) + metadata["Fine Time"].to_numpy(float)
        times.extend(eta)
        fields = ["PRI", "Range Decimation", "Tx Pulse Length", "Tx Ramp Rate",
                  "Tx Pulse Start Frequency", "Rank", "Swath Number", "Rx Channel ID",
                  "Rx Gain", "SWST", "SWL", "Number of Quads"]
        values = {f: [str(x) for x in metadata[f].unique()] for f in fields}
        chunks.append({"chunk": int(chunk), "lines": len(metadata), "header_values": values,
                       "start_gps_s": float(eta[0]), "last_gps_s": float(eta[-1])})
    times = np.asarray(times)
    epi = level0.ephemeris
    q = epi[[f"Q{i} Attitude Quaternion" for i in range(4)]].dropna().to_numpy(float)
    checks = {
        "max_pri_error_s": float(np.max(np.abs(np.diff(times) - float(chunks[0]["header_values"]["PRI"][0])))),
        "max_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1))),
        "orbit_gps_span": [float(epi["POD Solution Data Timestamp"].min()),
                           float(epi["POD Solution Data Timestamp"].max())],
        "attitude_gps_span": [float(epi["Attitude Data Timestamp"].min()),
                              float(epi["Attitude Data Timestamp"].max())],
        "pg_extrapolation_at_scene_start_s": max(0., float(reference.pg_times[0] - (times[0] + reference.time_offset_s))),
    }
    checks["scene_gps_span"] = [float(times[0]), float(times[-1])]
    for name in ("orbit_gps_span", "attitude_gps_span"):
        checks[name + "_scene_edge_overrun_s"] = [
            max(0., float(checks[name][0] - times[0])),
            max(0., float(times[-1] - checks[name][1])),
        ]
    assert checks["max_pri_error_s"] < 0.05 * float(chunks[0]["header_values"]["PRI"][0])
    groups = ["adsHeader", "generalAnnotation/productInformation",
              "imageAnnotation/processingInformation", "imageAnnotation/imageInformation"]
    return {"chunks": chunks, "checks": checks, "auxiliary": aux,
            "reference_input": reference.summary(),
            "l1_metadata": {p: ET.tostring(annotation.find(p), encoding="unicode") for p in groups}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, default=Path.home() / "sar_data/focusing_data/product_1")
    parser.add_argument("--output", type=Path, default=Path("output/sentinel-1/dce_audit"))
    parser.add_argument("--input-case", help="Chỉ chạy một preprocessing case; reference bao gồm estimator sweep.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = demo.one(args.product.glob("level-0/s1a-s4-raw-s-vv-*.dat"), "raw S4/VV")
    annotation_path = demo.one(args.product.glob("level-1/s1a-s4-slc-vv-*.xml"), "annotation")
    annotation = ET.parse(annotation_path).getroot()
    annotated = annotation.findall("dopplerCentroid/dcEstimateList/dcEstimate")
    aux = demo.read_auxiliary(args.product)
    reference = demo.ReferenceInput(annotation, aux)
    level0 = sentinel1decoder.Level0File(str(raw))
    geometry = demo.geometry.Estimator.from_level0_product(level0, 299792458. / aux["radar_frequency_hz"])
    geometry_cache = {}

    def provider(eta, tau):
        geometry.orbit.validate_time_coverage(np.array([eta]))
        key = (float(eta), tuple(tau))
        if key not in geometry_cache:
            geometry_cache[key] = geometry.estimate(eta, tau * 299792458. / 2)
        return geometry_cache[key]

    sources = [raw, annotation_path, *map(Path, aux["files"]), Path(__file__),
               Path(demo.__file__), Path("src/sentinel1_processing/utils/dce_reference_input.py")]
    sources += [Path("src/sentinel1_processing/doppler_centroid") / f for f in ("estimation.py", "geometry.py")]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "fingerprints": [fingerprint(p) for p in sources],
              "inventory": inventory(level0, annotation, aux, reference),
              "cases": [], "rejected_cases": []}

    def save():
        (args.output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        rows = [{"case": c["name"], "block": i + 1, **{k: v for k, v in b.items() if not isinstance(v, list)}}
                for c in report["cases"] for i, b in enumerate(c["blocks"])]
        if rows:
            with (args.output / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    def evaluate(name, segments, pri, config, starts, t0, *, same_time_support=True):
        # Không để build_azimuth_blocks âm thầm clamp cửa sổ thử nghiệm.
        n = sum(s.num_azimuth_lines for s in segments)
        assert all(0 <= s <= n - config.azimuth_block_size_lines for s in starts)
        begin = time.monotonic()
        records = demo.estimation.Estimator(1 / pri, config).estimate_segments(
            segments, custom_azimuth_starts=starts, dce_range_start_s=t0, t0_s=t0,
            geometry_dc_provider=provider,
        )
        assert [r.block.start_line for r in records] == starts
        blocks = [compare(r, a) for r, a in zip(records, annotated)]
        assert len(blocks) == len(annotated) == len(records)
        report["cases"].append({"name": name, "config": asdict(config),
                                "same_l1_time_support": same_time_support,
                                "estimation_seconds": time.monotonic() - begin, "blocks": blocks})
        save()
        print(name, [(None if b["fine_rmse_hz"] is None else round(b["fine_rmse_hz"],4),
                     None if b["fit_rms_hz"] is None else round(b["fit_rms_hz"],4)) for b in blocks],flush=True)

    variants = [("basic", None), ("reference", reference)]
    for name, attribute, value in [("no_bias", "bias", 0j), ("no_rx", "rx_enabled", False),
                                   ("no_pg", "pg_corrections", np.ones_like(reference.pg_corrections)),
                                   ("no_eap", "eap_enabled", False), ("no_rsl", "rsl_enabled", False),
                                   ("no_delay", "internal_delay_s", 0.)]:
        modified = copy.deepcopy(reference)
        setattr(modified, attribute, value)
        variants.append((name, modified))

    if args.input_case:
        variants = [(name, value) for name, value in variants if name == args.input_case]
        if not variants:
            parser.error(f"Không có input case {args.input_case!r}.")

    save()
    for name, correction in variants:
        begin = time.monotonic()
        with TemporaryDirectory(prefix="audit-", dir=args.output) as directory:
            try:
                segments, pri, rate, diagnostics = demo.compress_echoes(
                    level0, aux, directory, correction,
                    correct_rx_variation=False if name == "no_rx" else None,
                )
            except ValueError as error:
                # Bỏ delay guard có thể mở rộng ROI ra ngoài LUT; không ngoại suy để ép case chạy.
                if name != "no_delay" or "Range support nằm ngoài bank EAP" not in str(error):
                    raise
                report["rejected_cases"].append({"name": name, "reason": str(error)})
                save()
                print(f"{name}: từ chối đúng bởi EAP support guard: {error}", flush=True)
                continue
            config, starts, t0, _ = demo.infer_configuration(
                annotation, segments, pri, rate, aux["rms_threshold_hz"],
                diagnostics[0]["hardware_delay_far_throwaway_samples"],
            )
            evaluate(name, segments, pri, config, starts, t0)
            report["cases"][-1]["input_diagnostics"] = diagnostics
            report["cases"][-1]["preparation_and_estimation_seconds"] = time.monotonic() - begin
            if name == "reference":
                changes = [("phase_weight", {"accc_range_weighting": "phase"}),
                           ("coherence_fit", {"fit_weighting": "coherence"}),
                           ("uniform_unwrap", {"unwrap_weighting": "uniform"})]
                changes += [(f"fft_{n}", {"unwrap_fft_length": n}) for n in (128, 1024, 8192)]
                changes += [(f"outlier_{s}", {"outlier_sigma": s}) for s in (2., 2.5, 5.)]
                changes += [(f"iterations_{n}", {"max_fit_iterations": n}) for n in (1, 10)]
                changes += [(f"width_{n}", {"range_block_size_samples": n}) for n in (500, 1500)]
                changes += [(f"ranges_{n}", {"num_range_blocks": n}) for n in (10, 40)]
                changes += [(f"degree_{n}", {"polynomial_degree": n}) for n in (1, 3)]
                changes += [(f"threshold_{n}", {"rms_threshold_hz": n}) for n in (1., 100.)]
                for case, change in changes:
                    evaluate(case, segments, pri, replace(config, **change), starts, t0)
                for size in (4000, 8000):
                    alternate_starts = [0, sum(s.num_azimuth_lines for s in segments) - size]
                    evaluate(f"azimuth_{size}", segments, pri,
                             replace(config, azimuth_block_size_lines=size), alternate_starts, t0,
                             same_time_support=False)
                evaluate("shift_inward_554", segments, pri, config, [554, starts[1] - 554], t0,
                         same_time_support=False)
            save()
            del segments
    print(f"Đã lưu {len(report['cases'])} case DCE và {len(report['rejected_cases'])} case bị chặn tại {args.output}", flush=True)


if __name__ == "__main__":
    main()
