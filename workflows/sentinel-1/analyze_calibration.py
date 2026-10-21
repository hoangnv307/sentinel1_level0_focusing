"""Kiểm kê calibration L0 và dựng PG drift tương đối; L1 chỉ dùng đối chứng cuối."""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import sentinel1decoder

from sentinel1_processing.utils.pg_calibration_input import analyze_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, default=Path.home()/"sar_data/focusing_data/product_1")
    parser.add_argument("--output", type=Path, default=Path("output/sentinel-1/pg_calibration"))
    parser.add_argument("--oversampling-factor", type=int, default=16)
    parser.add_argument("--denominator-relative-floor", type=float, default=0.)
    args = parser.parse_args()
    raw = list(args.product.glob("level-0/s1a-s4-raw-s-vv-*.dat"))
    if len(raw) != 1:
        raise ValueError("Cần đúng một L0 S1A/S4/VV.")
    level0 = sentinel1decoder.Level0File(str(raw[0]))
    times, corrections, report = analyze_calibration(
        level0, args.product, oversampling_factor=args.oversampling_factor,
        denominator_relative_floor=args.denominator_relative_floor)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    # Read L1 only after the L0/AUX processing and selection have completed.
    annotation = list(args.product.glob("level-1/s1a-s4-slc-vv-*.xml"))
    if len(annotation) != 1:
        raise ValueError("Cần đúng một annotation S4/VV để đối chứng.")
    root = ET.parse(annotation[0]).getroot()
    offset = float(root.findtext("imageAnnotation/imageInformation/zeroDopMinusAcqTime"))
    annotated = root.findall("generalAnnotation/replicaInformationList/replicaInformation/replicaList/replica")
    ref_times = np.array([float(Time(n.findtext("azimuthTime"), scale="utc").gps) for n in annotated])
    values = np.array([float(n.findtext("pgProductAmplitude"))*np.exp(1j*float(n.findtext("pgProductPhase")))
                       for n in annotated])
    report["l1_diagnostic_only"] = {"path": str(annotation[0]), "time_offset_s": offset,
        "times_gps_s": ref_times.tolist(), "complex_values": np.column_stack((values.real, values.imag)).tolist(),
        "direct_relative_correction": np.column_stack(((values/values[0]).real, (values/values[0]).imag)).tolist()}
    if len(ref_times) == len(report["nodes"]):
        report["l1_diagnostic_only"]["aligned_time_residual_s"] = (
            np.array([n["time_gps_s"] for n in report["nodes"]])+offset-ref_times).tolist()
        report["l1_diagnostic_only"]["inverse_pgcal_amplitude_relative_error"] = [
            float(abs(complex(*n["inverse_pgcal"]))/abs(v)-1) if n["inverse_pgcal"] is not None else None
            for n, v in zip(report["nodes"], values)]
        report["l1_diagnostic_only"]["inverse_pgcal_phase_error_rad"] = [
            float(np.angle(complex(*n["inverse_pgcal"])/v)) if n["inverse_pgcal"] is not None else None
            for n, v in zip(report["nodes"], values)]
        report["l1_diagnostic_only"]["comparison_note"] = "Compare 1/PGcal to annotation without fitting a complex scale; PGref is not in annotation."
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if report["quality"]["interpolation_available"]:
        axes[0].plot(times-times[0], abs(corrections), "o-", label="L0/AUX, nominal 1/N reference")
        axes[1].plot(times-times[0], np.angle(corrections), "o-", label="L0/AUX")
    else:
        axes[0].text(.02, .05, report["quality"]["interpolation_unavailable_reason"], transform=axes[0].transAxes)
    axes[0].plot(ref_times-ref_times[0], abs(values/values[0]), "x--", label="L1 inverse-PG, direct normalization")
    axes[1].plot(ref_times-ref_times[0], np.angle(values/values[0]), "x--", label="L1")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    axes[0].set_ylabel("Relative amplitude")
    axes[1].set(xlabel="Time from first node [s]", ylabel="Relative phase [rad]")
    figure.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output/"pg_drift.png", dpi=160)
    plt.close(figure)
    (args.output/"calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{len(report['inventory'])} chunks; {len(report['nodes'])} PG nodes; source={report['quality']['source']}")
    print(f"Delay available={report['internal_delay']['available']}; PG interpolation available={report['quality']['interpolation_available']}")
    print("PG relative:", corrections)
    print(f"Đã ghi {args.output/'calibration.json'}")


if __name__ == "__main__":
    main()
