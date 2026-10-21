"""Phân tích raw I/Q S4/VV theo DAD §4.1; annotation L1 chỉ dùng đối chiếu sau tính toán."""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import sentinel1decoder

import estimate_doppler_centroid as demo
from audit_doppler_centroid import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, default=Path.home()/"sar_data/focusing_data/product_1")
    parser.add_argument("--output", type=Path, default=Path("output/sentinel-1/raw_analysis"))
    parser.add_argument("--batch-lines", type=int, default=demo.BATCH_LINES)
    parser.add_argument("--azimuth-start-line", type=int, default=0)
    parser.add_argument("--azimuth-step", type=int, default=1)
    parser.add_argument("--max-azimuth-lines", type=int, default=None)
    parser.add_argument("--range-start-sample", type=int, default=0)
    parser.add_argument("--range-stop-sample", type=int, default=None)
    args = parser.parse_args()
    raw = demo.one(args.product.glob("level-0/s1a-s4-raw-s-vv-*.dat"), "raw S4/VV")
    level0 = sentinel1decoder.Level0File(str(raw))
    selection = {key: getattr(args, key) for key in (
        "batch_lines", "azimuth_start_line", "azimuth_step", "max_azimuth_lines",
        "range_start_sample", "range_stop_sample")}
    analysis = demo.analyze_echoes(level0, **selection)
    auxiliary = demo.read_auxiliary(args.product)
    correction = demo.correction_parameters(
        analysis, correct_iq_bias=auxiliary["correct_iq_bias"], correct_iq_gain=auxiliary["correct_iq_gain"],
        correct_iq_quadrature=auxiliary["correct_iq_quadrature"])
    # L1 is read only after the independent L0 analysis and parameter selection.
    annotation_path = demo.one(args.product.glob("level-1/s1a-s4-slc-vv-*.xml"), "annotation")
    annotation = ET.parse(annotation_path).getroot()
    annotated = demo.one(annotation.findall("generalAnnotation/rawDataAnalysisList/rawDataAnalysis"), "raw analysis L1")
    quality = annotation.find("qualityInformation/qualityDataList/qualityData/rawDataAnalysisQuality")
    theta_literal = float(annotated.findtext("iqQuadratureDeparture"))
    l1 = {"i_bias": float(annotated.findtext("iBias")), "q_bias": float(annotated.findtext("qBias")),
          "gain_imbalance": float(annotated.findtext("iqGainImbalance")),
          "quadrature_literal": theta_literal,
          "support": {e.tag: float(e.text) for e in annotated.find("support")},
          "quality_flags_literal": {} if quality is None else {
              e.tag: e.text for e in quality if e.tag.endswith("Flag")},
          "quadrature_unit_note": "Product Spec table does not state units; report both radian and degree interpretations, not an input choice."}
    comparisons = {"i_bias_difference": analysis["mean_i"]-l1["i_bias"],
                   "q_bias_difference": analysis["mean_q"]-l1["q_bias"],
                   "gain_difference": None if analysis["iq_gain_imbalance"] is None else analysis["iq_gain_imbalance"]-l1["gain_imbalance"]}
    theta = analysis["iq_quadrature_departure_rad"]
    comparisons["quadrature_rad_difference_if_l1_radians"] = None if theta is None else theta-theta_literal
    comparisons["quadrature_rad_difference_if_l1_degrees"] = None if theta is None else theta-np.deg2rad(theta_literal)
    payload = {"analysis": analysis, "correction": correction, "auxiliary": auxiliary,
               "l1_diagnostic_only": l1, "comparison": comparisons,
               "conventions": ["population std (ddof=0)", "per-line centered covariance; equal Fisher-z line weights",
                               "outside_null_bounds true means departure, not IPF SignificanceFlag",
                               "I-bias upper sign and quadrature null lower sign use symmetric zero bounds",
                               "significance report never silently switches measured values to AUX defaults"],
               "fingerprints": [fingerprint(p) for p in (raw, annotation_path, *map(Path, auxiliary["files"]),
                   Path(__file__), Path(demo.__file__),
                   Path("src/sentinel1_processing/common/raw_data_analysis.py"))]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"raw_data_analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{analysis['line_count']} echo × {analysis['range_sample_count']} range samples; "
          f"bias={analysis['mean_i']:+.12f}{analysis['mean_q']:+.12f}j; "
          f"gain={analysis['iq_gain_imbalance']}; theta_rad={theta}")
    print(f"Đã ghi {args.output/'raw_data_analysis.json'}")


if __name__ == "__main__":
    main()
