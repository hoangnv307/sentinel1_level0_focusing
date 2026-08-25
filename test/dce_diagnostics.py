"""Compare DCE outlier/coherence strategies on the supplied S6 scene.

Run from the repository root:
    python -m test.dce_diagnostics
"""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sentinel1decoder

from sentinel1_processing.dce import (
    DCESegment,
    Sentinel1DCE,
    prepare_annotation_records,
    robust_polynomial_fit,
    unwrap_fine_dce_dad,
    build_range_blocks,
    _fine_dce_from_accumulators,
    _stream_accc_from_segments,
)
from sentinel1_processing.range_compression import compress_range, estimate_iq_bias


DATA_FILE = Path(
    "data/sao_paulo/"
    "s1a-s6-raw-s-vv-20251226t214356-20251226t214427-062491-07d496.dat"
)
CHUNK13_NPY = DATA_FILE.with_name(DATA_FILE.stem + "_acquisition_chunk_13.npy")
OUTPUT_DIR = Path("test/dce_diagnostics_output")
ZERO_DOPPLER_OFFSET_S = 0.386295160
T0_S = 6.095910535477454e-3

ANNOTATION_RECORDS = prepare_annotation_records([
    {"azimuthTime": "2025-12-26T21:43:59.100490", "t0": T0_S,
     "dataDcPolynomial": [4.152910e1, 1.012491e5, -4.252661e8]},
    {"azimuthTime": "2025-12-26T21:44:14.095877", "t0": T0_S,
     "dataDcPolynomial": [1.141131e1, 1.275731e4, 6.579813e7]},
    {"azimuthTime": "2025-12-26T21:44:25.484967", "t0": T0_S,
     "dataDcPolynomial": [3.263240e1, -2.579159e3, -1.314992e8]},
])

VARIANTS = (
    ("baseline_mad3.5", 3.5, None, False),
    ("mad3.0", 3.0, None, False),
    ("mad2.5", 2.5, None, False),
    ("coherence>=0.30", 3.5, 0.30, False),
    ("coherence_weighted_unwrap", 3.5, None, True),
)

RANGE_LAYOUTS = (
    ("full_B750", 0, None, 750),
    ("full_B1000", 0, None, 1000),
    ("full_B1250", 0, None, 1250),
    ("full_B1500", 0, None, 1500),
    ("full_equal", 0, None, None),
    ("common_intersection_B1000", 0, 19904, 1000),
    ("linear_valid_B1000", 1198, 18751, 1000),
)


def _packet_times(metadata):
    return (
        metadata["Coarse Time"].to_numpy(dtype=float)
        + metadata["Fine Time"].to_numpy(dtype=float)
    )


def _range_times(metadata, sample_count, sample_rate_hz, suppressed_time_s):
    return (
        metadata["Rank"].iloc[0] * metadata["PRI"].iloc[0]
        + metadata["SWST"].iloc[0]
        + suppressed_time_s
        + np.arange(sample_count) / sample_rate_hz
    )


def _estimate_records(output_support, correct_iq_bias=True):
    level0 = sentinel1decoder.Level0File(str(DATA_FILE))
    metadata13 = level0.get_acquisition_chunk_metadata(13)
    metadata14 = level0.get_acquisition_chunk_metadata(14)
    sample_rate_hz = sentinel1decoder.utilities.range_dec_to_sample_rate(
        metadata13["Range Decimation"].iloc[0]
    )
    pri_s = metadata13["PRI"].iloc[0]
    suppressed_time_s = 320.0 / (8.0 * sentinel1decoder.constants.F_REF)
    compression = {
        "sample_rate_hz": sample_rate_hz,
        "pulse_start_frequency_hz": metadata13["Tx Pulse Start Frequency"].iloc[0],
        "pulse_ramp_rate_hz_per_s": metadata13["Tx Ramp Rate"].iloc[0],
        "pulse_length_s": metadata13["Tx Pulse Length"].iloc[0],
    }

    raw13 = np.load(CHUNK13_NPY, mmap_mode="r")
    iq_bias13 = estimate_iq_bias(raw13) if correct_iq_bias else 0.0j
    compressed13, range_times13 = compress_range(
        raw13,
        _range_times(metadata13, raw13.shape[1], sample_rate_hz, suppressed_time_s),
        **compression,
        output=output_support,
        iq_bias=iq_bias13,
    )
    del raw13

    raw14 = level0.get_acquisition_chunk_data(14)
    iq_bias14 = estimate_iq_bias(raw14) if correct_iq_bias else 0.0j
    compressed14, range_times14 = compress_range(
        raw14,
        _range_times(metadata14, raw14.shape[1], sample_rate_hz, suppressed_time_s),
        **compression,
        output=output_support,
        iq_bias=iq_bias14,
    )
    del raw14
    gc.collect()

    azimuth13 = _packet_times(metadata13)
    azimuth14 = _packet_times(metadata14)
    estimator = Sentinel1DCE.for_s6_research(prf_hz=1.0 / pri_s)
    scene_stop_s = azimuth14[-1] + pri_s
    records, scene = estimator.estimate_segments(
        [
            DCESegment(compressed13, range_times13, azimuth13, "chunk13"),
            DCESegment(compressed14, range_times14, azimuth14, "chunk14"),
        ],
        t0_s=T0_S,
        slice_start_times_s=[azimuth13[0]],
        last_slice_stop_time_s=scene_stop_s,
        product_start_time_s=azimuth13[0] + ZERO_DOPPLER_OFFSET_S,
        product_stop_time_s=scene_stop_s + ZERO_DOPPLER_OFFSET_S,
        zero_dop_minus_acq_time_s=ZERO_DOPPLER_OFFSET_S,
        return_prepared_scene=True,
    )
    alignment = scene.alignment_summary()
    accumulators = [
        _stream_accc_from_segments(scene, record.block)[:3]
        for record in records
    ]
    config = estimator.config
    del scene, compressed13, compressed14
    gc.collect()
    return (
        records, range_times13, 1.0 / pri_s, alignment, accumulators, config,
        (iq_bias13, iq_bias14),
    )


def _sweep_range_layouts(accumulators, range_times, prf_hz, config):
    rows = []
    curves = {1: [], 2: [], 3: []}
    for name, roi_start, roi_stop, block_size in RANGE_LAYOUTS:
        if roi_start >= range_times.size or (
            roi_stop is not None and roi_stop > range_times.size
        ):
            continue
        layout_config = replace(
            config,
            range_roi_start=roi_start,
            range_roi_stop=roi_stop,
            range_block_size_samples=block_size,
            outlier_sigma=3.0,
        )
        blocks = build_range_blocks(range_times.size, range_times, layout_config)
        block_times = np.array([block.center_slant_range_time_s for block in blocks])
        for record_index, ((c_range, p0_range, p1_range), annotation) in enumerate(
            zip(accumulators, ANNOTATION_RECORDS), start=1
        ):
            fine, _, coherence = _fine_dce_from_accumulators(
                c_range, p0_range, p1_range, blocks, prf_hz
            )
            unwrapped = unwrap_fine_dce_dad(
                block_times, fine, prf_hz, fft_length=config.unwrap_fft_length
            )
            coefficients, used, fit_rms_hz = robust_polynomial_fit(
                block_times,
                unwrapped,
                t0_s=T0_S,
                outlier_sigma=layout_config.outlier_sigma,
            )
            estimated = np.polynomial.polynomial.polyval(
                range_times - T0_S, coefficients
            )
            annotated = np.polynomial.polynomial.polyval(
                range_times - T0_S, annotation["dataDcPolynomial"]
            )
            error = estimated - annotated
            rows.append({
                "record": record_index,
                "layout": name,
                "block_size": block_size,
                "roi_start": roi_start,
                "roi_stop": range_times.size if roi_stop is None else roi_stop,
                "used_points": int(np.count_nonzero(used)),
                "fit_rms_hz": fit_rms_hz,
                "mean_coherence": float(np.mean(coherence)),
                "bias_hz": float(np.mean(error)),
                "rmse_hz": float(np.sqrt(np.mean(error**2))),
                "max_abs_error_hz": float(np.max(np.abs(error))),
            })
            curves[record_index].append((name, estimated))
    return pd.DataFrame(rows), curves


def _refit(record, sigma, min_coherence, weighted_unwrap, prf_hz):
    fine = unwrap_fine_dce_dad(
        record.range_times_s,
        record.fine_baseband_hz,
        prf_hz,
        fft_length=4096,
        weights=record.coherence if weighted_unwrap else None,
    )
    eligible = np.isfinite(fine)
    if min_coherence is not None:
        eligible &= record.coherence >= min_coherence
    indices = np.flatnonzero(eligible)
    if indices.size < 3:
        raise RuntimeError("Coherence filter left fewer than three fine estimates.")

    coefficients, local_mask, fit_rms_hz = robust_polynomial_fit(
        record.range_times_s[indices],
        fine[indices],
        t0_s=T0_S,
        degree=2,
        outlier_sigma=sigma,
        max_iterations=5,
    )
    used = np.zeros(fine.size, dtype=bool)
    used[indices[local_mask]] = True
    return coefficients, fine, used, fit_rms_hz


def run(show=True, output_support="valid", correct_iq_bias=True):
    started = time.perf_counter()
    records, evaluation_times, prf_hz, alignment, accumulators, config, iq_biases = (
        _estimate_records(output_support, correct_iq_bias)
    )
    if len(records) != 3:
        raise AssertionError(f"Expected three DCE records, got {len(records)}.")
    shift14 = alignment[1]["source_start_index"]
    if not np.isclose(shift14, 81.25, atol=0.01):
        raise AssertionError(f"Unexpected chunk-14 range shift: {shift14}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    plot_data = {}

    for record_index, (record, annotation) in enumerate(
        zip(records, ANNOTATION_RECORDS), start=1
    ):
        annotation_hz = np.polynomial.polynomial.polyval(
            evaluation_times - T0_S,
            annotation["dataDcPolynomial"],
        )
        for name, sigma, min_coherence, weighted_unwrap in VARIANTS:
            coefficients, fine, used, fit_rms_hz = _refit(
                record, sigma, min_coherence, weighted_unwrap, prf_hz
            )
            estimated_hz = np.polynomial.polynomial.polyval(
                evaluation_times - T0_S, coefficients
            )
            error_hz = estimated_hz - annotation_hz
            rows.append({
                "record": record_index,
                "variant": name,
                "used_points": int(np.count_nonzero(used)),
                "rejected_points": int(used.size - np.count_nonzero(used)),
                "mean_used_coherence": float(np.mean(record.coherence[used])),
                "fit_rms_hz": fit_rms_hz,
                "bias_hz": float(np.mean(error_hz)),
                "mae_hz": float(np.mean(np.abs(error_hz))),
                "rmse_hz": float(np.sqrt(np.mean(error_hz**2))),
                "max_abs_error_hz": float(np.max(np.abs(error_hz))),
                "C0": coefficients[0],
                "C1": coefficients[1],
                "C2": coefficients[2],
            })
        plot_data[record_index] = annotation_hz

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    layout_results, layout_curves = _sweep_range_layouts(
        accumulators, evaluation_times, prf_hz, config
    )
    layout_results.to_csv(OUTPUT_DIR / "range_layout_metrics.csv", index=False)
    print("\nAlignment:")
    print("Range-compression support:", output_support)
    print("I/Q bias correction:", correct_iq_bias, iq_biases)
    print(pd.DataFrame(alignment).to_string(index=False))
    print("\nMetrics:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nBest RMSE per record:")
    best = results.loc[results.groupby("record")["rmse_hz"].idxmin()]
    print(best[["record", "variant", "rmse_hz", "used_points"]].to_string(index=False))
    print("\nRange-layout sweep (MAD 3.0):")
    print(layout_results.to_string(index=False, float_format=lambda value: f"{value:.6g}"))

    for record_index, annotation_hz in plot_data.items():
        figure, axis = plt.subplots(figsize=(10, 7))
        range_ms = evaluation_times * 1e3
        axis.plot(range_ms, annotation_hz, color="black", linewidth=2.5,
                  label="Annotation")
        curves = dict(layout_curves[record_index])
        axis.plot(range_ms, curves["full_B1000"], linewidth=1.8,
                  label="Before fix (B=1000)")
        axis.plot(range_ms, curves["full_B1250"], linewidth=1.8,
                  label="After fix (B=1250)")
        axis.set_title(f"DCE record {record_index}: polynomial comparison")
        axis.set_xlabel("Slant-range time [ms]")
        axis.set_ylabel("Doppler centroid [Hz]")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        output = OUTPUT_DIR / f"dce_record_{record_index}.png"
        figure.savefig(output, dpi=160)
        print("Saved:", output)
        if show:
            plt.show()
        else:
            plt.close(figure)

    print(f"Total runtime: {time.perf_counter() - started:.2f} s")
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--support", choices=("valid", "same"), default="valid")
    parser.add_argument(
        "--no-iq-bias", action="store_false", dest="iq_bias", default=True
    )
    arguments = parser.parse_args()
    run(
        show=not arguments.no_show,
        output_support=arguments.support,
        correct_iq_bias=arguments.iq_bias,
    )
