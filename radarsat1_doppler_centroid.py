"""Ước lượng và so sánh Doppler centroid RADARSAT-1."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radarsat1_processing import (
    compare_doppler_centroid,
    estimate_doppler_centroid,
    read_l1_reference,
    read_metadata,
)
from sentinel1_processing.dce_plotting import plot_comparisons


def _one(pattern: str) -> Path:
    files = list(Path("data/radarsat-1").glob(pattern))
    if len(files) != 1:
        raise SystemExit(f"Cần đúng một file khớp {pattern}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--l1-metadata", type=Path, default=None)
    parser.add_argument("--azimuth-lines", type=int, default=6000)
    parser.add_argument("--range-blocks", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("output/radarsat-1/radarsat1_dce.png")
    )
    parser.add_argument(
        "--residual-output",
        type=Path,
        default=Path("output/radarsat-1/radarsat1_dce_residual.png"),
    )
    args = parser.parse_args()

    raw = args.raw or _one("**/*.raw")
    l1_path = args.l1_metadata or _one("**/*.L.txt")
    estimated = estimate_doppler_centroid(
        raw,
        azimuth_lines=args.azimuth_lines,
        num_range_blocks=args.range_blocks,
    )
    result = compare_doppler_centroid(estimated, read_l1_reference(l1_path))
    metadata = read_metadata(raw)
    l0_pixels = (
        np.asarray(result["range_times_s"]) - metadata.range_start_time_s
    ) * metadata.sample_rate_hz
    l0_crt_hz = np.polynomial.polynomial.polyval(
        l0_pixels, metadata.doppler_centroid_coefficients
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_comparisons([result])[0]
    axis = figure.axes[0]
    axis.lines[0].remove()
    axis.lines[0].set_label("Estimated data DC polynomial")
    axis.lines[1].set_label("Orbit geometry DC polynomial")
    axis.collections[0].set_label(
        f"Fine DC points ({np.count_nonzero(estimated.valid_mask)})"
    )
    axis.plot(
        np.asarray(result["range_times_s"]) * 1e3,
        l0_crt_hz,
        color="0.45",
        linestyle="-.",
        label="L0 CRT reference polynomial",
    )
    axis.set_title(f"RADARSAT-1 DCE: fit RMS={estimated.rms_error_hz:.3f} Hz")
    axis.legend()
    figure.savefig(args.output, dpi=160)
    plt.close(figure)

    range_km = np.asarray(result["range_times_s"]) * 299_792_458.0 / 2e3
    residual, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        range_km,
        np.asarray(result["estimated_hz"]) - np.asarray(result["annotation_hz"]),
        label="Estimated data DC − L1 CRT",
    )
    axis.plot(
        range_km,
        np.asarray(result["geometry_hz"]) - np.asarray(result["annotation_hz"]),
        label="Orbit geometry DC − L1 CRT",
    )
    axis.plot(
        range_km,
        np.asarray(result["estimated_hz"]) - l0_crt_hz,
        color="0.45",
        linestyle="-.",
        label="Estimated data DC − L0 CRT",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Slant range [km]", ylabel="Residual [Hz]", title="RADARSAT-1 DCE residuals")
    axis.grid(True, alpha=0.3)
    axis.legend()
    residual.tight_layout()
    residual.savefig(args.residual_output, dpi=160)
    plt.close(residual)

    print(f"Fine DC fit RMS: {estimated.rms_error_hz:.3f} Hz")
    print(f"Estimated vs L1 CRT RMSE: {result['rmse_hz']:.3f} Hz")
    print(
        "Estimated vs L0 CRT RMSE: "
        f"{np.sqrt(np.mean((np.asarray(result['estimated_hz']) - l0_crt_hz) ** 2)):.3f} Hz"
    )
    print(f"PRF ambiguity number: {estimated.ambiguity_number}")
    print(f"Data DC polynomial [Hz, Hz/s, Hz/s²]: {estimated.data_dc_polynomial}")
    print(f"Geometry DC polynomial [Hz, Hz/s, Hz/s²]: {estimated.geometry_dc_polynomial}")
    print(f"Đã ghi {args.output} và {args.residual_output}")


if __name__ == "__main__":
    main()
