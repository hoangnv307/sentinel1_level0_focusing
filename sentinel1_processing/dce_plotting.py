"""Plots for Doppler centroid fitting diagnostics."""

import matplotlib.pyplot as plt
import numpy as np


def plot_comparisons(comparisons):
    """Return one fit-diagnostic figure per DCE comparison record."""
    figures = []
    for result in comparisons:
        fit_times_ms = np.asarray(result["fit_range_times_s"]) * 1e3
        fit_points_hz = np.asarray(result["fit_points_hz"])
        used = np.asarray(result["fit_valid_mask"], dtype=bool)
        same_shape = fit_times_ms.shape == fit_points_hz.shape == used.shape
        if not same_shape:
            raise ValueError("DCE fit points, times, and mask must have equal shapes.")

        range_times_ms = np.asarray(result["range_times_s"]) * 1e3
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(
            range_times_ms,
            result["annotation_hz"],
            color="black",
            linewidth=2,
            label="Annotation record",
        )
        axis.plot(
            range_times_ms,
            result["estimated_hz"],
            "--",
            linewidth=2,
            label="Estimated fit",
        )
        if result.get("geometry_hz") is not None:
            axis.plot(
                range_times_ms,
                result["geometry_hz"],
                ":",
                linewidth=2,
                label="Geometry DC polynomial",
            )
        axis.scatter(
            fit_times_ms[used],
            fit_points_hz[used],
            s=45,
            color="tab:blue",
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=f"Fit points ({np.count_nonzero(used)})",
        )
        axis.scatter(
            fit_times_ms[~used],
            fit_points_hz[~used],
            s=70,
            color="tab:red",
            marker="x",
            linewidth=2,
            zorder=4,
            label=f"Rejected points ({np.count_nonzero(~used)})",
        )
        axis.set_title(
            f'DCE {result["record"]}: RMSE={result["rmse_hz"]:.3f} Hz, '
            f'fit RMS={result["fit_rms_hz"]:.3f} Hz'
        )
        axis.set_xlabel("Slant-range time [ms]")
        axis.set_ylabel("Doppler centroid [Hz]")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figures.append(figure)

    return figures


def plot_effective_velocity(slant_range_m, result):
    """Plot fitted effective velocity and its orbit-fit control points."""
    ranges_km = np.asarray(slant_range_m) / 1e3
    velocity = np.asarray(result.vr_mps)
    if ranges_km.shape != velocity.shape:
        raise ValueError("Slant range and effective velocity must have equal shapes.")

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(ranges_km, velocity, linewidth=2, label="Effective velocity fit")
    axis.scatter(
        np.asarray(result.control_ranges_m) / 1e3,
        result.control_vr_mps,
        color="tab:red",
        zorder=3,
        label="Orbit-fit control points",
    )
    axis.set_title("Effective velocity versus slant range")
    axis.set_xlabel("Slant range [km]")
    axis.set_ylabel("Effective velocity [m/s]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


__all__ = ["plot_comparisons", "plot_effective_velocity"]
