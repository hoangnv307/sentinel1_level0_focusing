"""Doppler centroid from Sentinel-1 orbit and attitude (DAD Section 5.1)."""

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
from astropy.time import Time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from ..common.effective_velocity import Estimator as OrbitEstimator


class Estimator:
    """Estimate absolute Doppler centroid from orbit and attitude geometry."""

    def __init__(self, ephemeris, wavelength_m):
        self.orbit = OrbitEstimator.from_ephemeris(ephemeris, wavelength_m)
        self.wavelength_m = float(wavelength_m)
        table = ephemeris.dropna(subset=[
            "Attitude Data Timestamp",
            "Q0 Attitude Quaternion",
            "Q1 Attitude Quaternion",
            "Q2 Attitude Quaternion",
            "Q3 Attitude Quaternion",
        ])
        times = table["Attitude Data Timestamp"].to_numpy(dtype=float)
        keep = np.unique(times, return_index=True)[1]
        self.attitude_times_s = times[keep]
        q = table[[
            "Q0 Attitude Quaternion",
            "Q1 Attitude Quaternion",
            "Q2 Attitude Quaternion",
            "Q3 Attitude Quaternion",
        ]].to_numpy(dtype=float)[keep]
        if self.attitude_times_s.size < 2:
            raise ValueError("At least two Level-0 attitude records are required.")
        # PDU §3.2.3: Q0 is real; scipy expects [Q1, Q2, Q3, Q0].
        self._attitude = Slerp(
            self.attitude_times_s,
            Rotation.from_quat(np.column_stack([q[:, 1:], q[:, 0]])),
        )

    @classmethod
    def from_level0_product(cls, level0_product, wavelength_m):
        return cls(level0_product.ephemeris, wavelength_m)

    def _azimuth_plane_normal_ecef(self, time_s):
        """Return Eq. 5-9 u0, transforming L0 ECI/J2000 attitude to ECEF."""
        time = float(time_s)
        if not self.attitude_times_s[0] <= time <= self.attitude_times_s[-1]:
            raise ValueError("Time is outside the Level-0 attitude interval.")
        body_to_eci = self._attitude([time]).as_matrix()[0]
        epoch = Time(time, format="gps")
        eci_to_ecef = np.column_stack([
            GCRS(CartesianRepresentation(axis * u.m), obstime=epoch)
            .transform_to(ITRS(obstime=epoch))
            .cartesian.xyz.to_value(u.m)
            for axis in np.eye(3)
        ])
        return (eci_to_ecef @ body_to_eci)[:, 0]

    def _evaluate_one(self, time_s, slant_range_m, plane_normal):
        satellite = self.orbit.position(time_s)
        velocity = self.orbit.velocity(time_s)
        radius = float(slant_range_m)
        initial = self.orbit.solve_ground_target(time_s, radius, 0.0)

        def residual(target):
            view = target - satellite
            return np.array([
                (np.linalg.norm(view) - radius) / radius,
                (target[0] ** 2 + target[1] ** 2) / self.orbit.a_m**2
                + target[2] ** 2 / self.orbit.b_m**2
                - 1.0,
                np.dot(plane_normal, view) / radius,
            ])

        solution = least_squares(
            residual,
            initial,
            method="lm",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=500,
        )
        if not solution.success or np.linalg.norm(residual(solution.x)) > 1e-7:
            raise RuntimeError("Orbit/attitude Doppler ground-target solve failed.")
        view = solution.x - satellite
        # Eq. 5-1 uses target-to-satellite r0; ``view`` is the reverse vector.
        return 2.0 * np.dot(velocity, view) / (self.wavelength_m * radius)

    def estimate(self, time_s, slant_ranges_m, *, n_control_points=9):
        """Return geometry DC [Hz], quadratically interpolated over range."""
        ranges = np.asarray(slant_ranges_m, dtype=np.float64)
        if ranges.ndim != 1 or ranges.size < 2 or np.any(ranges <= 0.0):
            raise ValueError("slant_ranges_m must be a positive 1-D grid.")
        count = int(np.clip(n_control_points, 3, ranges.size))
        indices = np.unique(np.round(
            np.linspace(0, ranges.size - 1, count)
        ).astype(int))
        normal = self._azimuth_plane_normal_ecef(time_s)
        values = np.array([
            self._evaluate_one(time_s, ranges[index], normal)
            for index in indices
        ])
        centre = float(np.mean(ranges[indices]))
        scale = float(np.ptp(ranges[indices])) or 1.0
        coefficients = np.polyfit(
            (ranges[indices] - centre) / scale,
            values,
            min(2, indices.size - 1),
        )
        return np.polyval(coefficients, (ranges - centre) / scale)


__all__ = ["Estimator"]
