"""Effective velocity from Sentinel-1 L1 DAD Section 9.10."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares


WGS84_A_M = 6378137.0
WGS84_B_M = 6356752.314245


@dataclass
class ControlPoint:
    """Diagnostic result for one range control point."""
    range_m: float
    fdc_hz: float
    vr_mps: float
    r0_m: float
    closest_time_offset_s: float
    fit_rms_m: float
    target_ecef_m: np.ndarray


@dataclass
class BlockResult:
    """Result of one azimuth processing block."""
    vr_mps: np.ndarray
    control_ranges_m: np.ndarray
    control_vr_mps: np.ndarray
    polynomial_coefficients: np.ndarray
    control_points: list[ControlPoint]


class Estimator:
    """
    Estimate V_r(R, eta) block-by-block from actual orbit geometry.

    Orbit interpolation uses a cubic Hermite spline so that both measured
    position and velocity state vectors are honoured at each ephemeris epoch.
    """

    def __init__(
        self,
        orbit_times_s,
        positions_ecef_m,
        velocities_ecef_mps,
        wavelength_m,
        *,
        ellipsoid_a_m=WGS84_A_M,
        ellipsoid_b_m=WGS84_B_M,
    ):
        self.orbit_times_s = np.asarray(orbit_times_s, dtype=np.float64)
        self.positions_ecef_m = np.asarray(positions_ecef_m, dtype=np.float64)
        self.velocities_ecef_mps = np.asarray(velocities_ecef_mps, dtype=np.float64)
        self.wavelength_m = float(wavelength_m)
        self.a_m = float(ellipsoid_a_m)
        self.b_m = float(ellipsoid_b_m)

        if self.orbit_times_s.ndim != 1:
            raise ValueError("orbit_times_s must be a 1-D array.")
        if self.positions_ecef_m.shape != (self.orbit_times_s.size, 3):
            raise ValueError("positions_ecef_m must have shape (N, 3).")
        if self.velocities_ecef_mps.shape != (self.orbit_times_s.size, 3):
            raise ValueError("velocities_ecef_mps must have shape (N, 3).")
        if np.any(np.diff(self.orbit_times_s) <= 0):
            raise ValueError("orbit_times_s must be strictly increasing.")

        # r'(t) = v(t) is enforced at every supplied state-vector epoch.
        self._position_spline = CubicHermiteSpline(
            self.orbit_times_s,
            self.positions_ecef_m,
            self.velocities_ecef_mps,
            axis=0,
            extrapolate=True,
        )

    @classmethod
    def from_ephemeris(
        cls,
        ephemeris,
        wavelength_m,
        *,
        ellipsoid_a_m=WGS84_A_M,
        ellipsoid_b_m=WGS84_B_M,
    ):
        """Build an estimator from decoded Level-0 ephemeris records.

        Expected columns:
          POD Solution Data Timestamp
          X/Y/Z-axis position ECEF
          X/Y/Z-axis velocity ECEF
        """
        t = ephemeris["POD Solution Data Timestamp"].to_numpy(dtype=float)

        p = np.column_stack([
            ephemeris["X-axis position ECEF"].to_numpy(dtype=float),
            ephemeris["Y-axis position ECEF"].to_numpy(dtype=float),
            ephemeris["Z-axis position ECEF"].to_numpy(dtype=float),
        ])

        v = np.column_stack([
            ephemeris["X-axis velocity ECEF"].to_numpy(dtype=float),
            ephemeris["Y-axis velocity ECEF"].to_numpy(dtype=float),
            ephemeris["Z-axis velocity ECEF"].to_numpy(dtype=float),
        ])

        keep = np.unique(t, return_index=True)[1]
        t, p, v = t[keep], p[keep], v[keep]

        return cls(
            t, p, v, wavelength_m,
            ellipsoid_a_m=ellipsoid_a_m,
            ellipsoid_b_m=ellipsoid_b_m,
        )

    @classmethod
    def from_level0_product(cls, level0_product, wavelength_m):
        """Build the DAD Section 9.10 estimator from a decoded L0 product."""
        return cls.from_ephemeris(level0_product.ephemeris, wavelength_m)

    def validate_time_coverage(self, azimuth_times_s):
        """Require all azimuth times to lie inside the ephemeris interval."""
        times = np.asarray(azimuth_times_s, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("azimuth_times_s must be a non-empty 1-D array.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("azimuth_times_s must be strictly increasing.")
        if times[0] < self.orbit_times_s[0] or times[-1] > self.orbit_times_s[-1]:
            raise ValueError(
                "Azimuth times are outside the available ephemeris interval."
            )

    def position(self, time_s):
        """Interpolated ECEF spacecraft position [m]."""
        return np.asarray(self._position_spline(time_s), dtype=np.float64)

    def velocity(self, time_s):
        """Derivative of the interpolated orbit [m/s]."""
        return np.asarray(self._position_spline(time_s, 1), dtype=np.float64)

    # ------------------------------------------------------------------
    # Ground-target geometry
    # ------------------------------------------------------------------

    def _local_basis(self, position_m, velocity_mps):
        """
        Local radial / along-track / cross-track basis.

        cross-track sign is only used to form a robust initial guess.
        The final target is solved from ellipsoid + range + Doppler equations.
        """
        r_hat = position_m / np.linalg.norm(position_m)

        v_tangent = velocity_mps - np.dot(velocity_mps, r_hat) * r_hat
        along_hat = v_tangent / np.linalg.norm(v_tangent)

        # Right-side initial direction for a nominal right-looking SAR.
        cross_hat = np.cross(along_hat, r_hat)
        cross_hat /= np.linalg.norm(cross_hat)

        return r_hat, along_hat, cross_hat

    def _local_ellipsoid_radius(self, direction):
        u = np.asarray(direction, dtype=np.float64)
        u = u / np.linalg.norm(u)
        return 1.0 / np.sqrt(
            (u[0] ** 2 + u[1] ** 2) / self.a_m ** 2
            + u[2] ** 2 / self.b_m ** 2
        )

    def _initial_target_guess(
        self,
        position_m,
        velocity_mps,
        slant_range_m,
        look_side,
    ):
        """
        Spherical local-geometry guess used only to seed the nonlinear solver.
        """
        rs = np.linalg.norm(position_m)
        r_hat, _, cross_hat = self._local_basis(position_m, velocity_mps)

        re = self._local_ellipsoid_radius(r_hat)
        R = float(slant_range_m)

        # Triangle satellite-centre-target:
        # R^2 = Rs^2 + Re^2 - 2 Rs Re cos(gamma)
        cos_gamma = (rs**2 + re**2 - R**2) / (2.0 * rs * re)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        gamma = np.arccos(cos_gamma)

        side = -1.0 if str(look_side).lower() == "left" else 1.0

        ground_dir = (
            np.cos(gamma) * r_hat
            + side * np.sin(gamma) * cross_hat
        )
        ground_dir /= np.linalg.norm(ground_dir)

        re_ground = self._local_ellipsoid_radius(ground_dir)
        return re_ground * ground_dir

    def solve_ground_target(
        self,
        center_time_s,
        slant_range_m,
        fdc_hz=0.0,
        *,
        look_side="right",
    ):
        """
        Solve a ground point P in ECEF from three constraints:

          1) |P-S| = R
          2) P lies on WGS-84 ellipsoid
          3) v . (P-S) = -lambda * f_dc * R / 2

        Constraint (3) follows the SAR Doppler relation used in the DAD:
            f_d = -2/lambda * v . r_hat

        The two possible left/right solutions are separated by the initial guess.
        """
        S = self.position(center_time_s)
        V = self.velocity(center_time_s)
        R = float(slant_range_m)
        fdc = float(fdc_hz)

        x0 = self._initial_target_guess(S, V, R, look_side)

        # Scale residuals to comparable dimensionless magnitudes.
        def residual(P):
            d = P - S

            range_res = (np.linalg.norm(d) - R) / R

            ellipsoid_res = (
                (P[0] ** 2 + P[1] ** 2) / self.a_m ** 2
                + P[2] ** 2 / self.b_m ** 2
                - 1.0
            )

            # v.d + lambda*f_dc*R/2 = 0
            doppler_scale = max(np.linalg.norm(V) * R, 1.0)
            doppler_res = (
                np.dot(V, d)
                + 0.5 * self.wavelength_m * fdc * R
            ) / doppler_scale

            return np.array(
                [range_res, ellipsoid_res, doppler_res],
                dtype=np.float64,
            )

        sol = least_squares(
            residual,
            x0,
            method="trf",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=200,
        )

        if not sol.success:
            raise RuntimeError(
                f"Ground-target solve failed: {sol.message}"
            )

        return sol.x

    # ------------------------------------------------------------------
    # Hyperbolic fit
    # ------------------------------------------------------------------

    def _provisional_aperture_time(
        self,
        center_time_s,
        slant_range_m,
        fdc_hz,
        azimuth_bandwidth_hz,
    ):
        """
        Estimate T_mf for selecting the range-history fitting interval.

        This is only used to choose the fitting time support.  It does not
        become the final V_r estimate.
        """
        v0 = np.linalg.norm(self.velocity(center_time_s))

        D0_sq = 1.0 - (
            self.wavelength_m * float(fdc_hz) / (2.0 * v0)
        ) ** 2
        D0 = np.sqrt(max(D0_sq, 1e-12))

        ka_abs = (
            2.0 * v0**2 * D0**3
            / (self.wavelength_m * float(slant_range_m))
        )

        return float(azimuth_bandwidth_hz) / ka_abs

    def fit_effective_velocity(
        self,
        center_time_s,
        target_ecef_m,
        *,
        aperture_time_s,
        num_time_samples=65,
    ):
        """
        Fit actual range history to:

            R(t) = sqrt(R0^2 + Vr^2 * (t - t0)^2)

        Returns:
            Vr, R0, t0_offset, fit_rms
        """
        target = np.asarray(target_ecef_m, dtype=np.float64)

        half = 0.5 * float(aperture_time_s)
        dt = np.linspace(
            -half, half,
            int(num_time_samples),
            dtype=np.float64,
        )

        times = float(center_time_s) + dt
        sat_pos = self.position(times)

        actual_range = np.linalg.norm(
            sat_pos - target[None, :],
            axis=1,
        )

        # Robust initial estimates.
        imin = int(np.argmin(actual_range))
        t0_guess = dt[imin]
        r0_guess = actual_range[imin]
        v_guess = np.linalg.norm(self.velocity(center_time_s))

        def residual(par):
            r0, vr, t0 = par
            model = np.sqrt(
                r0**2 + vr**2 * (dt - t0) ** 2
            )
            return model - actual_range

        sol = least_squares(
            residual,
            x0=np.array([r0_guess, v_guess, t0_guess]),
            bounds=(
                np.array([0.5 * r0_guess, 0.5 * v_guess, -2.0 * half]),
                np.array([1.5 * r0_guess, 1.5 * v_guess, +2.0 * half]),
            ),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=300,
        )

        if not sol.success:
            raise RuntimeError(
                f"Hyperbolic fit failed: {sol.message}"
            )

        r0, vr, t0 = sol.x
        fit_rms = float(np.sqrt(np.mean(residual(sol.x) ** 2)))

        return float(vr), float(r0), float(t0), fit_rms

    # ------------------------------------------------------------------
    # Public block API
    # ------------------------------------------------------------------

    def evaluate_block(
        self,
        block_center_time_s,
        slant_range_m,
        *,
        fdc_hz=None,
        azimuth_bandwidth_hz,
        n_control_points=9,
        range_polynomial_degree=2,
        num_time_samples=65,
        look_side="right",
        return_diagnostics=False,
    ):
        """Compute effective velocity for one azimuth processing block.

        Parameters
        ----------
        block_center_time_s : float
            Azimuth time of the processing-block centre, in the same time base
            as the ephemeris timestamps.

        slant_range_m : ndarray, shape (N_range,)
            Slant-range vector for the processing block.

        fdc_hz : None, scalar, or ndarray, shape (N_range,)
            Doppler centroid versus range. If None, zero Doppler is used.

        azimuth_bandwidth_hz : float
            Processed azimuth bandwidth, e.g. 1398 Hz for SM_SL1__1/S6.

        n_control_points : int
            Number of ranges where the expensive orbit/history fit is carried
            out.  V_r is then fitted versus range.

        range_polynomial_degree : int
            Low-order polynomial degree for V_r(R), normally 2 is sufficient.

        return_diagnostics : bool
            False -> return only V_r vector.
            True  -> return BlockResult.

        Returns
        -------
        ndarray or BlockResult
            V_r(R) [m/s], same length as slant_range_m.
        """
        ranges = np.asarray(slant_range_m, dtype=np.float64)

        if ranges.ndim != 1 or ranges.size < 2:
            raise ValueError("slant_range_m must be a 1-D vector.")

        if fdc_hz is None:
            fdc = np.zeros_like(ranges)
        elif np.ndim(fdc_hz) == 0:
            fdc = np.full_like(ranges, float(fdc_hz))
        else:
            fdc = np.asarray(fdc_hz, dtype=np.float64)
            if fdc.shape != ranges.shape:
                raise ValueError(
                    "fdc_hz must be scalar or have same shape as slant_range_m."
                )

        ncp = int(np.clip(n_control_points, 3, ranges.size))
        control_idx = np.unique(
            np.round(
                np.linspace(0, ranges.size - 1, ncp)
            ).astype(int)
        )

        control_ranges = ranges[control_idx]
        control_fdc = fdc[control_idx]
        control_vr = np.empty(control_idx.size, dtype=np.float64)
        diagnostics = []

        for k, idx in enumerate(control_idx):
            R = float(ranges[idx])
            fd = float(fdc[idx])

            target = self.solve_ground_target(
                block_center_time_s,
                R,
                fd,
                look_side=look_side,
            )

            aperture_time = self._provisional_aperture_time(
                block_center_time_s,
                R,
                fd,
                azimuth_bandwidth_hz,
            )

            vr, r0, t0, rms = self.fit_effective_velocity(
                block_center_time_s,
                target,
                aperture_time_s=aperture_time,
                num_time_samples=num_time_samples,
            )

            control_vr[k] = vr

            diagnostics.append(
                ControlPoint(
                    range_m=R,
                    fdc_hz=fd,
                    vr_mps=vr,
                    r0_m=r0,
                    closest_time_offset_s=t0,
                    fit_rms_m=rms,
                    target_ecef_m=target.copy(),
                )
            )

        degree = int(
            np.clip(
                range_polynomial_degree,
                1,
                control_ranges.size - 1,
            )
        )

        # Centre/scale range before fitting to improve conditioning.
        r_ref = float(np.mean(control_ranges))
        r_scale = float(np.ptp(control_ranges))
        if r_scale == 0.0:
            r_scale = 1.0

        x_control = (control_ranges - r_ref) / r_scale
        x_all = (ranges - r_ref) / r_scale

        coeff = np.polyfit(
            x_control,
            control_vr,
            deg=degree,
        )

        vr_all = np.polyval(coeff, x_all)

        if not return_diagnostics:
            return vr_all

        # Coefficients are for normalized range:
        # x = (R-r_ref)/r_scale.
        result = BlockResult(
            vr_mps=vr_all,
            control_ranges_m=control_ranges,
            control_vr_mps=control_vr,
            polynomial_coefficients=np.concatenate(
                [coeff, np.array([r_ref, r_scale])]
            ),
            control_points=diagnostics,
        )
        return result

__all__ = ["Estimator", "ControlPoint", "BlockResult"]
