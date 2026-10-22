"""Doppler centroid from Sentinel-1 orbit and attitude (DAD Section 5.1)."""

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, PrecessedGeocentric, ITRS
from astropy.time import Time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from ..common.effective_velocity import Estimator as OrbitEstimator


def sar_attitude_rotation(quaternions_wxyz, *, norm_tolerance):
    """SAT→GM2000 rotation; transpose matches EOCFI-NOTE-052 §3.5.2 matrix."""
    q = np.asarray(quaternions_wxyz, dtype=float)
    if (q.ndim not in (1,2) or q.shape[-1]!=4 or not np.isfinite(q).all()
            or not np.isfinite(norm_tolerance) or norm_tolerance<0):
        raise ValueError("SAR quaternion/tolerance không hợp lệ.")
    if np.any(np.abs(np.linalg.norm(q,axis=-1)-1)>norm_tolerance):
        raise ValueError("SAR quaternion không có norm gần một.")
    return Rotation.from_quat(np.roll(q,-1,axis=-1))


def gm2000_to_ecef(time_s):
    """Mean equator/equinox J2000 → ITRS, including GCRS frame bias via Astropy."""
    if not np.isfinite(time_s):raise ValueError("Geometry time phải hữu hạn.")
    epoch = Time(time_s,format="gps")
    return np.column_stack([
        PrecessedGeocentric(CartesianRepresentation(axis*u.m),equinox=Time("J2000"),obstime=epoch)
        .transform_to(ITRS(obstime=epoch)).cartesian.xyz.to_value(u.m)
        for axis in np.eye(3)])


def geodetic_position(position_ecef_m, *, ellipsoid_a_m, ellipsoid_b_m):
    """Latitude/longitude (deg), ellipsoid height (m), for the supplied ellipsoid."""
    point=np.asarray(position_ecef_m,dtype=float)
    if point.shape!=(3,) or not np.isfinite(point).all():
        raise ValueError("Ground point ECEF không hợp lệ.")
    a,b=float(ellipsoid_a_m),float(ellipsoid_b_m)
    if not np.isfinite([a,b]).all() or not 0<b<=a:
        raise ValueError("Ellipsoid axes không hợp lệ.")
    x,y,z=point;radius=np.hypot(x,y);e2=1-b*b/(a*a)
    if np.linalg.norm(point)==0:raise ValueError("Geodetic position không xác định tại Earth center.")
    latitude=np.arctan2(z,radius*(1-e2))
    for _ in range(12):
        prime=a/np.sqrt(1-e2*np.sin(latitude)**2)
        latitude=np.arctan2(z+e2*prime*np.sin(latitude),radius)
    height=radius*np.cos(latitude)+z*np.sin(latitude)-a*np.sqrt(1-e2*np.sin(latitude)**2)
    return float(np.degrees(latitude)),float(np.degrees(np.arctan2(y,x))),float(height)


def nominal_boresight_angle_deg(altitude_m, *, reference_angle_deg, reference_height_m,
                                roll_sensitivity_deg_per_m):
    """DAD Eq. 9-19; caller explicitly authorizes nominal fallback."""
    if not np.isfinite([altitude_m,reference_angle_deg,reference_height_m,roll_sensitivity_deg_per_m]).all():
        raise ValueError("Nominal roll law parameters không hữu hạn.")
    return reference_angle_deg-roll_sensitivity_deg_per_m*(altitude_m-reference_height_m)


class Estimator:
    """Estimate absolute Doppler centroid from orbit and attitude geometry."""

    def __init__(self, ephemeris, wavelength_m, *, orbit_provider=None,
                 ellipsoid_a_m=6378137., ellipsoid_b_m=6356752.314245, quaternion_norm_tolerance=1e-3):
        self.orbit = (OrbitEstimator.from_ephemeris(ephemeris, wavelength_m,
            ellipsoid_a_m=ellipsoid_a_m,ellipsoid_b_m=ellipsoid_b_m) if orbit_provider is None else orbit_provider)
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
        for instant in self.attitude_times_s:
            duplicates = table.loc[table["Attitude Data Timestamp"]==instant,
                ["Q0 Attitude Quaternion","Q1 Attitude Quaternion","Q2 Attitude Quaternion","Q3 Attitude Quaternion"]].to_numpy(float)
            if not np.all(duplicates==duplicates[0]):
                raise ValueError("Conflicting attitude records at the same timestamp.")
        if not np.isfinite(self.attitude_times_s).all():
            raise ValueError("Attitude timestamps phải hữu hạn.")
        self._attitude = Slerp(self.attitude_times_s,
            sar_attitude_rotation(q,norm_tolerance=quaternion_norm_tolerance))

    @classmethod
    def from_level0_product(cls, level0_product, wavelength_m, **kwargs):
        return cls(level0_product.ephemeris, wavelength_m, **kwargs)

    def _azimuth_plane_normal_ecef(self, time_s):
        """Return Eq. 5-9 u0, transforming L0 ECI/J2000 attitude to ECEF."""
        time = float(time_s)
        if not self.attitude_times_s[0] <= time <= self.attitude_times_s[-1]:
            raise ValueError("Time is outside the Level-0 attitude interval.")
        body_to_gm2000 = self._attitude([time]).as_matrix()[0]
        return (gm2000_to_ecef(time) @ body_to_gm2000)[:, 0]

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

    def boresight_angle_deg(self, time_s, *, antenna_axis_body):
        self.orbit.validate_time_coverage([time_s])
        if not self.attitude_times_s[0]<=time_s<=self.attitude_times_s[-1]:
            raise ValueError("Boresight query ngoài attitude coverage.")
        axis=np.asarray(antenna_axis_body,dtype=float)
        if axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1.,atol=1e-12,rtol=0):
            raise ValueError("Antenna axis phải là unit vector trong SAR body frame.")
        direction=gm2000_to_ecef(time_s)@self._attitude([time_s]).as_matrix()[0]@axis
        position=self.orbit.position(time_s)
        return float(np.degrees(np.arccos(np.clip(-position@direction/np.linalg.norm(position),-1,1))))

    def terrain_look_angles_deg(self, time_s, slant_ranges_m, *, terrain_provider,
                               range_tolerance_m, height_tolerance_m, doppler_tolerance_mps,
                               max_nfev, look_side):
        """DAD §9.5.1: terrain targets in zero-Doppler elevation plane (ECEF)."""
        self.orbit.validate_time_coverage([time_s])
        ranges=np.asarray(slant_ranges_m,dtype=float)
        if (ranges.ndim!=1 or not ranges.size or not np.isfinite(ranges).all() or np.any(ranges<=0)
                or not np.isfinite([range_tolerance_m,height_tolerance_m,doppler_tolerance_mps]).all()
                or min(range_tolerance_m,height_tolerance_m,doppler_tolerance_mps)<=0
                or isinstance(max_nfev,bool) or not isinstance(max_nfev,int) or max_nfev<1
                or look_side not in ("right","left")):
            raise ValueError("Terrain geometry ranges/tolerances không hợp lệ.")
        satellite=self.orbit.position(time_s);velocity=self.orbit.velocity(time_s)
        speed=np.linalg.norm(velocity)
        if speed<=0:raise ValueError("Orbit velocity bằng zero.")
        angles=[]
        for radius in ranges:
            initial=self.orbit.solve_ground_target(time_s,radius,0.,look_side=look_side)
            def errors(point):
                lat,lon,height=geodetic_position(point,ellipsoid_a_m=self.orbit.a_m,ellipsoid_b_m=self.orbit.b_m)
                terrain=terrain_provider(lat,lon)
                if not np.isfinite(terrain):raise ValueError("Terrain height không hữu hạn.")
                view=point-satellite
                return np.array([np.linalg.norm(view)-radius,height-terrain,velocity@view/radius])
            def residual(point):
                return errors(point)/np.array([range_tolerance_m,height_tolerance_m,doppler_tolerance_mps])
            solution=least_squares(residual,initial,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=max_nfev)
            error=errors(solution.x)
            if not solution.success or np.any(abs(error)>[range_tolerance_m,height_tolerance_m,doppler_tolerance_mps]):
                raise ValueError("Terrain target solver không đạt range/height/Doppler tolerance.")
            view=solution.x-satellite
            angles.append(np.degrees(np.arccos(np.clip(-satellite@view/(np.linalg.norm(satellite)*np.linalg.norm(view)),-1,1))))
        return np.asarray(angles)

    def estimate(self, time_s, slant_ranges_m, *, n_control_points=9):
        """Return geometry DC [Hz], quadratically interpolated over range."""
        self.orbit.validate_time_coverage(np.array([time_s], dtype=float))
        ranges = np.asarray(slant_ranges_m, dtype=np.float64)
        if ranges.ndim != 1 or ranges.size < 2 or not np.isfinite(ranges).all() or np.any(ranges <= 0.0):
            raise ValueError("slant_ranges_m must be a positive 1-D grid.")
        if isinstance(n_control_points,bool) or not isinstance(n_control_points,(int,np.integer)) or n_control_points<3:
            raise ValueError("n_control_points must be an integer >=3.")
        count = min(n_control_points, ranges.size)
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


__all__ = ["Estimator", "sar_attitude_rotation", "gm2000_to_ecef", "geodetic_position", "nominal_boresight_angle_deg"]
