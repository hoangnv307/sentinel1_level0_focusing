"""Ví dụ số ESA EOCFI-NOTE-052 issue 2.1 §3.5.1–4 và GM2000 bias."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
import astropy.units as u
import erfa
from sentinel1_processing.doppler_centroid.geometry import sar_attitude_rotation, gm2000_to_ecef


class AttitudeTest(unittest.TestCase):
    def test_esa_numerical_example_matrix_axes_and_quaternion(self):
        q=[-.3229468762874603272,-.9336623549461364746,.02849436365067958832,-.1522108763456344604]
        rotation=sar_attitude_rotation(q,norm_tolerance=1e-6)
        gm_to_sat=rotation.as_matrix().T
        np.testing.assert_allclose(gm_to_sat,[[.952039848,.045103818,.302631414],
            [-.151520260,-.789786806,.594372284],[.265822757,-.611720890,-.745074369]],atol=1e-9)
        cfi=gm_to_sat[[1,0,2]]*-1
        np.testing.assert_allclose(cfi,[[.151520260,.789786806,-.594372284],
            [-.952039848,-.045103818,-.302631414],[-.265822757,.611720890,.745074369]],atol=1e-9)
        cfi_quaternion=Rotation.from_matrix(cfi.T).as_quat()
        reference=np.array([-.335987242547,.120728573839,.640050374327,.680347486678])
        if cfi_quaternion@reference<0:cfi_quaternion=-cfi_quaternion
        np.testing.assert_allclose(cfi_quaternion,reference,atol=1e-10)
        np.testing.assert_allclose(rotation.apply([1,0,0]),gm_to_sat[0],atol=1e-12)

    def test_mean_j2000_frame_is_not_assumed_equal_to_gcrs(self):
        epoch=Time('2025-09-03T18:36:28',scale='utc')
        rotation=gm2000_to_ecef(float(epoch.gps))
        gcrs_to_itrs=np.column_stack([GCRS(CartesianRepresentation(axis*u.m),obstime=epoch)
            .transform_to(ITRS(obstime=epoch)).cartesian.xyz.to_value(u.m) for axis in np.eye(3)])
        bias=erfa.pmat06(2451545.,0.)
        np.testing.assert_allclose(rotation,gcrs_to_itrs@bias.T,atol=2e-14)
        np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=2e-14)
        self.assertGreater(float(np.max(abs(rotation-gcrs_to_itrs))),1e-9)

    def test_invalid_quaternions_rejected_instead_of_silent_normalization(self):
        for q in ([0,0,0,0],[2,0,0,0],[np.nan,0,0,0],[1,0,0]):
            with self.assertRaises(ValueError):sar_attitude_rotation(q,norm_tolerance=1e-3)
