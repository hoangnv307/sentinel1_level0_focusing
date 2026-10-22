"""Terrain EAP tách riêng ellipsoid geometry-DCE; DEM datum/nodata/coverage."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import importlib.util
import unittest
import numpy as np
from astropy.time import Time
from sentinel1_processing.doppler_centroid.geometry import Estimator, geodetic_position, nominal_boresight_angle_deg
from sentinel1_processing.common.effective_velocity import Estimator as Orbit
from sentinel1_processing.utils.terrain import RasterTerrain


class TerrainGeometryTest(unittest.TestCase):
    def test_geodetic_height_and_roll_units(self):
        a,b=6378137.,6356752.314245
        for point,expected in (([a+100,0,0],[0,0,100]),([0,a+200,0],[0,90,200]),([0,0,b+300],[90,0,300])):
            np.testing.assert_allclose(geodetic_position(point,ellipsoid_a_m=a,ellipsoid_b_m=b),expected,atol=1e-6)
        self.assertAlmostEqual(nominal_boresight_angle_deg(712700,reference_angle_deg=29.45,
            reference_height_m=711700,roll_sensitivity_deg_per_m=5.66e-5),29.3934)

    def test_terrain_target_obeys_range_zero_doppler_and_height(self):
        a,b=6378137.,6356752.314245
        # Use a real Hermite orbit provider, no attitude needed for EAP look-angle calculation.
        orbit=Orbit([0,10],[[7e6,0,0],[7e6,70000,0]],[[0,7000,0],[0,7000,0]],.05,ellipsoid_a_m=a,ellipsoid_b_m=a)
        estimator=object.__new__(Estimator);estimator.orbit=orbit
        terrain=Mock(return_value=500.)
        angles=estimator.terrain_look_angles_deg(5,[900000,950000],terrain_provider=terrain,
            range_tolerance_m=.01,height_tolerance_m=.01,doppler_tolerance_mps=.001,max_nfev=200,look_side='right')
        self.assertTrue(np.isfinite(angles).all());self.assertGreater(angles[1],angles[0])
        radius=np.linalg.norm(orbit.position(5))
        expected=np.degrees(np.arccos((radius**2+np.array([900000,950000])**2-(a+500)**2)/
                                      (2*radius*np.array([900000,950000]))))
        np.testing.assert_allclose(angles,expected,atol=1e-6)
        self.assertTrue(terrain.called)
        with self.assertRaises(ValueError):
            estimator.terrain_look_angles_deg(11,[900000],terrain_provider=terrain,
                range_tolerance_m=.01,height_tolerance_m=.01,doppler_tolerance_mps=.001,max_nfev=200,look_side='right')


@unittest.skipUnless(importlib.util.find_spec('rasterio'),'Cần optional extra [dem].')
class RasterTerrainTest(unittest.TestCase):
    def test_explicit_datum_coverage_and_nodata(self):
        import rasterio
        from rasterio.transform import from_origin
        with TemporaryDirectory() as folder:
            path=Path(folder)/'dem.tif'
            with rasterio.open(path,'w',driver='GTiff',width=2,height=2,count=1,dtype='float32',
                    crs='EPSG:4326',transform=from_origin(10,20,1,1),nodata=-9999) as dataset:
                dataset.write(np.array([[10,20],[30,-9999]],dtype='float32'),1)
            with self.assertRaises(ValueError):RasterTerrain(path,height_datum='unknown',height_unit='m')
            with RasterTerrain(path,height_datum='ellipsoid',height_unit='m') as terrain:
                self.assertEqual(terrain(19.5,10.5),10)
                with self.assertRaises(ValueError):terrain(18.5,11.5)
                with self.assertRaises(ValueError):terrain(0,0)
