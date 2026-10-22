import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from sentinel1_processing.range_processing import dependent_gain as gain
from sentinel1_processing.utils.eap_input import read_eap_lut


class DependentGainTest(unittest.TestCase):
    def test_complex_eap_interpolation_preserves_phase_and_absolute_scale(self):
        lut=np.array([4j,16j])
        actual=gain.eap_correction([0,.5,1],[0,1],lut,minimum_magnitude=0.)
        np.testing.assert_allclose(actual,-1j/np.sqrt([4,10,16]))
        for angles,values in (([-1],lut),([.5],np.array([1,-1])),([np.nan],lut)):
            with self.assertRaises(ValueError):gain.eap_correction(angles,[0,1],values,minimum_magnitude=0.)

    def test_rsl_uses_amplitude_and_validates_reference(self):
        np.testing.assert_allclose(gain.rsl_correction([1,4],reference_range_m=1),[1,8])
        for reference in (0,-1,np.nan):
            with self.assertRaises(ValueError):gain.rsl_correction([1],reference_range_m=reference)

    def test_bank_endpoints_interpolation_and_coverage(self):
        times=gain.bank_times(10,12.5,update_interval_s=1)
        np.testing.assert_allclose(times,[10,11,12,12.5])
        bank=gain.build_eap_bank(times,[1,2],look_angle_provider=lambda t,r:np.zeros(2),
            boresight_angle_provider=lambda t:0.,angular_grid_deg=[-1,1],complex_lut=[4j,4j],minimum_magnitude=0.)
        np.testing.assert_allclose(gain.interpolate_bank([10,11.5,12.5],times,bank),np.full((3,2),-.5j))
        with self.assertRaises(ValueError):gain.interpolate_bank([13],times,bank)

    def test_aux_lut_grid_is_centered_independently_of_nominal_near_far(self):
        xml='''<auxiliaryCalibration schemaVersion="2.10"><calibrationParamsList><calibrationParams>
        <swath>S4</swath><polarisation>VV</polarisation><elevationAntennaPattern>
        <beamNominalNearRange>30</beamNominalNearRange><beamNominalFarRange>34</beamNominalFarRange>
        <elevationAngleIncrement>0.05</elevationAngleIncrement><values count="3">1 2 3 4 5 6</values>
        </elevationAntennaPattern></calibrationParams></calibrationParamsList></auxiliaryCalibration>'''
        with TemporaryDirectory() as folder:
            path=Path(folder)/'cal.xml';path.write_text(xml)
            angles,lut=read_eap_lut(path,swath='S4',polarisation='VV')
        np.testing.assert_allclose(angles,[-.05,0,.05]);np.testing.assert_allclose(lut,[1+2j,3+4j,5+6j])
