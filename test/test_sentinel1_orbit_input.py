"""Orbit EOF: mission/frame/unit/coverage, không thay file lỗi bằng PVT."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET
import numpy as np
from astropy.time import Time
from sentinel1_processing.utils.orbit_input import read_orbit


class OrbitInputTest(unittest.TestCase):
    def root(self):
        root=ET.fromstring('''<Earth_Explorer_File><Earth_Explorer_Header><Fixed_Header>
        <Mission>Sentinel-1A</Mission><File_Type>AUX_POEORB</File_Type><Validity_Period>
        <Validity_Start>UTC=2025-01-01T00:00:00</Validity_Start><Validity_Stop>UTC=2025-01-01T00:00:20</Validity_Stop>
        </Validity_Period></Fixed_Header><Variable_Header><Ref_Frame>EARTH_FIXED</Ref_Frame>
        <Time_Reference>UTC</Time_Reference></Variable_Header></Earth_Explorer_Header>
        <Data_Block><List_of_OSVs count="3"/></Data_Block></Earth_Explorer_File>''')
        for index in range(3):
            row=ET.SubElement(root.find('Data_Block/List_of_OSVs'),'OSV')
            ET.SubElement(row,'UTC').text=f'UTC=2025-01-01T00:00:{index*10:02d}'
            for field,value,unit in [('X',7e6,'m'),('Y',index*70000,'m'),('Z',0,'m'),
                                     ('VX',0,'m/s'),('VY',7000,'m/s'),('VZ',0,'m/s')]:
                ET.SubElement(row,field,unit=unit).text=str(value)
        return root

    def read(self,root):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'orbit.EOF';ET.ElementTree(root).write(path)
            return read_orbit(path,mission='Sentinel-1A',wavelength_m=.05,
                              ellipsoid_a_m=6378137.,ellipsoid_b_m=6356752.314245)

    def test_eof_orbit_interpolation_and_strict_coverage(self):
        orbit=self.read(self.root());epoch=float(Time('2025-01-01').gps)
        orbit.validate_time_coverage([epoch,epoch+20])
        np.testing.assert_allclose(orbit.position(epoch+5),[7e6,35000,0],atol=1e-7)
        np.testing.assert_allclose(orbit.velocity(epoch+5),[0,7000,0],atol=1e-7)
        for query in ([epoch-1],[epoch+21],[float('nan')]):
            with self.assertRaises(ValueError):orbit.validate_time_coverage(query)
        with self.assertRaises(ValueError):orbit.validate_time_coverage([epoch],max_extrapolation_s=1)

    def test_eof_rejects_wrong_mission_frame_units_duplicate_epochs_and_count(self):
        changes=[('Earth_Explorer_Header/Fixed_Header/Mission','Sentinel-1B'),
                 ('Earth_Explorer_Header/Variable_Header/Ref_Frame','J2000'),
                 ('Data_Block/List_of_OSVs/OSV/UTC','UTC=2025-01-01T00:00:10'),
                 ('Data_Block/List_of_OSVs/OSV/X','NaN')]
        for path,value in changes:
            root=self.root();root.find(path).text=value
            with self.subTest(path=path),self.assertRaises(ValueError):self.read(root)
        root=self.root();root.find('Data_Block/List_of_OSVs/OSV/X').set('unit','km')
        with self.assertRaises(ValueError):self.read(root)
        root=self.root();root.find('Data_Block/List_of_OSVs').set('count','4')
        with self.assertRaises(ValueError):self.read(root)
