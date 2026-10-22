"""Orbit EOF EARTH_FIXED/UTC theo Copernicus POD FFS §4.4.8."""
from xml.etree import ElementTree as ET
from astropy.time import Time
import numpy as np
from ..common.effective_velocity import Estimator as OrbitEstimator


class EOFOrbit(OrbitEstimator):
    def position(self,time_s):
        self.validate_time_coverage(np.unique(np.atleast_1d(time_s)))
        return super().position(time_s)

    def velocity(self,time_s):
        self.validate_time_coverage(np.unique(np.atleast_1d(time_s)))
        return super().velocity(time_s)

    def validate_time_coverage(self, azimuth_times_s, *, max_extrapolation_s=0.):
        if max_extrapolation_s!=0:
            raise ValueError('EOF orbit không cho phép extrapolation.')
        super().validate_time_coverage(azimuth_times_s,max_extrapolation_s=0.)
        if min(azimuth_times_s)<self.validity_interval_s[0] or max(azimuth_times_s)>self.validity_interval_s[1]:
            raise ValueError('Orbit query ngoài EOF validity period.')


def read_orbit(path, *, mission, wavelength_m, ellipsoid_a_m, ellipsoid_b_m):
    root=ET.parse(path).getroot()
    fixed=root.find('Earth_Explorer_Header/Fixed_Header')
    variable=root.find('Earth_Explorer_Header/Variable_Header')
    if fixed is None or variable is None or root.tag!='Earth_Explorer_File':
        raise ValueError('EOF thiếu Earth Explorer header.')
    if fixed.findtext('Mission')!=mission or fixed.findtext('File_Type') not in ('AUX_POEORB','AUX_RESORB','AUX_PREORB'):
        raise ValueError('EOF mission/product type không khớp.')
    if variable.findtext('Ref_Frame')!='EARTH_FIXED' or variable.findtext('Time_Reference')!='UTC':
        raise ValueError('EOF cần EARTH_FIXED và UTC.')
    def utc(text):
        if text is None or not text.startswith('UTC='):
            raise ValueError('EOF timestamp cần tiền tố UTC=.')
        return float(Time(text[4:],scale='utc').gps)
    validity=[utc(fixed.findtext(f'Validity_Period/Validity_{name}')) for name in ('Start','Stop')]
    if validity[0]>=validity[1]:raise ValueError('EOF validity period không hợp lệ.')
    container=root.find('Data_Block/List_of_OSVs')
    if container is None:raise ValueError('EOF không có List_of_OSVs.')
    rows=container.findall('OSV')
    if len(rows)<2 or int(container.get('count','-1'))!=len(rows):
        raise ValueError('EOF OSV count không đúng hoặc ít hơn hai node.')
    times=[];positions=[];velocities=[]
    for row in rows:
        times.append(utc(row.findtext('UTC')))
        vector=[]
        for field,unit in (('X','m'),('Y','m'),('Z','m'),('VX','m/s'),('VY','m/s'),('VZ','m/s')):
            node=row.find(field)
            if node is None or node.get('unit')!=unit:raise ValueError(f'EOF {field} unit không đúng.')
            vector.append(float(node.text))
        if not np.isfinite(vector).all():raise ValueError('EOF OSV không hữu hạn.')
        positions.append(vector[:3]);velocities.append(vector[3:])
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('EOF epochs phải hữu hạn, tăng nghiêm ngặt và không trùng.')
    orbit=EOFOrbit(times,positions,velocities,wavelength_m,
        ellipsoid_a_m=ellipsoid_a_m,ellipsoid_b_m=ellipsoid_b_m)
    orbit.validity_interval_s=validity
    return orbit
