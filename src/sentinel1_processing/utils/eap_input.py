"""Đọc EAP LUT AUX_CAL; grid tương đối tâm LUT, không dùng nominal near/far."""
from xml.etree import ElementTree as ET
import numpy as np


def read_eap_lut(path, *, swath, polarisation):
    root=ET.parse(path).getroot()
    if root.tag!='auxiliaryCalibration' or root.get('schemaVersion')!='2.10':
        raise ValueError('EAP reader hỗ trợ AUX_CAL payload 2.10.')
    records=[row for row in root.findall('calibrationParamsList/calibrationParams')
             if row.findtext('swath')==swath and row.findtext('polarisation')==polarisation]
    if len(records)!=1:raise ValueError('Cần đúng một calibration swath/polarisation.')
    pattern=records[0].find('elevationAntennaPattern')
    if pattern is None:raise ValueError('AUX_CAL thiếu EAP.')
    values=pattern.find('values')
    if values is None:raise ValueError('AUX_CAL thiếu complex EAP values.')
    count=int(values.get('count','-1'))
    pairs=np.fromstring(values.text or '',sep=' ')
    increment=float(pattern.findtext('elevationAngleIncrement'))
    if (count<2 or pairs.size!=2*count or not np.isfinite(pairs).all()
            or not np.isfinite(increment) or increment<=0):
        raise ValueError('AUX_CAL EAP count/increment/values không hợp lệ.')
    lut=pairs[0::2]+1j*pairs[1::2]
    return (np.arange(count)-(count-1)/2)*increment,lut


def read_roll_law(path):
    root=ET.parse(path).getroot()
    if root.get('schemaVersion')!='3.16':raise ValueError('Roll-law reader cần AUX_INS 3.16.')
    roll=root.find('rollSteeringParams')
    if roll is None:raise ValueError('AUX_INS thiếu roll steering parameters.')
    result={name:float(roll.findtext(field)) for name,field in (
        ('reference_angle_deg','referenceAntennaAngle'),('reference_height_m','referenceHeight'),
        ('roll_sensitivity_deg_per_m','rollSteeringSensitivity'))}
    if not np.isfinite(list(result.values())).all():raise ValueError('Roll steering parameters không hữu hạn.')
    return result
