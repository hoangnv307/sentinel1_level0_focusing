"""Kiểm chứng bộ so sánh lịch DCE bằng metadata tổng hợp, không cần file ngoài repo."""

from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET


spec = importlib.util.spec_from_file_location(
    "check_dce_azimuth_layout",
    Path(__file__).parents[1] / "workflows/sentinel-1/check_dce_azimuth_layout.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class DceAzimuthLayoutTest(unittest.TestCase):
    def test_spacing_changes_across_thirty_second_boundary(self):
        self.assertEqual(layout.candidate_starts(29.997, 3, nominal_spacing_s=10), [0, 14.9985])
        self.assertEqual(layout.candidate_starts(30.006, 3, nominal_spacing_s=10), [0, 10.002, 20.004])
        self.assertEqual(layout.candidate_starts(15, 3, nominal_spacing_s=10), [0])
        self.assertEqual(layout.candidate_starts(25, 3, nominal_spacing_s=12), [0, 12.5])

    def test_bad_inputs_and_extra_estimates_are_rejected(self):
        for duration, block, spacing in [(0, 3, 10), (2, -3, 10), (30, 3, 0), (30, 3, float('nan'))]:
            with self.assertRaises(ValueError):
                layout.candidate_starts(duration, block, nominal_spacing_s=spacing)
        self.assertFalse(layout.compare_starts([0, 10, 20], [0, 20], tolerance_s=1e-6)['matches'])
        with self.assertRaises(ValueError):
            layout.analyze_annotation('unused.xml', azimuth_block_size_lines=6000,
                nominal_spacing_s=10, time_tolerance_s=5e-6, tail_stop_convention='automatic')

    def test_annotation_timing_and_tail_use_different_endpoint_conventions(self):
        root = ET.Element('product')
        def put(path, value):
            parent = root
            for name in path.split('/'):
                child = parent.find(name)
                if child is None:
                    child = ET.SubElement(parent, name)
                parent = child
            parent.text = str(value)
        epoch = datetime(2025, 1, 1)
        def stamp(seconds):
            return (epoch + timedelta(seconds=seconds)).isoformat(timespec='microseconds')
        put('adsHeader/swath', 'S4')
        put('adsHeader/polarisation', 'VV')
        dl = 'generalAnnotation/downlinkInformationList/downlinkInformation/'
        sl = 'imageAnnotation/imageInformation/sliceList/slice/'
        for prefix in (dl, sl):
            put(prefix + ('firstLineSensingTime' if prefix == dl else 'sensingStartTime'), stamp(0))
            put(prefix + ('lastLineSensingTime' if prefix == dl else 'sensingStopTime'), stamp(25))
        put(dl + 'downlinkValues/pri', .001)
        put('imageAnnotation/imageInformation/zeroDopMinusAcqTime', .3)
        put('imageAnnotation/imageInformation/sliceNumber', 1)
        put(sl + 'sliceNumber', 1)
        put('dopplerCentroid/dcEstimateList', '')
        # 25.000 s between first/last lines = 25,001 lines. Last block starts at 19.001 s.
        for start in (0, 12.5, 19.001):
            record = ET.SubElement(root.find('dopplerCentroid/dcEstimateList'), 'dcEstimate')
            for key, value in [('fineDceAzimuthStartTime', start + .3),
                               ('fineDceAzimuthStopTime', start + 6.3),
                               ('azimuthTime', start + 3.3)]:
                ET.SubElement(record, key).text = stamp(value)
            ET.SubElement(record, 'fineDceList')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'annotation.xml'
            ET.ElementTree(root).write(path)
            result = layout.analyze_annotation(path, azimuth_block_size_lines=6000,
                                               nominal_spacing_s=10, time_tolerance_s=5e-6)
            self.assertTrue(result['matches'])
            self.assertEqual(result['inferred_line_count'], 25001)
            self.assertFalse(result['models']['fixed_3000_lines']['matches'])
            root.find('dopplerCentroid/dcEstimateList/dcEstimate/azimuthTime').text = stamp(4)
            ET.ElementTree(root).write(path)
            self.assertFalse(layout.analyze_annotation(
                path, azimuth_block_size_lines=6000, nominal_spacing_s=10,
                time_tolerance_s=5e-6)['matches'])

    def test_adjacent_slices_share_acquisition_window_despite_different_zd_offsets(self):
        epoch = datetime(2025, 1, 1)
        def stamp(seconds):
            return (epoch + timedelta(seconds=seconds)).isoformat(timespec='microseconds')
        context = [(0, 32), (25, 57), (50, 65)]
        with TemporaryDirectory() as directory:
            nested = Path(directory) / 'nested'
            nested.mkdir()
            results = []
            for index, (start, stop) in enumerate(context):
                offset = (.3, .5, .4)[index]
                root = ET.fromstring(f'''<product><adsHeader><missionId>S1C</missionId>
                    <absoluteOrbitNumber>100</absoluteOrbitNumber><missionDataTakeId>123</missionDataTakeId>
                    <swath>S6</swath><polarisation>HH</polarisation></adsHeader>
                    <generalAnnotation><downlinkInformationList><downlinkInformation>
                    <firstLineSensingTime>{stamp(start)}</firstLineSensingTime>
                    <lastLineSensingTime>{stamp(stop)}</lastLineSensingTime>
                    <downlinkValues><pri>0.001</pri></downlinkValues>
                    </downlinkInformation></downlinkInformationList></generalAnnotation>
                    <imageAnnotation><imageInformation><sliceNumber>{index+1}</sliceNumber>
                    <zeroDopMinusAcqTime>{offset}</zeroDopMinusAcqTime><sliceList/>
                    </imageInformation></imageAnnotation><dopplerCentroid><dcEstimateList/>
                    </dopplerCentroid></product>''')
                for n, (a, b) in enumerate(context, 1):
                    element = ET.SubElement(root.find('imageAnnotation/imageInformation/sliceList'), 'slice')
                    for tag, value in [('sliceNumber', str(n)), ('sensingStartTime', stamp(a)),
                                       ('sensingStopTime', stamp(b))]:
                        ET.SubElement(element, tag).text = value
                starts = ([0, 12.5, 25], [25, 37.5, 50], [50, 59])[index]
                for value in starts:
                    element = ET.SubElement(root.find('dopplerCentroid/dcEstimateList'), 'dcEstimate')
                    for tag, time_value in [('fineDceAzimuthStartTime', value+offset),
                                            ('fineDceAzimuthStopTime', value+offset+6),
                                            ('azimuthTime', value+offset+3)]:
                        ET.SubElement(element, tag).text = stamp(time_value)
                    ET.SubElement(element, 'fineDceList')
                path = nested / f'slice{index+1}.xml'
                ET.ElementTree(root).write(path)
                result = layout.analyze_annotation(path, azimuth_block_size_lines=6000,
                    nominal_spacing_s=10, time_tolerance_s=5e-6, tail_stop_convention='last_line')
                self.assertTrue(result['matches'])
                self.assertFalse(result['models']['adaptive_span']['matches'])
                results.append(result)
                if index == 2:
                    other = layout.analyze_annotation(path, azimuth_block_size_lines=6000,
                        nominal_spacing_s=10, time_tolerance_s=5e-6, tail_stop_convention='after_last_line')
                    self.assertFalse(other['matches'])
            results[1]['absolute_orbit']='101'
            results[2]['absolute_orbit']='101'
            shared = layout.shared_windows(results, time_tolerance_s=5e-6)
            self.assertEqual(len(shared), 2)
            self.assertEqual([(p['left_record'], p['right_record']) for p in shared], [(2, 0), (2, 0)])
            self.assertTrue(all(p['max_time_error_s'] == 0 for p in shared))
            import json
            import subprocess
            import sys
            output = Path(directory) / 'report.json'
            result = subprocess.run([sys.executable, str(Path(layout.__file__)), directory,
                                     '--tail-stop-convention', 'last_line', '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(len(report['annotations']), 3)
            self.assertEqual(len(report['shared_acquisition_windows']), 2)
            results[1]['slice_context'][0]['stop'] = stamp(31)
            with self.assertRaises(ValueError):
                layout.shared_windows(results, time_tolerance_s=5e-6)


if __name__ == '__main__':
    unittest.main()
