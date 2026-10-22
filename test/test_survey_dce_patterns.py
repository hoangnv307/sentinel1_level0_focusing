"""Kiểm các thống kê theo scene/group và giả thuyết biên cuối ngoài support."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from test_copernicus_metadata import annotation

workflow = Path(__file__).parents[1]/'workflows/sentinel-1'
spec = importlib.util.spec_from_file_location('check_dce_azimuth_layout', workflow/'check_dce_azimuth_layout.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
spec = importlib.util.spec_from_file_location('survey_dce_patterns', workflow/'survey_dce_patterns.py')
survey = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'check_dce_azimuth_layout': layout}):
    spec.loader.exec_module(survey)


class SurveyTest(unittest.TestCase):
    def test_end_start_hypothesis_is_reported_without_replacing_other_models(self):
        root = ET.fromstring(annotation())
        image = root.find('imageAnnotation/imageInformation')
        for tag,value in [('productFirstLineUtcTime','2025-03-30T18:04:11.300000'),
                          ('productLastLineUtcTime','2025-03-30T18:04:36.300000'),
                          ('azimuthTimeInterval','0.001'),('azimuthPixelSpacing','7'),('numberOfLines','25001')]:
            ET.SubElement(image,tag).text=value
        record = root.findall('dopplerCentroid/dcEstimateList/dcEstimate')[-1]
        record.find('fineDceAzimuthStartTime').text='2025-03-30T18:04:36.300000'
        record.find('fineDceAzimuthStopTime').text='2025-03-30T18:04:42.300000'
        record.find('azimuthTime').text='2025-03-30T18:04:39.300000'
        with TemporaryDirectory() as directory:
            path=Path(directory)/'sample.xml';ET.ElementTree(root).write(path)
            row=survey.read_annotation(path,block_size_lines=6000,nominal_spacing_s=10,tolerance_s=5e-6)
        self.assertTrue(row['hypotheses']['end_start']['matches'])
        self.assertFalse(row['hypotheses']['last_line']['matches'])
        self.assertFalse(row['hypotheses']['after_last_line']['matches'])
        self.assertAlmostEqual(row['tail_stop_offset_lines'],6000)
        self.assertAlmostEqual(row['output_azimuth_extent_m'],175000)
        result=survey.summarize([row])
        self.assertEqual(result['datatake_groups'][0]['metadata_extent_s'],25)
        self.assertEqual(result['rms_flags']['missing'],3)
        self.assertTrue(result['spacing_constraint']['feasible'])

    def test_catalogue_deduplicates_uuid_across_queries_and_keeps_truncation(self):
        prefix='S1A_S4_SLC__1SDV_20250101T'
        products=[{'Id':'one','Name':prefix+'000000_20250101T000025_000100_000ABC_0001.SAFE'},
                  {'Id':'two','Name':prefix+'000025_20250101T000035_000101_000ABC_0002.SAFE'}]
        with TemporaryDirectory() as directory:
            base=Path(directory)
            for name in ['query1','query2']:
                (base/(name+'.json')).write_text(json.dumps({'filter':'test','truncated':True,'products':products}))
            result=survey.catalogue_summary(base)
            outside_cohort=survey.catalogue_summary(base,min_year=2026)
        self.assertEqual(result['products'],2)
        self.assertEqual(result['unique_acquisitions'],2)
        self.assertEqual(result['datatake_swath_groups'],1)
        self.assertEqual(result['observed_product_envelope_s']['max'],35)
        self.assertTrue(all(s['truncated'] for s in result['sources']))
        self.assertEqual(outside_cohort['products'],0)
        self.assertEqual(outside_cohort['all_catalogue_products'],2)


if __name__=='__main__':
    unittest.main()
