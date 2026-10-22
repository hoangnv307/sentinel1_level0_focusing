"""Job/preflight/comparison độc lập L1 và validation trước decode."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch, Mock
from xml.etree import ElementTree as ET
import numpy as np

from sentinel1_processing.utils import dce_job
from sentinel1_processing.pre_processing.downlink_header_validation.sample_count import rgdec5_stage3_rx_samples

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('s4_dce_job_cli',ROOT/'workflows/sentinel-1/estimate_doppler_centroid.py')
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)


class JobTest(unittest.TestCase):
    def job(self):
        return json.loads((ROOT/'configs/s4_dce_research.json').read_text())

    def test_job_rejects_unknown_keys_nonfinite_values_and_implicit_mode(self):
        for change in ({'mode':'unknown'},{'strdie':3000},{'batch_lines':True},{'batch_lines':0}):
            with TemporaryDirectory() as folder:
                path=Path(folder)/'job.json';path.write_text(json.dumps({**self.job(),**change}))
                with self.assertRaises(ValueError):dce_job.read_job(path)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'job.json';path.write_text('{"mode":NaN}')
            with self.assertRaises(ValueError):dce_job.read_job(path)

    def test_l0_support_profile_needs_no_l1_and_uses_s4_aux_rms(self):
        job=self.job();times=np.arange(16500)*.000606
        config,windows,provenance=dce_job.resolve_layout(job,{'line_times_s':times,'pri_s':.000606},
            {'rms_threshold_hz':20,'dc_method':'Data Analysis'})
        self.assertEqual(config.rms_threshold_hz,20)
        self.assertEqual([w['start_line'] for w in windows],[0,10500])
        self.assertEqual(provenance['policy'],'l0_support')
        job['mode']='strict'
        with self.assertRaises(ValueError):dce_job.resolve_layout(job,{'line_times_s':times,'pri_s':.000606},
            {'rms_threshold_hz':20})

    def test_aux_selection_uses_validity_then_generation_not_payload_age(self):
        from astropy.time import Time
        with TemporaryDirectory() as folder:
            for kind,version in [('INS','3.16'),('PP1','3.15'),('CAL','2.10')]:
                for valid,generated in [('20190228T092500','20240327T102320'),
                                        ('20190228T092500','20250515T131849'),
                                        ('20270101T000000','20270102T000000')]:
                    directory=Path(folder)/f'auxiliary/S1A_AUX_{kind}_V{valid}_G{generated}.SAFE/data'
                    directory.mkdir(parents=True)
                    (directory/f's1a-aux-{kind.lower()}.xml').write_text(f'<payload schemaVersion="{version}"/>')
            result=dce_job.auxiliary_inventory(folder,selection_time_gps=float(Time('2025-09-03').gps))
            self.assertEqual(len(result),3)
            self.assertTrue(all(r['generation_utc']=='2025-05-15T13:18:49' for r in result))
            self.assertTrue(all(r['validity_start_utc']=='2019-02-28T09:25:00' for r in result))

    def test_old_attached_xsd_is_reported_separately_from_supported_payload(self):
        from astropy.time import Time
        with TemporaryDirectory() as folder:
            for kind,version in [('INS','3.16'),('PP1','3.15'),('CAL','2.10')]:
                base=Path(folder)/f'auxiliary/S1A_AUX_{kind}_V20190228T092500_G20250515T131849.SAFE'
                (base/'data').mkdir(parents=True);(base/'support').mkdir()
                (base/f'data/s1a-aux-{kind.lower()}.xml').write_text(f'<payload schemaVersion="{version}"/>')
                (base/f'support/s1-aux-{kind.lower()}.xsd').write_text('<schema version="3.14"/>')
            result=dce_job.auxiliary_inventory(folder,selection_time_gps=float(Time('2025-09-03').gps))
            pp1=next(item for item in result if item['kind']=='PP1')
            self.assertEqual(pp1['payload_version'],'3.15')
            self.assertEqual(pp1['attached_schema']['version'],'3.14')
            self.assertFalse(pp1['schema_version_matches_payload'])
            self.assertIn('sha256',pp1['attached_schema'])

    def test_preflight_without_l1_directory_does_not_decode_or_read_reference(self):
        times=np.arange(16500)*.000606
        inventory={'line_times_s':times,'pri_s':.000606,'chunks':[],'sample_rate_hz':1e6}
        geometry=SimpleNamespace(orbit=SimpleNamespace(validate_time_coverage=Mock(),orbit_times_s=[0,20]),
                                 attitude_times_s=[0,20])
        aux={'rms_threshold_hz':20,'radar_frequency_hz':5e9,'dc_method':'Data Analysis',
             'apply_eap':True,'apply_rsl':True,'rrf_spectrum':'Extended Tapered',
             'ellipsoid_a_m':6378137.,'ellipsoid_b_m':6356752.314245,
             'correct_iq_bias':True,'correct_iq_gain':False,'correct_iq_quadrature':False,
             'correct_rx_variation':True,'perform_internal_calibration':True}
        with TemporaryDirectory() as folder:
            base=Path(folder);(base/'level-0').mkdir();(base/'level-0/s1a-s4-raw-s-vv-test.dat').touch()
            config=base/'job.json';config.write_text(json.dumps(self.job()))
            args=SimpleNamespace(output=base/'out',product=base,config=config,preflight_only=True)
            with patch.object(demo.sentinel1decoder,'Level0File'), \
                    patch.object(dce_job,'header_inventory',return_value=inventory), \
                    patch.object(dce_job,'auxiliary_inventory',return_value=[]), \
                    patch.object(demo,'read_auxiliary',return_value=aux), \
                    patch.object(demo.geometry.Estimator,'from_level0_product',return_value=geometry), \
                    patch.object(demo,'analyze_echoes',side_effect=AssertionError('decode not allowed')), \
                    patch.object(demo,'ReferenceInput',side_effect=AssertionError('L1 not allowed')):
                demo.run_job(args)
            report=json.loads((base/'out/preflight.json').read_text())
            self.assertTrue(report['ready']);self.assertFalse((base/'level-1').exists())
            self.assertEqual(report['resolved_auxiliary']['rms_threshold_hz'],20)

    def test_comparison_masks_range_and_refuses_distant_time_without_mutating_result(self):
        from astropy.time import Time
        from sentinel1_processing.doppler_centroid import estimation as dce
        config=dce.Config(128,5,range_block_size_samples=8,azimuth_spacing_lines=128)
        eta=float(Time('2025-01-01').gps)+np.arange(128)/1000
        tau=.005+np.arange(40)/1e6
        signal=np.tile(np.exp(2j*np.pi*7*np.arange(128)/1000)[:,None],(1,40))
        result=dce.Estimator(1000,config).estimate_scene(signal,tau,azimuth_times_s=eta,known_ambiguity_number=0)[0]
        root=ET.fromstring('<product><imageAnnotation><imageInformation><zeroDopMinusAcqTime>0</zeroDopMinusAcqTime></imageInformation></imageAnnotation><dopplerCentroid><dcEstimateList><dcEstimate><fineDceList/></dcEstimate></dcEstimateList></dopplerCentroid></product>')
        record=root.find('dopplerCentroid/dcEstimateList/dcEstimate')
        for key,value in [('azimuthTime',Time(result.block.azimuth_time_s,format='gps').utc.isot),
                          ('t0','.005'),('dataDcPolynomial','7 0 0')]:ET.SubElement(record,key).text=value
        for value in (.005010,.005020):
            point=ET.SubElement(record.find('fineDceList'),'fineDce');ET.SubElement(point,'slantRangeTime').text=str(value)
        before=result.data_dc_polynomial.copy()
        with TemporaryDirectory() as folder:
            path=Path(folder)/'reference.xml';ET.ElementTree(root).write(path)
            report=dce_job.compare_annotations([result],path,max_time_delta_s=.001)
            self.assertTrue(report['records'][0]['compared'])
            self.assertLess(len(report['records'][0]['range_times_s']),5)
            record.find('azimuthTime').text='2025-01-02T00:00:00';ET.ElementTree(root).write(path)
            report=dce_job.compare_annotations([result],path,max_time_delta_s=.001)
            self.assertEqual(report['records'][0]['reason'],'outside_time_tolerance')
        np.testing.assert_array_equal(before,result.data_dc_polynomial)

    def test_range_gain_factory_uses_aux_lut_and_explicit_rsl_reference(self):
        from contextlib import ExitStack
        job=self.job()
        job['corrections']={'apply_eap':True,'apply_rsl':True,'reference_range_m':1000000.,
            'rrf_spectrum':'aux_nominal_fft','eap':{
                'update_interval_s':1.,'boresight_update_interval_s':1.,
                'antenna_axis_body':[0,0,1],'boresight_source':'attitude','minimum_magnitude':0.,
                'terrain':{'kind':'constant','height_m':0.},
                'solver':{'range_tolerance_m':.1,'height_tolerance_m':.1,'doppler_tolerance_mps':.01,
                          'max_nfev':200,'look_side':'right'}}}
        aux={'apply_eap':True,'apply_rsl':True}
        geom=Mock();geom.attitude_times_s=[0,2]
        geom.boresight_angle_deg.return_value=30.
        geom.terrain_look_angles_deg.side_effect=lambda time,r,**kw:np.full(len(r),30.)
        with TemporaryDirectory() as folder,ExitStack() as resources:
            folder=Path(folder)
            cal=folder/'cal.xml'
            cal.write_text('<auxiliaryCalibration schemaVersion="2.10"><calibrationParamsList><calibrationParams>\n            <swath>S4</swath><polarisation>VV</polarisation><elevationAntennaPattern>\n            <elevationAngleIncrement>0.05</elevationAngleIncrement><values count="3">0 16 0 16 0 16</values>\n            </elevationAntennaPattern></calibrationParams></calibrationParamsList></auxiliaryCalibration>')
            ins=folder/'ins.xml'
            ins.write_text('<instrument schemaVersion="3.16"><rollSteeringParams><referenceAntennaAngle>29.45</referenceAntennaAngle><referenceHeight>711700</referenceHeight><rollSteeringSensitivity>0.0000566</rollSteeringSensitivity></rollSteeringParams></instrument>')
            inventory=[{'kind':'CAL','path':str(cal)},{'kind':'INS','path':str(ins)}]
            factory,ledger=demo.prepare_range_gain(job,aux,inventory,{'line_times_s':np.array([0.,1.])},geom,resources)
            ranges=np.array([800000.,900000.]);tau=2*ranges/299792458.
            provider=factory(tau)
            expected=-.25j*(ranges/1000000.)**1.5
            np.testing.assert_allclose(provider(np.array([0.,.5,1.])),np.tile(expected,(3,1)))
            self.assertEqual(ledger[0]['boresight_sources'],['L0 attitude','L0 attitude'])
            self.assertEqual(geom.terrain_look_angles_deg.call_count,2)

    def test_s4_pdu_table_known_swl_values(self):
        # Direct table arithmetic: B=2*300-92-17=491, divmod(B,8)=(61,3), D[3]=1.
        self.assertEqual(rgdec5_stage3_rx_samples(300),370)
        self.assertEqual(rgdec5_stage3_rx_samples(301),372)
