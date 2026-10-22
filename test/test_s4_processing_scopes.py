"""Segmentation và pipeline thực trên echo tổng hợp, không cần L0/L1 ngoài repo."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from xml.etree import ElementTree as ET
import json
import unittest
import numpy as np
import pandas as pd
from astropy.time import Time
import sentinel1decoder
from sentinel1decoder.enums import RxChannelId, SignalType, RangeDecimation
from sentinel1_processing.utils import dce_job
from test_s4_dce_job import demo, ROOT


class ProcessingScopesTest(unittest.TestCase):
    def fixture(self, *, gap=False):
        epoch=float(Time('2025-01-01').gps)
        eta=epoch+np.r_[np.arange(8)*.001,.008+np.arange(8)*.002]
        if gap:eta[8:]+=.1
        frame=pd.DataFrame({
            'Signal Type':[SignalType.ECHO]*16,'Error Flag':[False]*16,'Swath Number':[3]*16,
            'Rx Channel ID':[RxChannelId.RXV_POL_CHANNEL]*16,'Polarisation':['Tx V, Rx V+H']*16,
            'PRI':np.r_[np.full(8,.001),np.full(8,.002)],'Range Decimation':[RangeDecimation.RGDEC_5]*16,
            'Tx Pulse Length':[1e-6]*16,'Tx Ramp Rate':[0.]*16,'Tx Pulse Start Frequency':[0.]*16,
            'Rank':[1]*16,'Rx Gain':[-6.]*16,'Instrument Configuration ID':[7]*16,
            'Number of Quads':[185]*16,'SWL':[300/sentinel1decoder.constants.F_REF]*16,
            'SWST':[.00009]*16,'Coarse Time':np.floor(eta),'Fine Time':eta-np.floor(eta),
            'Space Packet Count':np.arange(16),'Packet Sequence Count':np.arange(16)}, index=np.arange(100,116))
        original=SimpleNamespace(filename='s1a-s4-raw-s-vv-synthetic.dat',acquisition_chunks=[13],packet_metadata=frame,
            get_acquisition_chunk_metadata=lambda chunk:frame)
        return original,frame,eta

    def job(self):
        job=json.loads((ROOT/'configs/s4_dce_research.json').read_text())
        job['layout'].update(block_size_lines=4,num_range_blocks=5,range_block_size_samples=8,
                             mapping_tolerance_s=3e-6,continuity_tolerance_s=3e-6)
        job['corrections'].update(internal_delay_source='none',perform_internal_calibration=False,
                                 apply_rsl=True,reference_range_m=1000000.)
        job['batch_lines']=3
        return job

    def aux(self):
        return dict(rms_threshold_hz=20.,radar_frequency_hz=5e9,dc_method='Data Analysis',
            apply_eap=False,apply_rsl=False,rrf_spectrum='Extended Tapered',ellipsoid_a_m=6378137.,
            ellipsoid_b_m=6356752.314245,correct_iq_bias=False,correct_iq_gain=False,
            correct_iq_quadrature=False,correct_rx_variation=False,perform_internal_calibration=False,
            swst_bias_s=0.,nominal_amplitude_coefficients=[1,0,0,0],nominal_phase_coefficients_cycles=[0,0,0,0])

    def test_header_splits_pri_and_gap_and_keeps_swst_virtual_chunks_in_scope(self):
        level0,frame,eta=self.fixture(gap=True)
        frame.loc[104:107,'SWST']+=1e-6
        inventory=dce_job.header_inventory(level0,continuity_tolerance_s=3e-6)
        self.assertEqual(len(inventory['_scope_views']),2)
        first,second=inventory['_scope_views']
        self.assertEqual([len(first.get_acquisition_chunk_metadata(c)) for c in first.acquisition_chunks],[4,4])
        self.assertEqual(second.global_start_line,8)
        self.assertIn('changed:PRI',second.split_reasons)
        self.assertIn('acquisition_gap',second.split_reasons)
        self.assertEqual(list(second.get_acquisition_chunk_metadata(second.acquisition_chunks[0]).index),list(range(108,116)))
        plans=dce_job.resolve_processing_scopes(level0,inventory,self.job(),self.aux())
        self.assertEqual([p['headers']['pri_s'] for p in plans],[.001,.002])
        self.assertEqual([[w['global_start_line'] for w in p['windows']] for p in plans],[[0,4],[8,12]])

    def test_packet_counter_gap_splits_scope_and_duplicate_packet_is_rejected(self):
        level0,frame,_=self.fixture()
        frame['PRI']=.001
        frame['Fine Time']=np.arange(16)*.001
        for key in ('Space Packet Count','Packet Sequence Count'):frame.loc[108:,key]+=1
        inventory=dce_job.header_inventory(level0,continuity_tolerance_s=3e-6)
        self.assertEqual(len(inventory['_scope_views']),2)
        self.assertIn('packet_count_gap',inventory['_scope_views'][1].split_reasons)
        frame.loc[109,'Space Packet Count']=frame.loc[108,'Space Packet Count']
        with self.assertRaisesRegex(ValueError,'lặp'):
            dce_job.header_inventory(level0,continuity_tolerance_s=3e-6)

    def test_context_block_cannot_cross_a_pri_boundary_or_get_clamped(self):
        level0,frame,eta=self.fixture()
        headers=dce_job.header_inventory(level0,continuity_tolerance_s=3e-6)
        job=self.job();job['layout'].update(policy='slice_context',missing_support_policy='report')
        job['context']={'slice_start_times_gps_s':[float(eta[6]),float(eta[10])],
            'last_slice_stop_time_gps_s':float(eta[-1]),'current_slice_index':0,'source':'synthetic job context'}
        plans=dce_job.resolve_processing_scopes(level0,headers,job,self.aux())
        windows=[w for p in plans for w in p['windows']]
        self.assertFalse(windows[0]['available']);self.assertEqual(windows[0]['unavailable_reason'],'incomplete_block')
        self.assertTrue(windows[1]['available']);self.assertEqual(windows[1]['global_start_line'],10)

    def environment(self, folder, stack, *, job=None):
        level0,frame,eta=self.fixture()
        base=Path(folder);(base/'level-0').mkdir();raw=base/'level-0/s1a-s4-raw-s-vv-synthetic.dat';raw.touch()
        level0.filename=str(raw)
        path=base/'job.json';path.write_text(json.dumps(self.job() if job is None else job))
        geom=SimpleNamespace(orbit=SimpleNamespace(validate_time_coverage=Mock(),orbit_times_s=[eta[0]-1,eta[-1]+1]),
            attitude_times_s=[eta[0]-1,eta[-1]+1],estimate=lambda time,tau,**kwargs:np.full(len(tau),7.))
        decoder=SimpleNamespace(decode_packets=lambda headers:np.tile(np.exp(2j*np.pi*7*(
            headers['Coarse Time'].to_numpy(float)+headers['Fine Time'].to_numpy(float)-eta[0]))[:,None],(1,370)))
        calibration={'internal_delay':{'available':True,'mean_s':0.},'quality':{'interpolation_available':False}}
        stack.enter_context(patch.object(demo.sentinel1decoder,'Level0File',return_value=level0))
        stack.enter_context(patch.object(demo.sentinel1decoder,'Level0Decoder',return_value=decoder))
        stack.enter_context(patch.object(dce_job,'auxiliary_inventory',return_value=[]))
        stack.enter_context(patch.object(demo,'read_auxiliary',return_value=self.aux()))
        stack.enter_context(patch.object(demo.geometry.Estimator,'from_level0_product',return_value=geom))
        stack.enter_context(patch.object(demo,'analyze_calibration',return_value=([],[],calibration)))
        return SimpleNamespace(product=base,config=path,output=base/'out',preflight_only=False,
            compare_annotation=None,comparison_max_time_delta_s=.1,orbit_eof=None)

    def test_full_runner_processes_two_prfs_and_reference_change_does_not_change_processing_bytes(self):
        with TemporaryDirectory() as folder,ExitStack() as stack:
            args=self.environment(folder,stack)
            demo.run_job(args)
            output=args.output/'doppler_centroid.json'
            before=output.read_bytes();result=json.loads(before)
            self.assertEqual([r['start_line'] for r in result['estimates']],[0,4,8,12])
            self.assertEqual([r['processing_scope_id'] for r in result['estimates']],['scope-0']*2+['scope-1']*2)
            self.assertTrue(result['preflight']['range_support_validated'])
            for record in result['estimates']:np.testing.assert_allclose(record['selected_dc_hz'],7,atol=.01)
            reference=Path(folder)/'reference.xml'
            root=ET.fromstring('<product><imageAnnotation><imageInformation><zeroDopMinusAcqTime>0</zeroDopMinusAcqTime></imageInformation></imageAnnotation><dopplerCentroid><dcEstimateList><dcEstimate><fineDceList/></dcEstimate></dcEstimateList></dopplerCentroid></product>')
            record=root.find('dopplerCentroid/dcEstimateList/dcEstimate')
            for name,value in [('azimuthTime',Time(result['estimates'][0]['acquisition_center_time_gps_s'],format='gps').utc.isot),('t0','0.001'),('dataDcPolynomial','7 0 0')]:
                ET.SubElement(record,name).text=value
            for value in (.0001,.1):
                point=ET.SubElement(record.find('fineDceList'),'fineDce');ET.SubElement(point,'slantRangeTime').text=str(value)
            ET.ElementTree(root).write(reference)
            args.compare_annotation=reference
            demo.run_job(args);self.assertEqual(output.read_bytes(),before)
            original_comparison=json.loads((args.output/'comparison.json').read_text())
            record.find('dataDcPolynomial').text='70 0 0';ET.ElementTree(root).write(reference)
            demo.run_job(args);self.assertEqual(output.read_bytes(),before)
            changed=json.loads((args.output/'comparison.json').read_text())
            self.assertNotEqual(original_comparison['records'],changed['records'])
            self.assertFalse((Path(folder)/'level-1').exists())

    def test_bad_range_support_fails_before_image_decode_and_is_saved_in_preflight(self):
        with TemporaryDirectory() as folder,ExitStack() as stack:
            job=self.job();job['layout']['range_block_size_samples']=1000
            args=self.environment(folder,stack,job=job)
            with patch.object(demo,'analyze_echoes',side_effect=AssertionError('image decode not allowed')):
                with self.assertRaises(ValueError):demo.run_job(args)
            report=json.loads((args.output/'preflight.json').read_text())
            self.assertFalse(report['ready']);self.assertFalse(report['range_support_validated'])
            self.assertTrue(any('ROI' in error for error in report['errors']))

    def test_calibration_failure_marks_preflight_failed_before_echo_decode(self):
        with TemporaryDirectory() as folder,ExitStack() as stack:
            job=self.job();job['corrections']['internal_delay_source']='l0'
            args=self.environment(folder,stack,job=job)
            unavailable={'internal_delay':{'available':False,'mean_s':None},
                         'quality':{'interpolation_available':False}}
            with patch.object(demo,'analyze_calibration',return_value=([],[],unavailable)), \
                    patch.object(demo,'analyze_echoes',side_effect=AssertionError('echo decode not allowed')):
                with self.assertRaisesRegex(ValueError,'internal delay'):demo.run_job(args)
            report=json.loads((args.output/'preflight.json').read_text())
            self.assertFalse(report['ready']);self.assertFalse(report['range_support_validated'])
            self.assertTrue(any('internal delay' in error for error in report['errors']))
            self.assertEqual(json.loads((args.output/'calibration.json').read_text()),unavailable)

    def test_missing_rsl_reference_retains_requested_correction_in_failed_preflight(self):
        with TemporaryDirectory() as folder,ExitStack() as stack:
            job=self.job();del job['corrections']['reference_range_m']
            args=self.environment(folder,stack,job=job)
            with patch.object(demo,'analyze_echoes',side_effect=AssertionError('echo decode not allowed')):
                with self.assertRaisesRegex(ValueError,'reference_range_m'):demo.run_job(args)
            report=json.loads((args.output/'preflight.json').read_text())
            self.assertFalse(report['ready'])
            rsl=next(item for item in report['correction_ledger'] if item['name']=='apply_rsl')
            self.assertTrue(rsl['requested']);self.assertFalse(rsl['applied'])
