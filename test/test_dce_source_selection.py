"""Source selection/fallback trên dữ liệu biết trước; lỗi geometry không bị nuốt."""
import unittest
from dataclasses import replace
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import numpy as np
from sentinel1_processing.doppler_centroid import estimation as dce
from sentinel1_processing.utils.dce_job import write_estimates


class SourceSelectionTest(unittest.TestCase):
    def setUp(self):
        self.config=dce.Config(128,5,range_block_size_samples=8,rms_threshold_hz=20,
                               unwrap_weighting='uniform',fit_weighting='uniform',azimuth_spacing_lines=128)
        self.times=np.arange(128)/1000
        self.tau=.005+np.arange(40)/1e6
        self.geometry=lambda t,tau:np.full(len(tau),1007.)
        self.signal=np.tile(np.exp(2j*np.pi*7*self.times)[:,None],(1,40))

    def run_scene(self,data=None,**kwargs):
        return dce.Estimator(1000,self.config).estimate_scene(self.signal if data is None else data,
            self.tau,azimuth_times_s=self.times,**kwargs)[0]

    def test_absolute_data_and_streaming_use_same_selection(self):
        record=self.run_scene(geometry_dc_provider=self.geometry)
        self.assertEqual(record.selected_source,'data')
        np.testing.assert_allclose(record.evaluate(self.tau),1007,atol=1e-8)
        segments=[dce.Segment(self.signal[:64],self.tau,self.times[:64]),
                  dce.Segment(self.signal[64:],self.tau,self.times[64:])]
        other=dce.Estimator(1000,self.config).estimate_segments(segments,geometry_dc_provider=self.geometry)[0]
        self.assertEqual(other.selection_reason,record.selection_reason)
        np.testing.assert_allclose(record.evaluate(self.tau),other.evaluate(self.tau),atol=1e-8)

    def test_geometry_is_computed_before_fine_and_zero_input_falls_back(self):
        calls=[]
        original=dce.estimate_fine_dc
        def geom(t,tau):calls.append('geometry');return self.geometry(t,tau)
        def fine(*a,**k):calls.append('fine');return original(*a,**k)
        with patch.object(dce,'estimate_fine_dc',side_effect=fine):
            result=self.run_scene(np.zeros_like(self.signal),geometry_dc_provider=geom)
        self.assertEqual(calls,['geometry','fine'])
        self.assertEqual(result.selected_source,'geometry')
        self.assertEqual(result.selection_reason,'invalid_accc_or_insufficient_points')
        self.assertIsNone(result.data_dc_polynomial)
        np.testing.assert_allclose(result.evaluate(self.tau),1007)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'result.json';write_estimates(path,[result],{})
            saved=json.loads(path.read_text())['estimates'][0]
            self.assertIsNone(saved['data_dc_polynomial']);self.assertIsNone(saved['rms_error_hz'])
            self.assertEqual(saved['selected_source'],'geometry')

    def test_rms_failure_keeps_data_diagnostics_and_selects_geometry(self):
        self.config=replace(self.config,rms_threshold_hz=.01)
        fine=np.array([0.,4.,0.,4.,0.]);coherence=np.ones(5)
        with patch.object(dce,'estimate_fine_dc',return_value=(fine,np.exp(-2j*np.pi*fine/1000),coherence)):
            record=self.run_scene(geometry_dc_provider=self.geometry)
        self.assertEqual(record.selection_reason,'rms_above_threshold')
        self.assertIsNotNone(record.data_dc_polynomial)
        self.assertTrue(record.rms_above_threshold)
        np.testing.assert_allclose(record.evaluate(self.tau),1007)

    def test_config_and_geometry_errors_propagate(self):
        for config in (replace(self.config,unwrap_fft_length=0),replace(self.config,fit_weighting='bad'),
                       replace(self.config,rms_threshold_hz=float('nan'))):
            with self.assertRaises(ValueError):dce.Estimator(1000,config)
        for provider in (lambda t,tau:np.full(len(tau),np.nan),lambda t,tau:np.zeros(2)):
            with self.assertRaises(ValueError):self.run_scene(geometry_dc_provider=provider)
        def failed(t,tau):raise RuntimeError('programming failure')
        with self.assertRaisesRegex(RuntimeError,'programming failure'):
            self.run_scene(geometry_dc_provider=failed)
        with self.assertRaises(dce.DataQualityError):self.run_scene(np.zeros_like(self.signal))

    def test_geometry_and_predefined_skip_data_estimation(self):
        self.config=replace(self.config,dc_method='Geometry')
        with patch.object(dce,'estimate_fine_dc',side_effect=AssertionError('data not needed')):
            result=self.run_scene(geometry_dc_provider=self.geometry)
        self.assertEqual(result.selection_reason,'configured_geometry')
        self.config=replace(self.config,dc_method='Predefined',predefined_coefficients=(3.,),
            predefined_t0_s=.005,predefined_range_support_s=(.005,.006),predefined_time_support_s=(0,1))
        result=self.run_scene()
        self.assertEqual(result.selected_source,'predefined')
        np.testing.assert_allclose(result.evaluate(self.tau),3)
        self.config=replace(self.config,predefined_time_support_s=(2,3))
        with self.assertRaises(ValueError):self.run_scene()


    def test_unavailable_data_stays_null_in_audit_regression_and_plot_exports(self):
        import matplotlib.pyplot as plt
        from xml.etree import ElementTree as ET
        from sentinel1_processing.utils.dce_plotting import plot_comparisons
        from test_sentinel1_dce_demo import demo,audit
        record=self.run_scene(np.zeros_like(self.signal),geometry_dc_provider=self.geometry)
        reference={'azimuth_s':record.block.azimuth_time_s,'t0':record.t0_s,
                   'dataDcPolynomial':[1007,0,0],'geometryDcPolynomial':[1007,0,0]}
        comparison=dce.compare_with_annotations([reference],[record],record.range_times_s)[0]
        self.assertIsNone(comparison['fit_rms_hz'])
        self.assertEqual(comparison['fit_points_hz'].size,0)
        figures=plot_comparisons([comparison])
        for figure in figures:plt.close(figure)
        annotated=ET.Element('dcEstimate')
        ET.SubElement(annotated,'t0').text=str(record.t0_s)
        ET.SubElement(annotated,'dataDcPolynomial').text='1007 0 0'
        ET.SubElement(annotated,'geometryDcPolynomial').text='1007 0 0'
        points=ET.SubElement(annotated,'fineDceList')
        for tau in record.range_times_s:
            point=ET.SubElement(points,'fineDce')
            ET.SubElement(point,'slantRangeTime').text=str(tau)
            ET.SubElement(point,'frequency').text='7'
        audited=audit.compare(record,annotated)
        self.assertIsNone(audited['fine_rmse_hz']);self.assertIsNone(audited['fit_rms_hz'])
        self.assertIsNone(audited['fine_baseband_hz'])
        json.dumps(audited,allow_nan=False)
        with TemporaryDirectory() as folder:
            demo.save_results(Path(folder),[record],[annotated],{},0.)
            result=json.loads((Path(folder)/'doppler_centroid.json').read_text())['estimates'][0]
        self.assertIsNone(result['data_dc_polynomial']);self.assertIsNone(result['rms_error_hz'])
        self.assertIsNone(result['fine_baseband_hz']);self.assertEqual(result['selected_source'],'geometry')
