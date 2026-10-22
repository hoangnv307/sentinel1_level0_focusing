"""Lịch acquisition H/K và support thực, không cần annotation L1."""
import unittest
from dataclasses import replace
import numpy as np
from sentinel1_processing.doppler_centroid.layout import context_starts, map_starts
from sentinel1_processing.doppler_centroid.estimation import Config, build_azimuth_blocks


class ContextLayoutTest(unittest.TestCase):
    def starts(self, span, **overrides):
        options = dict(slice_start_times_s=[0], last_slice_stop_time_s=span, current_slice_index=0,
                       block_size_lines=6000, prf_hz=1000, nominal_spacing_s=10,
                       tail_stop_convention='last_line')
        options.update(overrides)
        return context_starts(**options)

    def test_k_thresholds_and_tail_conventions(self):
        for span, expected in [(15,[0,9]),(25,[0,12.5,19]),(29.997,[0,14.9985,23.997]),
                               (30,[0,10,20,24]),(39,[0,13,26,33])]:
            np.testing.assert_allclose(self.starts(span), expected)
        np.testing.assert_allclose(self.starts(25, tail_stop_convention='after_last_line'),[0,12.5,19.001])
        with self.assertRaises(ValueError): self.starts(2)
        with self.assertRaises(ValueError): self.starts(25,tail_stop_convention=None)

    def test_nonfinal_boundary_is_shared_and_not_product_filtered(self):
        times=np.arange(66001)/1000
        config=Config(6000,20,azimuth_placement='slice_timeline',nominal_spacing_s=10,
            tail_stop_convention='last_line',mapping_tolerance_s=1e-6,continuity_tolerance_s=1e-6)
        arguments=dict(n_lines=len(times),prf_hz=1000,azimuth_times_s=times,
            slice_start_times_s=[0,25,50],last_slice_stop_time_s=65,
            product_start_time_s=2,product_stop_time_s=24)
        blocks=build_azimuth_blocks(config=config,**arguments)
        self.assertEqual([b.start_line for b in blocks],[0,12500,25000])
        other=build_azimuth_blocks(config=replace(config,current_slice_index=1),**arguments)
        self.assertEqual(blocks[-1].start_line,other[0].start_line)
        self.assertEqual(blocks[-1].azimuth_time_s,28)

    def test_missing_support_and_gaps_are_reported_without_clamping(self):
        times=np.arange(25001)/1000
        options=dict(block_size_lines=6000,prf_hz=1000,mapping_tolerance_s=1e-6,
                     continuity_tolerance_s=1e-6,missing_support_policy='report')
        rows=map_starts([0,19.001,20,30],times,**options)
        self.assertEqual([r['unavailable_reason'] for r in rows],
                         [None,None,'incomplete_block','start_outside_input'])
        self.assertIsNone(rows[-1]['start_line'])
        with self.assertRaises(ValueError):map_starts([20],times,**{**options,'missing_support_policy':'error'})
        gap=times.copy();gap[5000:]+=.001
        self.assertEqual(map_starts([0],gap,**options)[0]['unavailable_reason'],'packet_gap_or_pri_change')
        with self.assertRaises(ValueError):map_starts([.0004],times,**options)

    def test_large_epoch_mapping_and_custom_starts_validation(self):
        epoch=1_440_000_000.
        starts=self.starts(25+epoch,slice_start_times_s=[epoch])
        times=epoch+np.arange(25001)/1000
        rows=map_starts(starts,times,block_size_lines=6000,prf_hz=1000,
            mapping_tolerance_s=1e-6,continuity_tolerance_s=1e-6,missing_support_policy='error')
        self.assertEqual([r['start_line'] for r in rows],[0,12500,19000])
        for bad in ([-1],[20001],[1.5],[True]):
            with self.assertRaises(ValueError):
                build_azimuth_blocks(25001,1000,Config(6000,20,azimuth_placement='custom'),custom_starts=bad)
