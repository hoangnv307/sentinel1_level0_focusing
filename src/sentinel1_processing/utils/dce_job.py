"""Cấu hình/preflight S4; chỉ đọc L0/AUX/job, không import annotation reader."""

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from astropy.time import Time
import numpy as np
import sentinel1decoder
from sentinel1decoder.enums import SignalType, RxChannelId

from ..doppler_centroid import estimation
from ..doppler_centroid.layout import context_starts, map_starts
from ..pre_processing.downlink_header_validation.sample_count import rgdec5_stage3_rx_samples


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def read_job(path):
    def reject_constant(value):
        raise ValueError(f'JSON constant không hợp lệ: {value}')
    def finite_float(value):
        number=float(value)
        if not np.isfinite(number):raise ValueError('Job JSON không cho phép số vô hạn.')
        return number
    job = json.loads(Path(path).read_text(), parse_constant=reject_constant,parse_float=finite_float)
    if not isinstance(job, dict):
        raise ValueError('Job phải là JSON object.')
    unknown = set(job)-{'mode','layout','estimator','corrections','batch_lines','context','minimum_far_range_discard_samples','calibration','geometry'}
    if unknown:
        raise ValueError(f'Job có key không hỗ trợ: {sorted(unknown)}')
    if job.get('mode') not in ('strict','research'):
        raise ValueError('Job cần mode strict hoặc research.')
    for name in ('layout','estimator','corrections'):
        if not isinstance(job.get(name),dict):
            raise ValueError(f'Job cần object {name}.')
    for name in ('batch_lines','minimum_far_range_discard_samples'):
        value=job.get(name,256 if name=='batch_lines' else 0)
        if isinstance(value,bool) or not isinstance(value,int) or value<(1 if name=='batch_lines' else 0):
            raise ValueError(f'{name} không hợp lệ.')
    for key in ('mapping_tolerance_s','continuity_tolerance_s'):
        value=job['layout'].get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
            raise ValueError(f'{key} cần số hữu hạn không âm.')
    calibration=job.get('calibration')
    if not isinstance(calibration,dict) or set(calibration)!={'oversampling_factor','denominator_relative_floor','output_normalization'}:
        raise ValueError('Calibration cần oversampling_factor, denominator_relative_floor và output_normalization tường minh.')
    factor=calibration['oversampling_factor'];floor=calibration['denominator_relative_floor']
    if isinstance(factor,bool) or not isinstance(factor,int) or factor<2:
        raise ValueError('Calibration oversampling_factor cần số nguyên >=2.')
    if isinstance(floor,bool) or not isinstance(floor,(int,float)) or not 0<=floor<1:
        raise ValueError('Calibration denominator_relative_floor phải trong [0,1).')
    if calibration['output_normalization'] not in ('absolute','relative'):
        raise ValueError('PG normalization không hỗ trợ.')
    if calibration['output_normalization']=='relative' and job['mode']!='research':
        raise ValueError('PG relative normalization chỉ dùng research.')
    geometry=job.get('geometry',{})
    if set(geometry)!={'n_control_points','quaternion_norm_tolerance'}:
        raise ValueError('Geometry cần n_control_points và quaternion_norm_tolerance.')
    count=geometry['n_control_points'];tolerance=geometry['quaternion_norm_tolerance']
    if isinstance(count,bool) or not isinstance(count,int) or count<3:
        raise ValueError('Geometry n_control_points cần số nguyên >=3.')
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Quaternion norm tolerance không hợp lệ.')
    return job


def auxiliary_inventory(product, *, selection_time_gps):
    """Aux Spec §3.2.1: latest validity <= job start, then latest generation."""
    selected = []
    job_time = Time(selection_time_gps, format='gps').utc.datetime
    for kind, version in (('INS',{'3.16'}),('PP1',{'3.15','3.16'}),('CAL',{'2.10'})):
        candidates = []
        for folder in Path(product).glob(f'auxiliary/S1A_AUX_{kind}_*.SAFE'):
            match = re.fullmatch(rf'S1A_AUX_{kind}_V(\d{{8}}T\d{{6}})_G(\d{{8}}T\d{{6}})\.SAFE',folder.name)
            if not match:
                raise ValueError(f'Tên AUX không hợp lệ: {folder.name}')
            valid, generated = [datetime.strptime(v,'%Y%m%dT%H%M%S') for v in match.groups()]
            if valid <= job_time:
                candidates.append((valid,generated,folder))
        if not candidates:
            raise ValueError(f'Không có AUX_{kind} hợp lệ ở job start.')
        valid, generated, folder = max(candidates)
        paths = list(folder.glob(f'data/s1a-aux-{kind.lower()}.xml'))
        if len(paths)!=1:
            raise ValueError(f'AUX_{kind} cần đúng một payload.')
        root=ET.parse(paths[0]).getroot()
        if root.get('schemaVersion') not in version:
            raise ValueError(f'AUX_{kind} payload version chưa được hỗ trợ.')
        schema_path=folder/f'support/s1-aux-{kind.lower()}.xsd'
        schema=None
        if schema_path.is_file():
            schema={**fingerprint(schema_path),'version':ET.parse(schema_path).getroot().get('version')}
        selected.append({**fingerprint(paths[0]),'kind':kind,'validity_start_utc':valid.isoformat(),
                         'generation_utc':generated.isoformat(),'payload_version':root.get('schemaVersion'),
                         'attached_schema':schema,'schema_version_matches_payload':
                             None if schema is None else schema['version']==root.get('schemaVersion'),
                         'validation_contract':'supported payload version and required S4/VV nodes; attached XSD not authoritative',
                         'selection_time_utc':job_time.isoformat(),'selection_rule':'Aux Spec 3/16 §3.2.1'})
    return selected


class EchoScope:
    """Vue header-only của một miền echo liên tục; packet indices giữ nguyên cho decoder."""
    def __init__(self, level0, pieces, *, scope_id, global_start_line, split_reasons):
        self.original = level0
        self.filename = level0.filename
        self.pieces = pieces
        self.scope_id = scope_id
        self.global_start_line = global_start_line
        self.split_reasons = split_reasons
        self.acquisition_chunks = [piece["virtual_chunk"] for piece in pieces]

    @property
    def ephemeris(self):
        return self.original.ephemeris

    def get_acquisition_chunk_metadata(self, chunk):
        return next(p["metadata"] for p in self.pieces if p["virtual_chunk"] == chunk)

    def inventory(self):
        times = np.concatenate([p["times"] for p in self.pieces])
        first = self.pieces[0]["metadata"].iloc[0]
        return {"scope_id": self.scope_id, "global_start_line": self.global_start_line,
            "split_reasons": self.split_reasons, "line_times_s": times,
            "constant_headers": {field: (first[field].value if hasattr(first[field],"value") else
                first[field].item() if isinstance(first[field],np.generic) else first[field]) for field in
                ("PRI","Range Decimation","Tx Pulse Length","Tx Ramp Rate","Tx Pulse Start Frequency","Rank",
                 "Rx Gain","Instrument Configuration ID","Number of Quads")},
            "pri_s": float(first["PRI"]),
            "sample_rate_hz": sentinel1decoder.utilities.range_dec_to_sample_rate(first["Range Decimation"]),
            "chunks": [{"chunk": p["virtual_chunk"], "source_chunk": p["source_chunk"],
                "first_packet": int(p["metadata"].index[0]), "last_packet": int(p["metadata"].index[-1]),
                "lines": len(p["metadata"]), "range_samples": 2*int(p["metadata"]["Number of Quads"].iloc[0]),
                "swst_s": float(p["metadata"]["SWST"].iloc[0])} for p in self.pieces]}


def header_inventory(level0, *, continuity_tolerance_s):
    """Chia header theo PRI/config/gap trước decode; không nối vượt ranh scope."""
    if not np.isfinite(continuity_tolerance_s) or continuity_tolerance_s < 0:
        raise ValueError('Continuity tolerance không hợp lệ.')
    packets = level0.packet_metadata
    break_counts = set()
    packet_gaps = []
    for field, modulus in (('Packet Sequence Count', 16384), ('Space Packet Count', 2**32)):
        counts = packets[field].to_numpy(np.int64)
        delta = np.diff(counts) % modulus
        if np.any((delta == 0) | (delta > modulus//2)):
            raise ValueError(f'Packet lặp hoặc đảo thứ tự ở {field}.')
        for index in np.flatnonzero(delta != 1):
            count = int(packets['Space Packet Count'].iloc[index+1])
            break_counts.add(count)
            packet_gaps.append({'field': field, 'next_space_packet_count': count, 'missing': int(delta[index]-1)})
    scope_fields = ('PRI', 'Range Decimation', 'Tx Pulse Length', 'Tx Ramp Rate',
        'Tx Pulse Start Frequency', 'Rank', 'Rx Gain', 'Instrument Configuration ID', 'Number of Quads')
    piece_fields = (*scope_fields, 'SWST', 'SWL')
    scopes = []
    current = []
    current_reasons = []
    offset = 0
    previous_time = previous_key = previous_pri = None
    virtual = max(level0.acquisition_chunks)+1
    def flush():
        nonlocal current, offset
        if current:
            scopes.append(EchoScope(level0, current, scope_id=f'scope-{len(scopes)}',
                global_start_line=offset, split_reasons=current_reasons.copy()))
            offset += sum(len(p['metadata']) for p in current)
            current = []
    for chunk in level0.acquisition_chunks:
        metadata = level0.get_acquisition_chunk_metadata(chunk)
        if not (metadata['Signal Type'] == SignalType.ECHO).any():
            continue
        if (not (metadata['Signal Type'] == SignalType.ECHO).all() or metadata['Error Flag'].any()
                or not metadata.index.is_unique or not metadata.index.is_monotonic_increasing):
            raise ValueError('Echo header lỗi, trộn signal hoặc packet lặp/đảo thứ tự.')
        if (not (metadata['Swath Number'] == 3).all()
                or not (metadata['Rx Channel ID'] == RxChannelId.RXV_POL_CHANNEL).all()
                or not metadata['Polarisation'].astype(str).str.startswith('Tx V').all()):
            raise ValueError('Job chỉ hỗ trợ S4/VV.')
        numeric=metadata[[field for field in piece_fields if field!='Range Decimation']].to_numpy(float)
        if not np.isfinite(numeric).all():raise ValueError('Echo numeric header không hữu hạn.')
        eta = metadata['Coarse Time'].to_numpy(float)+metadata['Fine Time'].to_numpy(float)
        pri = metadata['PRI'].to_numpy(float)
        if (not np.isfinite(eta).all() or not np.isfinite(pri).all() or np.any(pri <= 0)
                or np.any(np.diff(eta) <= 0) or continuity_tolerance_s >= .5*min(pri)):
            raise ValueError('Echo time/PRI/tolerance không hợp lệ.')
        changes = np.zeros(len(metadata), dtype=bool)
        changes[0] = True
        for field in piece_fields:
            values = metadata[field].to_numpy()
            changes[1:] |= values[1:] != values[:-1]
        changes[1:] |= np.abs(np.diff(eta)-pri[:-1]) > continuity_tolerance_s
        changes |= metadata['Space Packet Count'].isin(break_counts).to_numpy()
        edges = np.r_[np.flatnonzero(changes), len(metadata)]
        for piece_index, (lo, hi) in enumerate(zip(edges, edges[1:])):
            part = metadata.iloc[lo:hi]
            first = part.iloc[0]
            if first['Range Decimation'].value != 5:
                raise ValueError('S4 adapter hỗ trợ PDU RGDEC5; không thay sample-rate của filter khác.')
            swl = round(float(first['SWL'])*sentinel1decoder.constants.F_REF)
            if 2*int(first['Number of Quads']) != rgdec5_stage3_rx_samples(swl):
                raise ValueError('Số mẫu header khác PDU Stage-3 RGDEC5.')
            key = tuple(first[f] for f in scope_fields)
            reasons = []
            if previous_key is not None:
                reasons += [f'changed:{f}' for f, a, b in zip(scope_fields, previous_key, key) if a != b]
                if eta[lo] <= previous_time:
                    raise ValueError('Echo chunks bị trùng/đảo thứ tự thời gian.')
                if abs(eta[lo]-previous_time-previous_pri) > continuity_tolerance_s:
                    reasons.append('acquisition_gap')
                if int(first['Space Packet Count']) in break_counts:
                    reasons.append('packet_count_gap')
            if reasons:
                flush()
                current_reasons = reasons
            identity = chunk if piece_index == 0 else virtual
            if piece_index:
                virtual += 1
            current.append({'virtual_chunk': int(identity), 'source_chunk': int(chunk),
                            'metadata': part, 'times': eta[lo:hi]})
            previous_time, previous_key, previous_pri = eta[hi-1], key, float(first['PRI'])
    flush()
    if not scopes:
        raise ValueError('Không có echo.')
    summaries = [scope.inventory() for scope in scopes]
    return {'line_times_s': np.concatenate([r['line_times_s'] for r in summaries]),
        'pri_s': summaries[0]['pri_s'] if len(scopes) == 1 else None,
        'sample_rate_hz': summaries[0]['sample_rate_hz'] if len(scopes) == 1 else None,
        'chunks': [chunk for r in summaries for chunk in r['chunks']],
        'scopes': [{k: v for k, v in r.items() if k != 'line_times_s'} for r in summaries],
        'packet_gaps': packet_gaps, '_scope_views': scopes}


def resolve_layout(job, inventory, aux, *, requested_starts_s=None, allow_empty=False):
    layout=job['layout']
    allowed={'policy','block_size_lines','num_range_blocks','range_block_size_samples','nominal_spacing_s',
             'tail_stop_convention','mapping_tolerance_s','continuity_tolerance_s','missing_support_policy',
             'range_roi_start','range_roi_stop','range_start_rounding'}
    if set(layout)-allowed:raise ValueError('Layout có key không hỗ trợ.')
    required=allowed-{'missing_support_policy','range_roi_start','range_roi_stop','range_start_rounding'}
    if required-set(layout):raise ValueError(f'Thiếu layout keys: {sorted(required-set(layout))}')
    policy=layout['policy'];times=inventory['line_times_s'];prf=1/inventory['pri_s']
    if policy=='l0_support':
        if job['mode']!='research':raise ValueError('l0_support chỉ dùng research, chưa là IPF context.')
        if 'context' in job:raise ValueError('l0_support không nhận context bị bỏ qua.')
        starts=[times[0]];stop=times[-1];current=0
        source='L0 continuous support; research assumption of one internal slice'
    elif policy=='slice_context':
        context=job.get('context',{})
        if set(context)!={'slice_start_times_gps_s','last_slice_stop_time_gps_s','current_slice_index','source'}:
            raise ValueError('Context cần starts/last stop/current index/source từ L0 hoặc job.')
        if not isinstance(context['source'],str) or not context['source'].strip():
            raise ValueError('Context source phải tường minh.')
        starts=context['slice_start_times_gps_s'];stop=context['last_slice_stop_time_gps_s']
        current=context['current_slice_index'];source=context['source']
    else:raise ValueError('Layout policy phải là slice_context hoặc research l0_support.')
    if requested_starts_s is not None:
        requested=np.asarray(requested_starts_s,dtype=float)
    elif allow_empty and len(times)<layout['block_size_lines']:
        requested=np.array([times[0]])
    else:
        requested=context_starts(slice_start_times_s=starts,last_slice_stop_time_s=stop,current_slice_index=current,
            block_size_lines=layout['block_size_lines'],prf_hz=prf,nominal_spacing_s=layout['nominal_spacing_s'],
            tail_stop_convention=layout['tail_stop_convention'])
    windows=map_starts(requested,times,block_size_lines=layout['block_size_lines'],prf_hz=prf,
        mapping_tolerance_s=layout['mapping_tolerance_s'],continuity_tolerance_s=layout['continuity_tolerance_s'],
        missing_support_policy='report' if allow_empty else layout.get('missing_support_policy','error'))
    est=job['estimator']
    if set(est)-{'unwrap_fft_length','polynomial_degree','outlier_sigma','max_fit_iterations',
                 'unwrap_weighting','fit_weighting','accc_range_weighting','fallback_policy',
                 'predefined_t0_s','predefined_range_support_s','predefined_time_support_s'}:
        raise ValueError('Estimator có key không hỗ trợ.')
    predefined={}
    if aux['dc_method']=='Predefined':
        if aux.get('predefined_coefficients') is None:
            raise ValueError('AUX thiếu predefined DC polynomial.')
        predefined['predefined_coefficients']=tuple(aux['predefined_coefficients'])
    elif any(key.startswith('predefined_') for key in est):
        raise ValueError('Predefined config không được bỏ qua khi AUX chọn method khác.')
    config=estimation.Config(azimuth_block_size_lines=layout['block_size_lines'],
        num_range_blocks=layout['num_range_blocks'],range_block_size_samples=layout['range_block_size_samples'],range_roi_start=layout.get('range_roi_start',0),
        range_roi_stop=layout.get('range_roi_stop'),range_start_rounding=layout.get('range_start_rounding','floor'),
        azimuth_placement='custom',rms_threshold_hz=aux['rms_threshold_hz'],dc_method=aux['dc_method'],**predefined,**est)
    if not allow_empty and not any(w['available'] for w in windows):raise ValueError('Không có DCE window đủ support.')
    return config,windows,{'source':source,'policy':policy,'timeline':'acquisition GPS seconds',
        'nominal_spacing_s':layout['nominal_spacing_s'],'spacing_evidence':'inferred; metadata cohort',
        'tail_stop_convention':layout['tail_stop_convention'],'config':asdict(config)}


def resolve_processing_scopes(level0, headers, job, aux):
    views=headers.get('_scope_views')
    inputs=[(view,view.inventory()) for view in views] if views is not None else [(level0,headers)]
    requested=None
    if job['layout']['policy']=='slice_context':
        context=job.get('context',{})
        if set(context)!={'slice_start_times_gps_s','last_slice_stop_time_gps_s','current_slice_index','source'}:
            raise ValueError('Slice context thiếu trường bắt buộc.')
        stop=context['last_slice_stop_time_gps_s'];tolerance=job['layout']['mapping_tolerance_s']
        matching=[header for _,header in inputs if header['line_times_s'][0]-tolerance<=stop<=header['line_times_s'][-1]+tolerance]
        tail_header=matching[0] if matching else inputs[-1][1]
        requested=context_starts(slice_start_times_s=context['slice_start_times_gps_s'],
            last_slice_stop_time_s=context['last_slice_stop_time_gps_s'],current_slice_index=context['current_slice_index'],
            block_size_lines=job['layout']['block_size_lines'],prf_hz=1/tail_header['pri_s'],
            nominal_spacing_s=job['layout']['nominal_spacing_s'],tail_stop_convention=job['layout']['tail_stop_convention'])
    plans=[]
    for index,(view,header) in enumerate(inputs):
        config,windows,layout=resolve_layout(job,header,aux,requested_starts_s=requested,allow_empty=True)
        estimation.Estimator(1/header['pri_s'],config)
        plans.append({'level0':view,'headers':header,'config':config,'windows':windows,'layout':layout,
            'scope_id':header.get('scope_id',f'scope-{index}'),'global_start_line':header.get('global_start_line',0)})
    # For a supplied context, a window belongs to the scope containing its start.
    # Failure to span a boundary stays unavailable; do not re-anchor it in the next scope.
    if requested is not None:
        for plan in plans:
            lo,hi=plan['headers']['line_times_s'][[0,-1]]
            tolerance=job['layout']['mapping_tolerance_s']
            plan['windows']=[w for w in plan['windows'] if lo-tolerance<=w['requested_start_time_s']<=hi+tolerance]
        seen={w['requested_start_time_s'] for p in plans for w in p['windows']}
        plans[0]['windows'].extend({'requested_start_time_s':float(t),'start_line':None,'stop_line_exclusive':None,
            'start_time_s':None,'stop_time_s':None,'center_time_s':None,'available':False,
            'unavailable_reason':'acquisition_gap_or_outside_input','mapping_error_s':None,
            'window_id':f'{float(t).hex()}:missing'} for t in requested if t not in seen)
    for plan in plans:
        for window in plan['windows']:
            window['scope_id']=plan['scope_id']
            window['global_start_line']=None if not window['available'] else window['start_line']+plan['global_start_line']
    windows=[w for p in plans for w in p['windows']]
    if not any(w['available'] for w in windows):
        raise ValueError('Không có DCE window đủ support sau segmentation.')
    return plans


def range_support(level0, aux, config, *, internal_delay_s, minimum_far_range_discard_samples):
    """Validate range ROI from headers and measured delay before image decode."""
    from ..azimuth_pre_processing.range.compression import slc_support
    metadata=[(chunk,level0.get_acquisition_chunk_metadata(chunk).iloc[0]) for chunk in level0.acquisition_chunks]
    first=metadata[0][1]
    pri=float(first['PRI'])
    rate=sentinel1decoder.utilities.range_dec_to_sample_rate(first['Range Decimation'])
    suppressed=320./(8*sentinel1decoder.constants.F_REF)
    origins=[float(row['Rank'])*pri+float(row['SWST'])+suppressed for _,row in metadata]
    records=[];grids=[]
    for (chunk,row),origin in zip(metadata,origins):
        offset=(origin-min(origins))*rate
        shift=(round(offset)-offset)/rate
        support=slc_support(2*int(row['Number of Quads']),sample_rate_hz=rate,
            pulse_length_s=float(first['Tx Pulse Length']),slant_range_origin_s=origin,swst_bias_s=aux['swst_bias_s'],
            range_time_shift_s=shift,internal_delay_s=internal_delay_s,
            minimum_far_range_discard_samples=minimum_far_range_discard_samples)
        grids.append(support['range_times_s'])
        records.append({'chunk':int(chunk),'range_start_s':float(grids[-1][0]),'range_last_s':float(grids[-1][-1]),
            'range_samples':len(grids[-1]),'fractional_shift_s':shift,
            'far_discard_samples':support['far_throwaway'],'required_far_discard_samples':support['required_far_discard']})
    t0=max(grid[0] for grid in grids)
    end=min(grid[-1] for grid in grids)+1/rate
    roi=int(round((end-t0)*rate))
    if roi<2:raise ValueError('Không có giao range support giữa các chunk.')
    resolved=replace(config,range_roi_stop=roi if config.range_roi_stop is None else config.range_roi_stop)
    blocks=estimation.build_range_blocks(roi,t0+np.arange(roi)/rate,resolved)
    if len(blocks)!=resolved.num_range_blocks or len({b.start_sample for b in blocks})!=len(blocks):
        raise ValueError('Range ROI không đủ để đặt các range block phân biệt theo cấu hình.')
    return resolved,float(t0),{'available_range_samples':roi,'roi_samples':resolved.range_roi_stop-resolved.range_roi_start,
        'range_roi_start':resolved.range_roi_start,'range_roi_stop':resolved.range_roi_stop,'range_origin_s':float(t0),'sample_rate_hz':float(rate),
        'internal_delay_s':float(internal_delay_s),'chunks':records,'_grids':grids}


def real_values(array):
    """Real diagnostics as JSON values; nonfinite/unavailable is null, never zero."""
    if array is None or np.asarray(array).size==0 or not np.isfinite(array).any():
        return None
    return [float(v) if np.isfinite(v) else None for v in np.asarray(array)]


def write_estimates(output, estimates, provenance):
    values=real_values
    rows=[]
    for record in estimates:
        rows.append({'processing_scope_id':record.processing_scope_id,'acquisition_window_id':record.acquisition_window_id,
            'start_line':record.block.start_line,'stop_line_exclusive':record.block.stop_line,
            'acquisition_start_time_gps_s':record.block.start_time_s,
            'acquisition_stop_time_gps_s':record.block.stop_time_s,
            'acquisition_center_time_gps_s':record.block.azimuth_time_s,
            't0_s':record.t0_s,'range_times_s':record.range_times_s.tolist(),
            'data_dc_polynomial':values(record.data_dc_polynomial),
            'geometry_dc_polynomial':values(record.geometry_dc_polynomial),
            'fine_baseband_hz':values(record.fine_baseband_hz),'fine_unwrapped_hz':values(record.fine_unwrapped_hz),
            'fine_absolute_hz':values(record.fine_absolute_hz),'selected_dc_hz':record.evaluate(record.range_times_s).tolist(),
            'rms_error_hz':float(record.rms_error_hz) if np.isfinite(record.rms_error_hz) else None,
            'rms_above_threshold':record.rms_above_threshold,'ambiguity_number':record.ambiguity_number,
            'selected_source':record.selected_source,'selection_reason':record.selection_reason,
            'selected_dc_polynomial':values(record.selected_dc_polynomial),
            'absolute_ambiguity_resolved':record.absolute_ambiguity_resolved,
            'valid_mask':record.valid_mask.tolist(),'coherence':record.coherence.tolist()})
    Path(output).write_text(json.dumps({**provenance,'estimates':rows},ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def compare_annotations(estimates, path, *, max_time_delta_s):
    """So reference riêng trên miền giao, không đổi kết quả hoặc ghép theo index."""
    if not np.isfinite(max_time_delta_s) or max_time_delta_s<0:
        raise ValueError('Comparison time delta phải hữu hạn, không âm.')
    root=ET.parse(path).getroot()
    offset=float(root.findtext('imageAnnotation/imageInformation/zeroDopMinusAcqTime'))
    if not np.isfinite(offset):raise ValueError('Comparison ZD offset không hữu hạn.')
    refs=root.findall('dopplerCentroid/dcEstimateList/dcEstimate')
    times=np.array([float(Time(r.findtext('azimuthTime'),scale='utc').gps)-offset for r in refs])
    if not len(refs) or not np.isfinite(times).all():raise ValueError('Reference DCE rỗng hoặc time không hợp lệ.')
    rows=[]
    for estimate in estimates:
        index=int(np.argmin(abs(times-estimate.block.azimuth_time_s)))
        delta=float(times[index]-estimate.block.azimuth_time_s)
        row={'start_line':estimate.block.start_line,'reference_record_index':index,'time_delta_s':delta}
        if abs(delta)>max_time_delta_s:
            rows.append({**row,'compared':False,'reason':'outside_time_tolerance'})
            continue
        record=refs[index]
        points=record.findall('fineDceList/fineDce')
        tau=np.array([float(p.findtext('slantRangeTime')) for p in points])
        if tau.size<2 or not np.isfinite(tau).all() or np.any(np.diff(tau)<=0):
            raise ValueError('Reference range support không hợp lệ.')
        actual=estimate.range_times_s
        valid=(actual>=tau[0])&(actual<=tau[-1])
        if not valid.any():
            rows.append({**row,'compared':False,'reason':'no_range_overlap'})
            continue
        coefficients=np.fromstring(record.findtext('dataDcPolynomial'),sep=' ')
        reference=np.polynomial.polynomial.polyval(actual[valid]-float(record.findtext('t0')),coefficients)
        difference=estimate.evaluate(actual[valid])-reference
        rows.append({**row,'compared':True,'range_times_s':actual[valid].tolist(),
                     'selected_vs_reference_data_polynomial_rmse_hz':float(np.sqrt(np.mean(difference**2)))})
    return {'annotation':fingerprint(path),'zero_dop_minus_acq_time_s':offset,
            'max_time_delta_s':max_time_delta_s,'records':rows,'used_for_processing':False}


def parameter_inventory(job, config, headers):
    """Source/unit/state cho config thực dùng; L1 không tham gia resolver này."""
    units = {'azimuth_block_size_lines':'lines','block_size_lines':'lines','num_range_blocks':'count',
        'range_block_size_samples':'samples','range_roi_start':'samples','range_roi_stop':'samples',
        'azimuth_spacing_lines':'lines','nominal_spacing_s':'s','mapping_tolerance_s':'s','continuity_tolerance_s':'s','azimuth_time_offset_s':'s',
        'current_slice_index':'index (0-based)',
        'unwrap_fft_length':'samples','polynomial_degree':'degree','outlier_sigma':'sigma multiplier',
        'max_fit_iterations':'count','rms_threshold_hz':'Hz','predefined_coefficients':'Hz * s^(-k), ascending k',
        'predefined_t0_s':'s (two-way range)','predefined_range_support_s':'s (two-way range)',
        'predefined_time_support_s':'GPS s','n_control_points':'count','quaternion_norm_tolerance':'dimensionless',
        'oversampling_factor':'factor','denominator_relative_floor':'dimensionless',
        'antenna_axis_body':'unit vector in SAR body frame','reference_range_m':'m',
        'update_interval_s':'s','boresight_update_interval_s':'s','minimum_magnitude':'AUX_CAL complex LUT magnitude',
        'height_m':'m (ellipsoid)','range_tolerance_m':'m','height_tolerance_m':'m','doppler_tolerance_mps':'m/s',
        'max_nfev':'count','predefined_i_bias':'DN','predefined_q_bias':'DN','predefined_gain_imbalance':'dimensionless',
        'predefined_quadrature_rad':'rad','slice_start_times_gps_s':'GPS s','last_slice_stop_time_gps_s':'GPS s'}
    layout_keys={'azimuth_block_size_lines':'block_size_lines','num_range_blocks':'num_range_blocks',
        'range_block_size_samples':'range_block_size_samples','range_roi_start':'range_roi_start',
        'range_roi_stop':'range_roi_stop','range_start_rounding':'range_start_rounding'}
    rows=[]
    for key,value in asdict(config).items():
        if key=='rms_threshold_hz':source='AUX_PP1 SM_SL1__1/dcProcParams/dcRmsErrorThreshold'
        elif key=='dc_method':source='AUX_PP1 SM_SL1__1/dcProcParams/dcMethod'
        elif key=='predefined_coefficients':source='AUX_PP1 SM_SL1__1/dcProcParams/dcPredefinedCoefficients'
        elif key in layout_keys and layout_keys[key] in job['layout']:source=f'job.layout.{layout_keys[key]}'
        elif key in job['estimator']:source=f'job.estimator.{key}'
        elif key=='range_roi_stop':source='L0 range support after compression/delay/guard'
        elif key=='azimuth_placement':source='resolved acquisition windows, job.layout policy'
        else:source='Config default; recorded implementation choice'
        rows.append({'name':f'estimator.{key}','value':value,'unit':units.get(key,'enum'),
                     'source':source,'state':'not_applicable' if value is None else 'resolved'})
    def add_job(prefix,value):
        if isinstance(value,dict):
            for key,item in value.items():add_job(f'{prefix}.{key}',item)
        else:
            key=prefix.rsplit('.',1)[-1]
            unit=units.get(key,'boolean' if isinstance(value,bool) else 'text/enum')
            rows.append({'name':prefix,'value':value,'unit':unit,'source':prefix,
                         'state':'not_applicable' if value is None else 'configured'})
    for group in ('layout','geometry','calibration','context','corrections'):
        if group in job:add_job(f'job.{group}',job[group])
    header_units={'PRI':'s','Range Decimation':'PDU code','Tx Pulse Length':'s',
        'Tx Ramp Rate':'Hz/s','Tx Pulse Start Frequency':'Hz','Rank':'PRI count','Rx Gain':'dB',
        'Instrument Configuration ID':'identifier','Number of Quads':'quadrature pairs'}
    for key,value in headers.get('constant_headers',{}).items():
        rows.append({'name':f'header.{key}','value':value,'unit':header_units[key],
            'source':f'L0 header: {key}','state':'resolved'})
    return rows
