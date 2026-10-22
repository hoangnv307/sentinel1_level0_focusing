"""Thống kê DCE, slice, product và extent datatake từ metadata; giữ mọi phản ví dụ."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from xml.etree import ElementTree as ET

from sentinel1_processing.utils.copernicus import product_info, write_json
from check_dce_azimuth_layout import candidate_starts, compare_starts, shared_windows


def instant(value):
    return datetime.fromisoformat(value.rstrip('Z'))


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'min': values[0], 'median': median(values), 'max': values[-1]}


def read_annotation(path, *, block_size_lines, nominal_spacing_s, tolerance_s):
    payload = path.read_bytes()
    root = ET.fromstring(payload)
    header = root.find('adsHeader')
    image = root.find('imageAnnotation/imageInformation')
    downlinks = root.findall('generalAnnotation/downlinkInformationList/downlinkInformation')
    if header is None or image is None or len(downlinks) != 1:
        raise ValueError('Cần header, image information và đúng một downlink record.')
    dl = downlinks[0]
    pri = float(dl.findtext('downlinkValues/pri'))
    if pri <= 0:
        raise ValueError('PRI phải dương.')
    first, last = (instant(dl.findtext(k)) for k in ('firstLineSensingTime', 'lastLineSensingTime'))
    span = (last-first).total_seconds()
    offset = float(image.findtext('zeroDopMinusAcqTime'))
    slices = [{'number': int(s.findtext('sliceNumber')), 'start': s.findtext('sensingStartTime'),
               'stop': s.findtext('sensingStopTime')} for s in image.findall('sliceList/slice')]
    number = int(image.findtext('sliceNumber'))
    current = next(i for i,s in enumerate(slices) if s['number'] == number)
    has_next = current+1 < len(slices)
    h = (instant(slices[current+1]['start'])-first).total_seconds() if has_next else span
    records = root.findall('dopplerCentroid/dcEstimateList/dcEstimate')
    if not records:
        raise ValueError('Không có DCE record.')
    starts, widths, center_errors, counts, windows, rms, flags = [], [], [], [], [], [], []
    range_widths, range_spreads = [], []
    rate_text=root.findtext('generalAnnotation/productInformation/rangeSamplingRate')
    range_rate=float(rate_text) if rate_text is not None else None
    for record in records:
        start, stop, center = [instant(record.findtext(k)) for k in
            ('fineDceAzimuthStartTime','fineDceAzimuthStopTime','azimuthTime')]
        starts.append((start-first).total_seconds()-offset)
        widths.append((stop-start).total_seconds()/pri)
        center_errors.append((center-start).total_seconds()-(stop-start).total_seconds()/2)
        fine=record.findall('fineDceList/fineDce')
        counts.append(len(fine))
        if fine and range_rate is not None:
            tau=[float(p.findtext('slantRangeTime')) for p in fine]
            range_widths.append(2*(tau[0]-float(record.findtext('t0')))*range_rate)
            steps=[(b-a)*range_rate for a,b in zip(tau,tau[1:])]
            if steps:range_spreads.append(max(steps)-min(steps))
        value=record.findtext('dataDcRmsError')
        rms.append(float(value) if value is not None else None)
        flag=record.findtext('dataDcRmsErrorAboveThreshold')
        if flag not in ('true','false',None):raise ValueError('Cờ RMS DCE không hợp lệ.')
        flags.append(None if flag is None else flag=='true')
        windows.append({key:(value-timedelta(seconds=offset)).isoformat()
                        for key,value in (('start',start),('stop',stop),('center',center))})
    duration = block_size_lines*pri
    regular = candidate_starts(h, duration, nominal_spacing_s=nominal_spacing_s)
    end_points = {'last_line': span-duration,
                  'after_last_line': (round(span/pri)+1-block_size_lines)*pri,
                  'end_start': span}
    predictions = {}
    for convention, endpoint in end_points.items():
        predicted = sorted(set([*regular, h if has_next else endpoint]))
        comparison = compare_starts(predicted, starts, tolerance_s=tolerance_s)
        comparison['anchored_start_residual_s'] = (max(abs((p-predicted[0])-(s-starts[0]))
            for p,s in zip(predicted,starts)) if len(predicted)==len(starts) else None)
        combined=None
        if len(predicted)==len(starts):
            combined=max(abs(error) for p,s,w,ce in zip(predicted,starts,widths,center_errors)
                for error in (p-s,p+duration-s-w*pri,p+duration/2-s-w*pri/2-ce))
        comparison['max_start_stop_center_error_s']=combined
        comparison['matches'] = combined is not None and combined<=tolerance_s
        predictions[convention] = comparison
    manifest = path.parent.parent/'manifest.safe' if path.parent.name=='annotation' else path.parent/'manifest.safe'
    versions = []
    if manifest.exists():
        for node in ET.parse(manifest).getroot().iter():
            if node.tag.split('}')[-1]=='software' and node.get('name')=='Sentinel-1 IPF':
                versions.append(node.get('version'))
    output_first, output_last = [instant(image.findtext(k)) for k in ('productFirstLineUtcTime','productLastLineUtcTime')]
    output_span = (output_last-output_first).total_seconds()
    return {'file': str(path.resolve()), 'sha256': hashlib.sha256(payload).hexdigest(),
        'mission': header.findtext('missionId'), 'swath': header.findtext('swath'),
        'polarisation': header.findtext('polarisation'), 'orbit': header.findtext('absoluteOrbitNumber'),
        'datatake': header.findtext('missionDataTakeId'), 'ipf': ','.join(sorted(set(versions))) or 'unknown',
        'slice_number': number, 'slice_count': len(slices), 'slices': slices,
        'position': 'single' if len(slices)==1 else 'first' if current==0 else 'last' if not has_next else 'middle',
        'pri_s': pri, 'downlink_span_s': span, 'schedule_span_s': h,
        'output_first': output_first.isoformat(), 'output_last': output_last.isoformat(),
        'output_span_s': output_span, 'output_line_interval_s': float(image.findtext('azimuthTimeInterval')),
        'output_exclusive_duration_s':output_span+float(image.findtext('azimuthTimeInterval')),
        'output_azimuth_extent_m': (int(image.findtext('numberOfLines'))-1)*float(image.findtext('azimuthPixelSpacing')),
        'dc_records': len(records), 'block_widths_lines': widths, 'range_counts': counts,
        'range_width_candidate_samples':range_widths,'range_center_step_spread_samples':range_spreads,
        'configured_dc_method':root.findtext('imageAnnotation/processingInformation/dcMethod'),
        'data_dc_rms_hz':rms,'rms_above_threshold_flags':flags,
        'acquisition_windows':windows,
        'observed_starts_s': starts, 'first_start_offset_lines': starts[0]/pri,
        'tail_stop_offset_lines': ((starts[-1]+widths[-1]*pri-span)/pri) if not has_next else None,
        'hypotheses': predictions}


def catalogue_summary(directory, *, min_year=2020):
    products, sources = {}, []
    for path in sorted(Path(directory).glob('*.json')):
        value = json.loads(path.read_text())
        if 'filter' not in value or 'products' not in value:
            continue
        sources.append({'path': str(path.resolve()), 'truncated': value.get('truncated', False),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for product in value['products']:
            products[product['Id']] = product
    source_count = len(products)
    products = {uid:p for uid,p in products.items() if int(product_info(p['Name'])['start'][:4])>=min_year}
    groups = defaultdict(list)
    durations = []
    acquisition_keys = set()
    for p in products.values():
        i = product_info(p['Name'])
        a,b = [datetime.strptime(i[k],'%Y%m%dT%H%M%S') for k in ('start','stop')]
        durations.append((b-a).total_seconds())
        groups[(i['mission'],i['swath'],i['datatake'])].append((a,b))
        acquisition_keys.add(tuple(i[k] for k in ('mission','swath','orbit','datatake','pol','start','stop')))
    return {'sources': sources, 'all_catalogue_products': source_count, 'products': len(products),
        'min_year':min_year, 'unique_acquisitions': len(acquisition_keys),
        'mission_counts': dict(Counter(product_info(p['Name'])['mission'] for p in products.values())),
        'swath_counts': dict(Counter(product_info(p['Name'])['swath'] for p in products.values())),
        'product_duration_name_seconds': distribution(durations),
        'product_duration_histogram_seconds': dict(sorted(Counter(durations).items())),
        'datatake_swath_groups': len(groups),
        'observed_product_envelope_s': distribution((max(b for a,b in g)-min(a for a,b in g)).total_seconds() for g in groups.values()),
        'products_per_group': dict(sorted(Counter(len(g) for g in groups.values()).items()))}


def summarize(rows, *, time_tolerance_s=5e-6):
    by_version = defaultdict(list)
    by_position = defaultdict(list)
    groups = defaultdict(list)
    for row in rows:
        by_version[row['ipf']].append(row)
        by_position[row['position']].append(row['output_span_s'])
        key=tuple(row[k] for k in ('mission','swath','datatake','polarisation'))+(row['slices'][0]['start'],)
        groups[key].append(row)
    group_reports = []
    for key, group in sorted(groups.items()):
        context = group[0]['slices']
        compatible = all(row['slices']==context for row in group)
        expected = {s['number'] for s in context}
        present = {row['slice_number'] for row in group}
        ordered = sorted(group,key=lambda r:r['slice_number'])
        group_reports.append({'identity': key, 'orbits':sorted({row['orbit'] for row in group}),
            'context_consistent': compatible,
            'missing_slices': sorted(expected-present), 'duplicate_slice_numbers': len(present)!=len(group),
            'declared_slices': len(context), 'metadata_extent_s': (instant(context[-1]['stop'])-instant(context[0]['start'])).total_seconds(),
            'slice_start_steps_s': [(instant(b['start'])-instant(a['start'])).total_seconds() for a,b in zip(context,context[1:])],
            'slice_overlaps_s': [(instant(a['stop'])-instant(b['start'])).total_seconds() for a,b in zip(context,context[1:])],
            'observed_output_boundary_gap_s': [(instant(b['output_first'])-instant(a['output_last'])).total_seconds()
                for a,b in zip(ordered,ordered[1:]) if b['slice_number']==a['slice_number']+1]})
    normalized=[{**r,'absolute_orbit':r['orbit'],'datatake_id':r['datatake'],'slice_context':r['slices']}
                for r in rows]
    shared_error=None
    try:shared=shared_windows(normalized,time_tolerance_s=time_tolerance_s)
    except ValueError as error:shared=[];shared_error=str(error)
    eligible=[r for r in rows if r['dc_records']>=2]
    lower=max((r['schedule_span_s']/r['dc_records'] for r in eligible),default=None)
    upper=min((r['schedule_span_s']/(r['dc_records']-1) for r in eligible if r['dc_records']>2),default=None)
    tail_groups=defaultdict(list)
    for row in rows:
        value=row['tail_stop_offset_lines']
        if value is not None and abs(value-round(value))*row['pri_s']<=time_tolerance_s:
            tail_groups[tuple(row[k] for k in ('mission','swath','polarisation','ipf','pri_s','downlink_span_s'))].append(row)
    conflicts=[]
    for key,group in tail_groups.items():
        labels={round(r['tail_stop_offset_lines']) for r in group}
        if 0 in labels and 1 in labels:
            conflicts.append({'mission_swath_pol_ipf_pri_span':key,
                'last_line_example':next(r['file'] for r in group if round(r['tail_stop_offset_lines'])==0),
                'after_last_line_example':next(r['file'] for r in group if round(r['tail_stop_offset_lines'])==1)})
    return {'annotations': len(rows), 'mission_counts': dict(Counter(r['mission'] for r in rows)),
        'swath_counts': dict(Counter(r['swath'] for r in rows)),
        'range_count_histogram': dict(sorted(Counter(n for r in rows for n in r['range_counts']).items())),
        'range_width_candidate_samples':distribution(v for r in rows for v in r['range_width_candidate_samples']),
        'range_width_assumption':'first range block begins at polynomial t0; width=2*(first center-t0)*Fs',
        'range_center_step_spread_samples':distribution(v for r in rows for v in r['range_center_step_spread_samples']),
        'configured_dc_method_counts':dict(Counter(r['configured_dc_method'] for r in rows)),
        'rms_flags':{'above_threshold':sum(flag is True for r in rows for flag in r['rms_above_threshold_flags']),
                     'not_above_threshold':sum(flag is False for r in rows for flag in r['rms_above_threshold_flags']),
                     'missing':sum(flag is None for r in rows for flag in r['rms_above_threshold_flags'])},
        'data_dc_rms_hz':distribution(v for r in rows for v in r['data_dc_rms_hz'] if v is not None),
        'block_widths_lines': distribution(w for r in rows for w in r['block_widths_lines']),
        'output_span_s_by_position': {k: distribution(v) for k,v in sorted(by_position.items())},
        'spacing_constraint':{'assumption':'K=number of annotated DCE records minus one; max(1,floor(H/S))',
            'samples':len(eligible),'lower_open_s':lower,'upper_closed_s':upper,
            'feasible':lower is not None and (upper is None or lower<upper)},
        'single_scene_echo_grids':{mission:{
            'rounded_span_seconds':dict(sorted(Counter(round(r['downlink_span_s']) for r in rows
                if r['mission']==mission and r['position']=='single').items())),
            'distance_to_3s_grid_s':distribution(abs(r['downlink_span_s']-3*round(r['downlink_span_s']/3))
                for r in rows if r['mission']==mission and r['position']=='single'),
            'distance_to_5s_grid_s':distribution(abs(r['downlink_span_s']-5*round(r['downlink_span_s']/5))
                for r in rows if r['mission']==mission and r['position']=='single')}
            for mission in sorted({r['mission'] for r in rows})},
        'by_ipf': {version: {'annotations': len(group),
            'hypothesis_matches': {c: sum(r['hypotheses'][c]['matches'] for r in group)
                                   for c in ('last_line','after_last_line','end_start')},
            'single_or_last_hypothesis_matches': {c: sum(r['hypotheses'][c]['matches'] for r in group
                if r['position'] in ('single','last')) for c in ('last_line','after_last_line','end_start')},
            'nonfinal_annotations':sum(r['position'] in ('first','middle') for r in group),
            'final_tail_offset_lines': distribution(r['tail_stop_offset_lines'] for r in group if r['tail_stop_offset_lines'] is not None)}
            for version,group in sorted(by_version.items())},
        'shared_boundary_windows':shared,'shared_windows_error':shared_error,
        'tail_conflicts_with_identical_visible_parameters':conflicts,
        'datatake_groups': group_reports}


def overview(rows, groups, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9));axes=axes.ravel()
    for position in ('single','first','middle','last'):
        axes[0].hist([r['output_span_s'] for r in rows if r['position']==position],
                     bins=range(0,41),histtype='step',label=position)
    axes[0].axvline(25,color='gray',linestyle='--')
    axes[0].set(xlabel='Product output span (s)',ylabel='Annotations')
    axes[0].legend()
    for mission in sorted({r['mission'] for r in rows}):
        selected=[r for r in rows if r['mission']==mission and r['dc_records']>=3]
        axes[1].scatter([r['schedule_span_s'] for r in selected],
                        [r['observed_starts_s'][1]-r['observed_starts_s'][0] for r in selected],
                        s=12,alpha=.55,label=mission)
    axes[1].set(xlabel='DCE scheduling span H (s)',ylabel='First DCE start spacing (s)')
    axes[1].legend()
    extents=sorted(g['metadata_extent_s'] for g in groups)
    axes[2].step(extents,[(i+1)/len(extents) for i in range(len(extents))],where='post')
    axes[2].set(xlabel='Datatake internal metadata extent (s)',ylabel='Fraction of datatake/swath groups')
    for mission in sorted({r['mission'] for r in rows}):
        axes[3].hist([r['downlink_span_s'] for r in rows if r['mission']==mission and r['position']=='single'],
                     bins=[v+.5 for v in range(40)],histtype='step',label=mission)
    axes[3].set(xlabel='Single-scene internal echo span (s)',ylabel='Annotations')
    axes[3].legend()
    for axis in axes:axis.grid(alpha=.25)
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--catalogue-directory',type=Path)
    parser.add_argument('--output',type=Path,default=Path('output/sentinel-1/copernicus/survey/patterns.json'))
    parser.add_argument('--block-size-lines',type=int,default=6000)
    parser.add_argument('--nominal-spacing-s',type=float,default=10)
    parser.add_argument('--tolerance-s',type=float,default=5e-6)
    parser.add_argument('--min-year',type=int,default=2020)
    parser.add_argument('--figure',type=Path)
    args = parser.parse_args()
    if not args.directory.expanduser().is_dir():parser.error('Directory không tồn tại.')
    if args.block_size_lines<=1 or not all(math.isfinite(v) and v>0 for v in (args.nominal_spacing_s,args.tolerance_s)):
        parser.error('Block size phải >1; spacing/tolerance phải hữu hạn và dương.')
    rows,errors,excluded = [],[],[]
    for path in sorted(args.directory.expanduser().rglob('s1*-s*-slc-*.xml')):
        try:
            year=int(path.name.split('-')[4][:4])
            if year<args.min_year:
                excluded.append(str(path.resolve()))
                continue
            rows.append(read_annotation(path,block_size_lines=args.block_size_lines,
                nominal_spacing_s=args.nominal_spacing_s,tolerance_s=args.tolerance_s))
        except (ValueError,TypeError,StopIteration,ET.ParseError,OSError) as error:
            errors.append({'file':str(path.resolve()),'error':str(error)})
    result = {'config': {'block_size_lines':args.block_size_lines,'nominal_spacing_s':args.nominal_spacing_s,
                        'tolerance_s':args.tolerance_s,'min_year':args.min_year}, 'annotations':rows,
              'excluded_before_min_year':excluded,'errors':errors,
              'summary':summarize(rows,time_tolerance_s=args.tolerance_s)}
    if args.catalogue_directory:
        result['catalogue_summary'] = catalogue_summary(args.catalogue_directory,min_year=args.min_year)
    write_json(args.output,result)
    if args.figure and rows:
        overview(rows,result['summary']['datatake_groups'],args.figure)
    print(json.dumps({'annotations':len(rows),'errors':len(errors),'by_ipf':result['summary']['by_ipf']},ensure_ascii=False))


if __name__=='__main__':
    main()
