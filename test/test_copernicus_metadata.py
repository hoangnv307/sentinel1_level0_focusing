"""Kiểm protocol CDSE bằng mock; không dùng tài khoản hoặc tải dữ liệu thật."""

from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from xml.etree import ElementTree as ET

import requests

from sentinel1_processing.utils import copernicus as cdse


NAME = "S1C_S6_SLC__1SDH_20250330T180412_20250330T180440_001674_002C4C_A098.SAFE"
ANN = "s1c-s6-slc-hh-20250330t180412-20250330t180440-001674-002c4c-001.xml"
UID = "9484cbcc-c1d0-48d8-0de6-01075648c4bd"
MANIFEST = b'<XFDU><software name="Sentinel-1 IPF" version="003.90"/></XFDU>'


def annotation():
    epoch = datetime(2025, 3, 30, 18, 4, 11)
    def stamp(value):
        return (epoch + timedelta(seconds=value)).isoformat(timespec='microseconds')
    root = ET.fromstring(f'''<product><adsHeader><missionId>S1C</missionId><swath>S6</swath>
      <productType>SLC</productType><polarisation>HH</polarisation><absoluteOrbitNumber>1674</absoluteOrbitNumber>
      <missionDataTakeId>11340</missionDataTakeId></adsHeader><generalAnnotation><downlinkInformationList>
      <downlinkInformation><firstLineSensingTime>{stamp(0)}</firstLineSensingTime>
      <lastLineSensingTime>{stamp(25)}</lastLineSensingTime><downlinkValues><pri>0.001</pri></downlinkValues>
      </downlinkInformation></downlinkInformationList></generalAnnotation><imageAnnotation><imageInformation>
      <sliceNumber>1</sliceNumber><zeroDopMinusAcqTime>0.3</zeroDopMinusAcqTime><sliceList><slice>
      <sliceNumber>1</sliceNumber><sensingStartTime>{stamp(0)}</sensingStartTime>
      <sensingStopTime>{stamp(25)}</sensingStopTime></slice></sliceList></imageInformation></imageAnnotation>
      <dopplerCentroid><dcEstimateList/></dopplerCentroid></product>''')
    for start in (0, 12.5, 19.001):
        record = ET.SubElement(root.find('dopplerCentroid/dcEstimateList'), 'dcEstimate')
        for tag, value in [('fineDceAzimuthStartTime', start+.3),
                           ('fineDceAzimuthStopTime', start+6.3), ('azimuthTime', start+3.3)]:
            ET.SubElement(record, tag).text = stamp(value)
        ET.SubElement(record, 'fineDceList')
    return ET.tostring(root)


def response(value=b'', status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(value).encode() if isinstance(value, dict) else value
    result._content_consumed = True
    result.headers.update(headers or {})
    return result


def plan():
    root = f"{cdse.DOWNLOAD}/Products({UID})/Nodes({NAME})"
    return {'Id': UID, 'Name': NAME, 'files': [
        {'path': 'manifest.safe', 'url': root+'/Nodes(manifest.safe)/$value', 'bytes': len(MANIFEST)},
        {'path': 'annotation/'+ANN, 'url': root+f'/Nodes(annotation)/Nodes({ANN})/$value',
         'bytes': len(annotation())}]}


class CopernicusTest(unittest.TestCase):
    def setUp(self):
        self.client = cdse.Client(access_token='private-access-token', username='private-user', password='private-password')
        self.addCleanup(self.client.close)
        self.client.session.request = Mock()

    def test_seed_filter_and_input_validation(self):
        expression = cdse.search_filter(seed=NAME, polarisation='HH')
        self.assertIn("_001674_002C4C_", expression)
        self.assertIn("2025-03-30T16:04:12Z", expression)
        self.assertIn("_1SDH_", expression)
        for options in [{'seed': '../'+NAME}, {'seed': NAME, 'swath': 'S4'},
                        {'start': '2025-02-01', 'stop': '2025-01-01'},
                        {'seed': NAME, 'polarisation': "HH' or true"}]:
            with self.assertRaises(ValueError):
                cdse.search_filter(**options)

    def test_pagination_and_limit_are_explicit(self):
        products = [{'Id': f'00000000-0000-0000-0000-{i:012d}', 'Name': NAME, 'Online': True}
                    for i in range(3)]
        self.client.session.request.side_effect = [
            response({'value': products[:1], '@odata.nextLink': cdse.CATALOGUE+'?$skip=1'}),
            response({'value': products[1:]})]
        result = self.client.search(seed=NAME, limit=2)
        self.assertTrue(result['truncated'])
        self.assertEqual(len(result['products']), 2)
        self.assertTrue(all(call.kwargs['headers'] == {} for call in self.client.session.request.call_args_list))

    def test_plan_only_selects_manifest_and_matching_main_annotation(self):
        self.client.session.request.side_effect = [response({'result': [
            {'Name': 'manifest.safe', 'ContentLength': len(MANIFEST)}, {'Name': 'annotation'}]}),
            response({'result': [{'Name': ANN, 'ContentLength': len(annotation())},
                                 {'Name': ANN.replace('-hh-', '-hv-'), 'ContentLength': 5},
                                 {'Name': 'noise-'+ANN, 'ContentLength': 5},
                                 {'Name': 'measurement.tiff', 'ContentLength': 10**10}]})]
        result = self.client.plan({'Id': UID, 'Name': NAME, 'Online': True}, polarisation='HH')
        self.assertEqual([f['path'] for f in result['files']], ['manifest.safe', 'annotation/'+ANN])
        self.assertEqual(result['metadata_bytes'], len(MANIFEST)+len(annotation()))
        self.assertTrue(all('/Nodes(' in f['url'] and f['url'].endswith('/$value') for f in result['files']))
        self.assertTrue(all(call.kwargs['headers'] == {} for call in self.client.session.request.call_args_list))

    def test_download_hash_cache_and_secret_free_provenance(self):
        self.client.session.request.side_effect = [response(MANIFEST), response(annotation())]
        with TemporaryDirectory() as directory:
            result = self.client.download(plan(), directory)
            self.assertEqual(result['files']['annotation/'+ANN]['sha256'], hashlib.sha256(annotation()).hexdigest())
            self.client.download(plan(), directory)
            self.assertEqual(self.client.session.request.call_count, 2)
            cached = (Path(directory)/NAME/'metadata.json').read_text()
            for secret in ['private-access-token', 'private-user', 'private-password']:
                self.assertNotIn(secret, cached)
            (Path(directory)/NAME/'annotation'/ANN).write_bytes(b'!'*len(annotation()))
            with self.assertRaises(ValueError):
                self.client.download(plan(), directory)
            self.assertEqual(self.client.session.request.call_count, 2)

    def test_partial_or_invalid_download_never_promotes_file(self):
        for invalid in (b'<', b'!'*len(annotation())):
            with self.subTest(length=len(invalid)), TemporaryDirectory() as directory:
                self.client.session.request.side_effect = [response(MANIFEST), response(invalid)]
                with self.assertRaises((ValueError, ET.ParseError)):
                    self.client.download(plan(), directory)
                self.assertFalse((Path(directory)/NAME/'annotation'/ANN).exists())
                self.assertEqual(list(Path(directory).rglob('*.part')), [])
                record = json.loads((Path(directory)/NAME/'metadata.json').read_text())
                self.assertEqual(list(record['files']), ['manifest.safe'])

    def test_401_refreshes_once_and_never_sends_internal_kwargs(self):
        self.client.access_token, self.client.refresh_token = 'old-token', 'refresh-secret'
        self.client.session.request.side_effect = [response(status=401),
            response({'access_token': 'new-token', 'refresh_token': 'new-refresh', 'expires_in': 600}),
            response(b'xml')]
        result = self.client._request('GET', cdse.DOWNLOAD+'/Products(test)/Nodes', authenticated=True)
        self.assertEqual(result.content, b'xml')
        calls = self.client.session.request.call_args_list
        self.assertEqual(calls[1].args[0], 'POST')
        self.assertEqual(calls[1].kwargs['data']['grant_type'], 'refresh_token')
        self.assertEqual(calls[2].kwargs['headers']['Authorization'], 'Bearer new-token')
        self.assertTrue(all('_renewed' not in c.kwargs for c in calls))

    def test_password_login_totp_and_expiry_refresh(self):
        self.client.access_token, self.client.totp = None, '123456'
        self.client.session.request.side_effect = [response({
            'access_token': 'initial', 'refresh_token': 'refresh', 'expires_in': 600}),
            response({'access_token': 'renewed', 'refresh_token': 'refresh2', 'expires_in': 600})]
        self.assertEqual(self.client._token(), 'initial')
        call = self.client.session.request.call_args
        self.assertEqual(call.kwargs['data']['totp'], '123456')
        self.assertEqual(call.kwargs['data']['grant_type'], 'password')
        self.client.expires_at = 0
        self.assertEqual(self.client._token(), 'renewed')
        self.assertEqual(self.client.session.request.call_args.kwargs['data']['grant_type'], 'refresh_token')

    def test_repeated_401_stops_and_authentication_error_body_is_not_logged(self):
        self.client.refresh_token = 'refresh-secret'
        self.client.session.request.side_effect = [response(status=401),
            response({'access_token': 'renewed', 'expires_in': 600}),
            response({'error_description': 'private-password'}, status=401)]
        with self.assertRaises(RuntimeError) as caught:
            self.client._request('GET', cdse.DOWNLOAD+'/Products(test)/Nodes', authenticated=True)
        self.assertIn('HTTP 401', str(caught.exception))
        self.assertNotIn('private-password', str(caught.exception))
        self.assertEqual(self.client.session.request.call_count, 3)

    def test_saved_session_refreshes_without_persisting_password(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'session.json'
            client = cdse.Client(username='private-user', password='private-password', session_file=path)
            self.addCleanup(client.close)
            client.session.request = Mock(return_value=response({
                'access_token': 'initial', 'refresh_token': 'refresh', 'expires_in': 600}))
            client.authenticate()
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn('private-password', path.read_text())
            saved = json.loads(path.read_text());saved['expires_at'] = 0
            cdse.write_json(path, saved)
            restored = cdse.Client(session_file=path)
            self.addCleanup(restored.close)
            restored.session.request = Mock(return_value=response({
                'access_token': 'new', 'refresh_token': 'rotated', 'expires_in': 600}))
            restored.authenticate()
            self.assertEqual(restored.session.request.call_args.kwargs['data']['grant_type'], 'refresh_token')
            self.assertEqual(json.loads(path.read_text())['refresh_token'], 'rotated')
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                cdse.Client(session_file=path)

    def test_foreign_redirect_and_pagination_do_not_receive_token(self):
        self.client.session.request.return_value = response(status=302, headers={'Location': 'https://example.com/steal'})
        with self.assertRaises(ValueError):
            self.client._request('GET', cdse.DOWNLOAD+'/Products(test)/Nodes', authenticated=True)
        self.assertEqual(self.client.session.request.call_count, 1)
        self.client.session.request.return_value = response({'value': [], '@odata.nextLink': 'https://example.com/next'})
        with self.assertRaises(ValueError):
            self.client.search(seed=NAME)

    def test_offline_and_path_traversal_are_rejected(self):
        with self.assertRaises(ValueError):
            self.client.plan({'Id': UID, 'Name': NAME, 'Online': False})
        with TemporaryDirectory() as directory:
            malicious = plan()
            malicious['files'][0]['path'] = '../../escape.xml'
            with self.assertRaises(ValueError):
                self.client.download(malicious, directory)
        self.client.session.request.assert_not_called()


workflow = Path(__file__).parents[1]/'workflows/sentinel-1'
spec = importlib.util.spec_from_file_location('check_dce_azimuth_layout', workflow/'check_dce_azimuth_layout.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
spec = importlib.util.spec_from_file_location('copernicus_metadata', workflow/'copernicus_metadata.py')
tool = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'check_dce_azimuth_layout': layout}):
    spec.loader.exec_module(tool)


class CopernicusVerificationTest(unittest.TestCase):
    def test_reports_both_conventions_and_keeps_invalid_annotation(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            (base/ANN).write_bytes(annotation())
            (base/'manifest.safe').write_bytes(MANIFEST)
            broken = base/ANN.replace('-001.xml', '-002.xml')
            broken.write_text('<product>')
            report = tool.verify([base], output=base/'report.json')
            self.assertEqual(report['summary']['annotations'], 2)
            self.assertEqual(report['summary']['neither_matches'], 1)
            self.assertEqual(report['summary']['matching_annotations_by_convention'], {'last_line': 0, 'after_last_line': 1})
            self.assertEqual(report['annotations'][0]['software'][0]['version'], '003.90')
            self.assertIn('error', report['annotations'][1]['conventions']['last_line'])
            self.assertEqual(report['datatake_groups'][0]['observed_dc_records'], 3)
            self.assertEqual(report['datatake_groups'][0]['missing_slices_from_context'], [])

    def test_dry_run_never_prompts_or_authenticates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'catalogue.json'
            path.write_text(json.dumps({'products': [{'Id': UID, 'Name': NAME, 'Online': True}], 'truncated': False}))
            client = Mock()
            client.plan.return_value = {**plan(), 'metadata_bytes': 100}
            with patch('sys.argv', ['tool', 'download', '--catalogue', str(path), '--directory', directory,
                                    '--dry-run', '--login']), patch.object(tool, 'Client', return_value=client) as factory:
                self.assertEqual(tool.main(), 0)
            factory.assert_called_once_with()
            client.download.assert_not_called()
            self.assertEqual(json.loads((Path(directory)/'download_plan.json').read_text())['products'][0]['status'], 'planned')


if __name__ == '__main__':
    unittest.main()
