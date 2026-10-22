"""CDSE OData cho metadata Stripmap SLC; không tải measurement hoặc toàn SAFE."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import quote, urljoin, urlparse
from uuid import UUID
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"
TOKEN = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PRODUCT = re.compile(
    r"(?P<mission>S1[ABCD])_(?P<swath>S[1-6])_SLC__1S(?P<pol>SH|SV|DH|DV)_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<orbit>\d{6})_"
    r"(?P<datatake>[0-9A-F]{6})_[0-9A-F]{4}\.SAFE")
ANNOTATION = re.compile(r"s1[abcd]-s[1-6]-slc-(hh|hv|vv|vh)-\d{8}t\d{6}-\d{8}t\d{6}-\d{6}-[0-9a-f]{6}-\d{3}\.xml")


def product_info(name):
    match = PRODUCT.fullmatch(name)
    if not match:
        raise ValueError("Cần tên SAFE Sentinel-1 Stripmap SLC hợp lệ.")
    return match.groupdict()


def utc(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return instant.replace(tzinfo=timezone.utc) if instant.tzinfo is None else instant.astimezone(timezone.utc)


def search_filter(*, start=None, stop=None, seed=None, satellite=None, swath=None, polarisation=None):
    """Seed giới hạn cùng mission/orbit/datatake/swath, trong ±2 giờ quanh seed."""
    if satellite not in (None, "S1A", "S1B", "S1C", "S1D"):
        raise ValueError("Satellite không hợp lệ.")
    if swath not in (None, *(f"S{i}" for i in range(1, 7))):
        raise ValueError("Swath không hợp lệ.")
    if polarisation not in (None, "HH", "HV", "VV", "VH"):
        raise ValueError("Polarisation không hợp lệ.")
    parts = ["Collection/Name eq 'SENTINEL-1'", "contains(Name,'_SLC__1S')"]
    if seed:
        info = product_info(seed)
        if satellite not in (None, info["mission"]) or swath not in (None, info["swath"]):
            raise ValueError("Satellite/swath khác seed.")
        satellite, swath = info["mission"], info["swath"]
        instant = datetime.strptime(info["start"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        start = start or (instant - timedelta(hours=2)).isoformat()
        stop = stop or (instant + timedelta(hours=2)).isoformat()
        parts.append(f"contains(Name,'_{info['orbit']}_{info['datatake']}_')")
    if not start or not stop or utc(start) >= utc(stop):
        raise ValueError("Cần start < stop, hoặc seed để suy khoảng tìm kiếm.")
    parts.extend([f"ContentDate/Start ge {utc(start).isoformat().replace('+00:00', 'Z')}",
                  f"ContentDate/Start lt {utc(stop).isoformat().replace('+00:00', 'Z')}"])
    if satellite:
        parts.append(f"startswith(Name,'{satellite}_')")
    modes = [swath] if swath else [f"S{i}" for i in range(1, 7)]
    parts.append("(" + " or ".join(f"contains(Name,'_{s}_')" for s in modes) + ")")
    if polarisation:
        codes = {"HH": ("SH", "DH"), "HV": ("DH",), "VV": ("SV", "DV"), "VH": ("DV",)}[polarisation]
        parts.append("(" + " or ".join(f"contains(Name,'_1S{p}_')" for p in codes) + ")")
    return " and ".join(parts)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name, suffix=".part", delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Client:
    def __init__(self, *, access_token=None, username=None, password=None, totp=None,
                 timeout_s=30, max_file_bytes=16 * 1024 * 1024, session_file=None):
        if not math.isfinite(timeout_s) or timeout_s <= 0 or max_file_bytes <= 0:
            raise ValueError("Timeout và giới hạn file phải dương.")
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=.5, status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",), respect_retry_after_header=False)))
        self.session.mount(TOKEN, HTTPAdapter(max_retries=0))
        self.access_token, self.refresh_token = access_token, None
        self.username, self.password, self.totp = username, password, totp
        self.expires_at = math.inf if access_token else 0
        self.timeout_s, self.max_file_bytes = timeout_s, max_file_bytes
        self.session_file = Path(session_file).expanduser() if session_file else None
        if self.session_file and self.session_file.exists():
            metadata = self.session_file.stat()
            if self.session_file.is_symlink() or metadata.st_mode & 0o077 or metadata.st_uid != os.getuid():
                raise ValueError("File phiên phải thuộc user hiện tại, quyền 600 và không là symlink.")
            if not access_token and not (username and password):
                saved = json.loads(self.session_file.read_text())
                if (not isinstance(saved, dict) or not isinstance(saved.get("access_token"), str)
                        or not isinstance(saved.get("refresh_token"), str)
                        or not isinstance(saved.get("expires_at"), (int, float))
                        or not math.isfinite(saved["expires_at"])):
                    raise ValueError("File phiên CDSE không hợp lệ.")
                self.access_token, self.refresh_token = saved["access_token"], saved["refresh_token"]
                self.expires_at = time.monotonic() + saved["expires_at"] - time.time() - 30

    def close(self):
        self.session.close()

    def _request(self, method, url, *, authenticated=False, _renewed=False, **kwargs):
        original = url
        for redirect in range(6):
            host = urlparse(url)
            allowed = ({"download.dataspace.copernicus.eu", "zipper.dataspace.copernicus.eu"}
                       if authenticated else {"catalogue.dataspace.copernicus.eu", "download.dataspace.copernicus.eu",
                                              "identity.dataspace.copernicus.eu"})
            if (host.scheme != "https" or host.hostname not in allowed or host.username
                    or host.password or host.port not in (None, 443)):
                raise ValueError("URL CDSE không nằm trong endpoint cho phép.")
            headers = {"Authorization": f"Bearer {self._token()}"} if authenticated else {}
            try:
                response = self.session.request(method, url, headers=headers, timeout=self.timeout_s,
                                                allow_redirects=False, **kwargs)
            except requests.RequestException:
                raise RuntimeError("Lỗi kết nối CDSE; không ghi payload xác thực vào log.") from None
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                response.close()
                if method != "GET" or not location or redirect == 5:
                    raise RuntimeError("Redirect CDSE không hợp lệ.")
                url = urljoin(url, location)
                kwargs.pop("params", None)
                continue
            if response.status_code == 401 and authenticated and not _renewed:
                response.close()
                self._token(force=True)
                return self._request(method, original, authenticated=True, _renewed=True, **kwargs)
            if not response.ok:
                status = response.status_code
                response.close()
                raise RuntimeError(f"CDSE trả HTTP {status}; kiểm tra quyền, token hoặc trạng thái sản phẩm.")
            return response
        raise RuntimeError("Quá nhiều redirect CDSE.")

    def _token(self, *, force=False):
        if self.access_token and not force and time.monotonic() < self.expires_at:
            return self.access_token
        if self.refresh_token:
            data = {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
        elif self.username and self.password:
            data = {"grant_type": "password", "username": self.username, "password": self.password}
            if self.totp:
                data["totp"] = self.totp
        else:
            raise RuntimeError("Cần CDSE_ACCESS_TOKEN hoặc CDSE_USERNAME/CDSE_PASSWORD; có thể dùng --login.")
        with self._request("POST", TOKEN, data={"client_id": "cdse-public", **data}) as response:
            try:
                token = response.json()
                expires = float(token.get("expires_in", 600))
                access = token["access_token"]
                if not isinstance(access, str) or not access or not math.isfinite(expires) or expires <= 0:
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                raise RuntimeError("Token response không hợp lệ.") from None
        self.access_token = access
        self.refresh_token = token.get("refresh_token")
        self.expires_at = time.monotonic() + max(0, expires - 30)
        if self.session_file:
            if not self.refresh_token:
                raise RuntimeError("CDSE không trả refresh token để tạo phiên dùng lại.")
            write_json(self.session_file, {"access_token": access, "refresh_token": self.refresh_token,
                                           "expires_at": time.time() + expires})
        return access

    def authenticate(self):
        """Xác thực và lưu token nếu session_file được yêu cầu; không lưu password."""
        self._token()

    def search(self, *, limit=30, **filters):
        if not 1 <= limit <= 1000:
            raise ValueError("Limit phải trong khoảng 1–1000.")
        expression = search_filter(**filters)
        params = {"$filter": expression, "$select": "Id,Name,Online,S3Path,ContentLength,ContentDate,PublicationDate",
                  "$orderby": "ContentDate/Start asc", "$top": min(100, limit + 1)}
        url, products, seen = CATALOGUE, [], set()
        while url:
            if url in seen:
                raise ValueError("Catalogue pagination bị lặp.")
            seen.add(url)
            with self._request("GET", url, params=params) as response:
                page = response.json()
            values = page["value"]
            for product in values:
                product_info(product["Name"])
                product["Id"] = str(UUID(product["Id"]))
                if product["Id"] not in {p["Id"] for p in products}:
                    products.append(product)
                if len(products) > limit:
                    break
            url = page.get("@odata.nextLink")
            if len(products) > limit:
                break
            params = None
            if url and urlparse(url).hostname != "catalogue.dataspace.copernicus.eu":
                raise ValueError("Catalogue nextLink không hợp lệ.")
        return {"queried_at": datetime.now(timezone.utc).isoformat(), "filter": expression,
                "limit": limit, "truncated": len(products) > limit,
                "products": products[:limit]}

    def plan(self, product, *, polarisation=None):
        info = product_info(product["Name"])
        uid = str(UUID(product["Id"]))
        if product.get("Online") is not True:
            raise ValueError("Sản phẩm không online; tool không tự đặt lệnh khôi phục archive.")
        if polarisation not in (None, "HH", "HV", "VV", "VH"):
            raise ValueError("Polarisation không hợp lệ.")
        root = f"{DOWNLOAD}/Products({uid})/Nodes({quote(product['Name'], safe='')})"
        with self._request("GET", root + "/Nodes") as response:
            children = response.json()["result"]
        if not any(n["Name"] == "annotation" for n in children):
            raise ValueError("Sản phẩm thiếu thư mục annotation.")
        with self._request("GET", root + "/Nodes(annotation)/Nodes") as response:
            annotations = response.json()["result"]
        selected = [("manifest.safe", n) for n in children if n["Name"] == "manifest.safe"]
        for node in annotations:
            match = ANNOTATION.fullmatch(node["Name"])
            if match and (polarisation is None or match[1].upper() == polarisation):
                parts = node["Name"].split("-")
                if [parts[0].upper(), parts[1].upper(), parts[6], parts[7].upper()] != [
                        info["mission"], info["swath"], info["orbit"], info["datatake"]]:
                    raise ValueError("Tên annotation không khớp identity sản phẩm.")
                selected.append(("annotation/" + node["Name"], node))
        if len(selected) < 2 or selected[0][0] != "manifest.safe":
            raise ValueError("Thiếu manifest hoặc annotation đúng kênh yêu cầu.")
        files = []
        for relative, node in selected:
            size = int(node["ContentLength"])
            if not 0 < size <= self.max_file_bytes:
                raise ValueError("File metadata rỗng hoặc vượt giới hạn dung lượng.")
            segments = "/".join(f"Nodes({quote(p, safe='')})" for p in relative.split("/"))
            files.append({"path": relative, "url": f"{root}/{segments}/$value", "bytes": size})
        return {"Id": uid, "Name": product["Name"], "identity": info, "files": files,
                "metadata_bytes": sum(p["bytes"] for p in files)}

    @staticmethod
    def _validate_xml(path, relative):
        root = ET.parse(path).getroot()
        if relative == "manifest.safe":
            if root.tag.split("}")[-1] != "XFDU":
                raise ValueError("Manifest không phải XFDU XML.")
        elif root.find("dopplerCentroid/dcEstimateList") is None:
            raise ValueError("Annotation không có DC estimate list.")
        else:
            name = Path(relative).name.split("-")
            if [root.findtext("adsHeader/" + k) for k in ("missionId", "swath", "productType", "polarisation")] != [
                    name[0].upper(), name[1].upper(), "SLC", name[3].upper()]:
                raise ValueError("Annotation header khác tên file.")

    def download(self, plan, directory):
        product_info(plan["Name"])
        str(UUID(plan["Id"]))
        base = Path(directory).expanduser().resolve()
        target = base / plan["Name"]
        if target.is_symlink():
            raise ValueError("Không ghi qua symlink product directory.")
        target.mkdir(parents=True, exist_ok=True)
        record_path = target / "metadata.json"
        if record_path.is_symlink():
            raise ValueError("Không đọc cache metadata qua symlink.")
        record = json.loads(record_path.read_text()) if record_path.exists() else {
            "Id": plan["Id"], "Name": plan["Name"], "files": {}}
        if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
            raise ValueError("Cache metadata.json không đúng cấu trúc.")
        if record.get("Id") != plan["Id"] or record.get("Name") != plan["Name"]:
            raise ValueError("Cache có product identity khác.")
        for item in plan["files"]:
            relative = item["path"]
            if (relative != "manifest.safe"
                    and not (relative.startswith("annotation/") and ANNOTATION.fullmatch(relative[11:]))):
                raise ValueError("Chỉ cho phép manifest và annotation chính.")
            path = target / relative
            if not path.resolve().is_relative_to(target.resolve()) or path.is_symlink():
                raise ValueError("Đường dẫn metadata vượt product directory.")
            if path.exists():
                cached = record["files"].get(relative, {})
                if (path.stat().st_size != item["bytes"]
                        or hashlib.sha256(path.read_bytes()).hexdigest() != cached.get("sha256")
                        or cached.get("url") != item["url"]):
                    raise ValueError(f"Cache không hợp lệ: {relative}; kiểm tra/di chuyển file trước khi tải lại.")
                self._validate_xml(path, relative)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as file:
                    temporary = Path(file.name)
                    digest, length = hashlib.sha256(), 0
                    try:
                        with self._request("GET", item["url"], authenticated=True, stream=True) as response:
                            for chunk in response.iter_content(65536):
                                length += len(chunk)
                                if length > min(item["bytes"], self.max_file_bytes):
                                    raise ValueError("Metadata tải về vượt dung lượng đã kê khai.")
                                file.write(chunk)
                                digest.update(chunk)
                    except requests.RequestException:
                        raise RuntimeError("Tải metadata bị gián đoạn; không lưu file chưa hoàn tất.") from None
                if length != item["bytes"]:
                    raise ValueError("Metadata tải về bị thiếu byte.")
                self._validate_xml(temporary, relative)
                temporary.replace(path)
                record["files"][relative] = {**item, "sha256": digest.hexdigest(),
                                             "downloaded_at": datetime.now(timezone.utc).isoformat()}
                write_json(record_path, record)
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
        return record
