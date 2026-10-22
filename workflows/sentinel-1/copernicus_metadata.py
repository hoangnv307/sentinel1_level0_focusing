"""Tìm, tải metadata Stripmap SLC từ CDSE và kiểm chứng giả thuyết lịch DCE."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import json
import math
import os
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

from sentinel1_processing.utils.copernicus import Client, product_info, write_json
import check_dce_azimuth_layout as layout


def verify(inputs, *, output, block_size_lines=6000, nominal_spacing_s=10, tolerance_s=5e-6):
    """Báo cả hai convention; không fit/chọn model theo kết quả hoặc bỏ file lỗi."""
    if (block_size_lines <= 1 or not all(math.isfinite(v) and v > 0
                                       for v in (nominal_spacing_s, tolerance_s))):
        raise ValueError("Block size phải >1, spacing và tolerance phải hữu hạn, dương.")
    paths = set()
    for item in inputs:
        item = Path(item).expanduser()
        if not item.exists():
            raise ValueError(f"Không tìm thấy {item}.")
        paths.update(item.rglob("s1*-s*-slc-*.xml") if item.is_dir() else [item])
    if not paths:
        raise ValueError("Không tìm thấy annotation Stripmap SLC.")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "config": {"block_size_lines": block_size_lines, "nominal_spacing_s": nominal_spacing_s,
                         "tolerance_s": tolerance_s}, "annotations": [], "shared_acquisition_windows": [],
              "status": "Giả thuyết nghiên cứu; không xác nhận tương đương IPF"}
    rows = []
    for path in sorted(paths):
        entry = {"file": str(path.resolve()), "conventions": {}}
        manifest = (path.parent.parent if path.parent.name == "annotation" else path.parent)/"manifest.safe"
        if manifest.exists():
            try:
                root = ET.parse(manifest).getroot()
                entry["manifest"] = str(manifest.resolve())
                entry["software"] = [e.attrib for e in root.iter()
                                     if e.tag.split("}")[-1] == "software" and e.get("name")]
                entry["auxiliary_resources"] = [e.attrib for e in root.iter()
                    if e.tag.split("}")[-1] == "resource" and e.get("role", "").startswith("AUX_")]
            except (ET.ParseError, OSError) as error:
                entry["manifest_error"] = str(error)
        for convention in ("last_line", "after_last_line"):
            try:
                result = layout.analyze_annotation(path, azimuth_block_size_lines=block_size_lines,
                    nominal_spacing_s=nominal_spacing_s, time_tolerance_s=tolerance_s,
                    tail_stop_convention=convention)
                entry["conventions"][convention] = result
                if convention == "last_line":
                    rows.append(result)
            except (ValueError, ET.ParseError, OSError) as error:
                entry["conventions"][convention] = {"matches": False, "error": str(error)}
        report["annotations"].append(entry)
    try:
        report["shared_acquisition_windows"] = layout.shared_windows(rows, time_tolerance_s=tolerance_s)
    except ValueError as error:
        report["shared_windows_error"] = str(error)
    # Một annotation chỉ được đếm một lần dù cả hai convention cùng khớp ở slice chưa cuối.
    report["summary"] = {"annotations": len(paths), "datatakes": len({
        (r["mission"], r["absolute_orbit"], r["datatake_id"]) for r in rows}),
        "matching_annotations_by_convention": {c: sum(e["conventions"][c]["matches"] for e in report["annotations"])
                                              for c in ("last_line", "after_last_line")},
        "neither_matches": sum(not any(v["matches"] for v in e["conventions"].values())
                               for e in report["annotations"]),
        "shared_windows": len(report["shared_acquisition_windows"])}
    groups = {}
    for row in rows:
        key = (row["mission"], row["absolute_orbit"], row["datatake_id"], row["swath"], row["polarisation"])
        groups.setdefault(key, []).append(row)
    report["datatake_groups"] = []
    for (mission, orbit, datatake, swath, pol), group in sorted(groups.items(), key=lambda p: str(p[0])):
        expected = {s["number"] for row in group for s in row["slice_context"]}
        present = {row["slice_number"] for row in group}
        report["datatake_groups"].append({"mission": mission, "orbit": orbit, "datatake_id": datatake,
            "swath": swath, "polarisation": pol, "annotations": len(group),
            "observed_dc_records": sum(len(row["acquisition_windows"]) for row in group),
            "slice_numbers": sorted(present), "missing_slices_from_context": sorted(expected - present)})
    write_json(output, report)
    return report


def client_from_environment(*, login=False, timeout_s=30):
    access = os.environ.get("CDSE_ACCESS_TOKEN")
    username, password, totp = (os.environ.get(k) for k in ("CDSE_USERNAME", "CDSE_PASSWORD", "CDSE_TOTP"))
    if login and not access:
        username = username or input("CDSE username/email: ").strip()
        password = password or getpass.getpass("CDSE password: ")
        totp = totp or getpass.getpass("OTP nếu bật 2FA (Enter nếu không): ")
    return Client(access_token=access, username=username, password=password, totp=totp, timeout_s=timeout_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Tìm sản phẩm; không cần đăng nhập.")
    search.add_argument("--seed", help="Tên sản phẩm .SAFE làm mốc tìm cùng datatake.")
    search.add_argument("--start")
    search.add_argument("--stop")
    search.add_argument("--satellite", choices=("S1A", "S1B", "S1C", "S1D"))
    search.add_argument("--swath", choices=[f"S{i}" for i in range(1, 7)])
    search.add_argument("--polarisation", choices=("HH", "HV", "VV", "VH"))
    search.add_argument("--limit", type=int, default=30)
    search.add_argument("--output", type=Path, default=Path("output/sentinel-1/copernicus/catalogue.json"))
    download = commands.add_parser("download", help="Lập danh sách hoặc tải annotation + manifest.")
    download.add_argument("--catalogue", required=True, type=Path)
    download.add_argument("--directory", type=Path, default=Path.home()/"sar_data/focusing_data/copernicus_metadata")
    download.add_argument("--polarisation", choices=("HH", "HV", "VV", "VH"))
    download.add_argument("--dry-run", action="store_true", help="Chỉ duyệt file/dung lượng, không đăng nhập.")
    download.add_argument("--login", action="store_true", help="Nhập mật khẩu ẩn tại terminal.")
    download.add_argument("--allow-truncated", action="store_true", help="Cho phép catalogue đã bị cắt ở limit.")
    check = commands.add_parser("verify", help="Kiểm cả hai tail convention trên metadata cục bộ.")
    check.add_argument("inputs", nargs="+", type=Path)
    check.add_argument("--output", type=Path, default=Path("output/sentinel-1/copernicus/verification.json"))
    check.add_argument("--block-size-lines", type=int, default=6000)
    check.add_argument("--nominal-spacing-s", type=float, default=10)
    check.add_argument("--tolerance-s", type=float, default=5e-6)
    args = parser.parse_args()
    if args.command == "verify":
        report = verify(args.inputs, output=args.output, block_size_lines=args.block_size_lines,
                        nominal_spacing_s=args.nominal_spacing_s, tolerance_s=args.tolerance_s)
        print(json.dumps(report["summary"], ensure_ascii=False))
        return int(report["summary"]["neither_matches"] > 0 or "shared_windows_error" in report)
    client = client_from_environment(login=args.command == "download" and args.login and not args.dry_run)
    try:
        if args.command == "search":
            report = client.search(limit=args.limit, seed=args.seed, start=args.start, stop=args.stop,
                                   satellite=args.satellite, swath=args.swath, polarisation=args.polarisation)
            groups = Counter((p["Name"].split("_")[0], product_info(p["Name"])["orbit"],
                              product_info(p["Name"])["datatake"]) for p in report["products"])
            report["datatakes"] = [{"mission": m, "orbit": o, "datatake": d, "products": n}
                                    for (m, o, d), n in sorted(groups.items())]
            write_json(args.output, report)
            print(f"Tìm thấy {len(report['products'])} sản phẩm, {len(groups)} datatake; truncated={report['truncated']}.")
            return 0
        catalogue = json.loads(args.catalogue.expanduser().read_text())
        if catalogue.get("truncated") and not args.allow_truncated:
            raise ValueError("Catalogue bị cắt bởi limit; tăng limit hoặc thu hẹp tìm kiếm trước khi tải.")
        products = catalogue.get("products", [])
        if not products:
            raise ValueError("Catalogue không có sản phẩm.")
        if len({p["Id"] for p in products}) != len(products):
            raise ValueError("Catalogue chứa UUID trùng.")
        index_path = args.directory.expanduser()/("download_plan.json" if args.dry_run else "download_index.json")
        index = {"created_at": datetime.now(timezone.utc).isoformat(), "dry_run": args.dry_run,
                 "catalogue": str(args.catalogue.resolve()), "products": []}
        for product in products:
            entry = {"Id": product["Id"], "Name": product["Name"]}
            try:
                plan = client.plan(product, polarisation=args.polarisation)
                entry["plan"] = plan
                if not args.dry_run:
                    client.download(plan, args.directory)
                entry["status"] = "planned" if args.dry_run else "downloaded_or_verified_cache"
                print(f"{product['Name']}: {len(plan['files'])} file, {plan['metadata_bytes']} byte, {entry['status']}", flush=True)
            except (ValueError, RuntimeError, OSError, KeyError, ET.ParseError) as error:
                entry["status"], entry["error"] = "error", str(error)
                print(f"{product['Name']}: {entry['error']}", file=sys.stderr)
            index["products"].append(entry)
            write_json(index_path, index)
        return int(any(e["status"] == "error" for e in index["products"]))
    finally:
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, ET.ParseError) as error:
        raise SystemExit(f"Lỗi: {error}") from None
