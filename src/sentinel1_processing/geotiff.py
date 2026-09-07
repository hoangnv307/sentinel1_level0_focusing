"""Xuất SLC thành GeoTIFF có GCP lấy từ annotation Sentinel-1 L1."""

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import tifffile


def export_slc_with_gcps(
    slc, annotation_xml, destination, *, scale=1.0, overwrite=False
):
    """Ghi SLC thành một band complex-int16 và nhúng toàn bộ GCP WGS-84.

    ``slc`` phải có cùng shape với ``numberOfLines/numberOfSamples`` trong
    annotation. ``scale`` mặc định giữ nguyên đơn vị biên độ; hàm từ chối ghi
    nếu kết quả vượt miền int16 thay vì âm thầm clipping.
    """
    data = np.asarray(slc)
    if data.ndim != 2 or not np.issubdtype(data.dtype, np.complexfloating):
        raise ValueError("slc phải là mảng complex 2-D (azimuth, range).")

    root = ElementTree.parse(annotation_xml).getroot()
    image_info = root.find("./imageAnnotation/imageInformation")
    if image_info is None:
        raise ValueError("Annotation không có imageAnnotation/imageInformation.")
    expected_shape = (
        int(image_info.findtext("numberOfLines")),
        int(image_info.findtext("numberOfSamples")),
    )
    if data.shape != expected_shape:
        raise ValueError(
            f"Shape SLC {data.shape} không khớp annotation {expected_shape}."
        )

    points = root.findall(
        "./geolocationGrid/geolocationGridPointList/geolocationGridPoint"
    )
    if not points:
        raise ValueError("Annotation không có geolocationGridPoint.")
    tiepoints = []
    for point in points:
        pixel = float(point.findtext("pixel"))
        line = float(point.findtext("line"))
        longitude = float(point.findtext("longitude"))
        latitude = float(point.findtext("latitude"))
        height = float(point.findtext("height", "0"))
        values = (pixel, line, 0.0, longitude, latitude, height)
        if not np.all(np.isfinite(values)):
            raise ValueError("GCP chứa giá trị không hữu hạn.")
        if not (0 <= pixel < data.shape[1] and 0 <= line < data.shape[0]):
            raise ValueError(f"GCP ({pixel}, {line}) nằm ngoài ảnh.")
        tiepoints.extend(values)

    # GeoKeyDirectory: geographic model, pixel-is-point, WGS-84, degree.
    geokeys = (
        1, 1, 0, 4,
        1024, 0, 1, 2,
        1025, 0, 1, 2,
        2048, 0, 1, 4326,
        2054, 0, 1, 9102,
    )
    output = Path(destination)
    if output.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("destination phải có đuôi .tif hoặc .tiff.")
    output.parent.mkdir(parents=True, exist_ok=True)

    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale phải là số hữu hạn dương.")
    max_component = 0.0
    rows_per_strip = 64
    for start in range(0, data.shape[0], rows_per_strip):
        block = data[start:start + rows_per_strip]
        if not np.all(np.isfinite(block)):
            raise ValueError("SLC chứa NaN hoặc Inf.")
        max_component = max(
            max_component,
            float(np.max(np.maximum(np.abs(block.real), np.abs(block.imag)))),
        )
    if max_component * scale > np.iinfo(np.int16).max:
        raise ValueError(
            f"scale={scale:g} gây clipping; scale tối đa là "
            f"{np.iinfo(np.int16).max / max_component:.9g}."
        )

    def quantized_strips():
        for start in range(0, data.shape[0], rows_per_strip):
            block = data[start:start + rows_per_strip]
            iq = np.stack(
                (np.rint(block.real * scale), np.rint(block.imag * scale)), axis=-1
            ).astype("<i2")
            yield iq.view("<i4").reshape(block.shape)

    output_bytes = data.size * 2 * np.dtype(np.int16).itemsize
    tifffile.imwrite(
        output,
        data=quantized_strips(),
        shape=data.shape,
        dtype="<i4",
        mode="w" if overwrite else "x",
        bigtiff=output_bytes >= 2**32 - 2**25,
        byteorder="<",
        photometric="minisblack",
        rowsperstrip=rows_per_strip,
        metadata=None,
        software="sentinel1_level0_decoder_demo",
        extratags=[
            (33922, "d", len(tiepoints), tuple(tiepoints), False),
            (34735, "H", len(geokeys), geokeys, False),
        ],
    )
    # NumPy không có dtype complex-int16. Ghi đúng byte IQ trước, sau đó đổi
    # SampleFormat từ signed integer (2) sang complex signed integer (5).
    with tifffile.TiffFile(output, mode="r+b") as product:
        product.pages[0].tags["SampleFormat"].overwrite(5)
    return output


__all__ = ["export_slc_with_gcps"]
