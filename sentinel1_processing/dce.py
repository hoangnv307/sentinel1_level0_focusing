"""Compatibility aliases for :mod:`doppler_centroid_estimation`."""

from .doppler_centroid_estimation import *  # noqa: F401,F403
from .doppler_centroid_estimation import (  # noqa: F401
    AzimuthDCEBlock,
    DCEConfig,
    DCERecord,
    DCESegment,
    PreparedSegmentScene,
    RangeDCEBlock,
    SegmentAlignment,
    Sentinel1DCE,
    compare_annotation_dce,
    evaluate_annotation_dce,
    fine_dce_cdce,
    prepare_annotation_records,
    prepare_dce_segments,
    records_summary,
    resolve_absolute_dce_with_geometry,
    robust_polynomial_fit,
    unwrap_fine_dce_dad,
)
