"""Tham số SM/S6/VV dùng bởi demo, phân nhóm theo DAD Appendix D.

Các giá trị AUX là snapshot của ba file trong ``references/``. LUT antenna
không được sao chép vào Python; khi triển khai calibration đầy đủ phải đọc trực
tiếp AUX_CAL.
"""

AUX_INS_FILE = "references/s1a-aux-ins.xml"
AUX_PP1_FILE = "references/s1a-aux-pp1.xml"
AUX_CAL_FILE = "references/s1a-aux-cal.xml"
AUX_PP1_PRODUCT_ID = "SM_SL1__1"
SWATH = "S6"
POLARISATION = "VV"


# AUX_INS: auxiliaryInstrument và swathParams[swath="S6"].
RADAR_FREQUENCY_HZ = 5_405_000_454.33435
RADAR_WAVELENGTH_M = 299_792_458.0 / RADAR_FREQUENCY_HZ
SWST_BIAS_S = -8.2256909e-9
RX_GAIN_TREND_COEFFICIENTS_V = (
    -41896.866,
    566.024172,
    -4.86310364,
    0.016242762,
    -1.57185656e-5,
    68941.3636,
    848.897319,
    1.0,
)
RX_GAIN_OVERSHOOT_COEFFICIENTS_V = (
    1.2195,
    0.039369,
    0.0034977,
    0.0042522,
    -3.2706,
)


# AUX_PP1: productId="SM_SL1__1", swath="S6". Các boolean dưới đây là yêu
# cầu cấu hình của IPF, không phải trạng thái implementation của demo.
CORRECT_IQ_BIAS = True
CORRECT_RX_VARIATION = True
CORRECT_BISTATIC_DELAY = True
BISTATIC_DELAY_METHOD = "Coarse"
DCE_RMS_ERROR_THRESHOLD_HZ = 20.0
# Empirical L0-estimate/L1 Fine-DC parity limit for the supplied S6 scene.
# This is a regression-test threshold, not the AUX_PP1 polynomial-fit limit.
DCE_L1_FINE_RMSE_THRESHOLD_HZ = 3.0
# NOTE: maxDeltaFdc (100 Hz) giới hạn biến thiên DC giữa các azimuth block, dùng
# tính overlap focus (DAD §9.12) — KHÔNG phải giới hạn |f_DC| và không suy ra N_amb.
FOCUS_AZIMUTH_BANDWIDTH_HZ = 1398.0
FOCUS_FFT_LENGTH = 4096
EXTRA_AZIMUTH_OVERLAP_SAMPLES = 50
RRF_SPECTRUM = "Extended Tapered"
APPLY_ELEVATION_ANTENNA_PATTERN = True
APPLY_RANGE_SPREADING_LOSS = True
APPLY_AZIMUTH_ANTENNA_PATTERN = True
RFI_MITIGATION = "Never"
RANGE_WINDOW = "Hamming"
RANGE_WINDOW_COEFFICIENT = 0.75
RANGE_PROCESSING_BANDWIDTH_HZ = 42_200_000.0
RANGE_LOOK_BANDWIDTH_HZ = 42_200_000.0
RANGE_LOOKS = 1
RANGE_PIXEL_SPACING_M = 3.1
AZIMUTH_WINDOW = "Hamming"
AZIMUTH_WINDOW_COEFFICIENT = 0.75
AZIMUTH_POST_PROCESSING_BANDWIDTH_HZ = 1398.0
AZIMUTH_LOOK_BANDWIDTH_HZ = 1398.0
AZIMUTH_LOOKS = 1
AZIMUTH_PIXEL_SPACING_M = 4.1


# AUX_CAL: calibrationParams[swath="S6", polarisation="VV"].
ABSOLUTE_CALIBRATION_CONSTANT = 1.0
NOISE_CALIBRATION_FACTOR = 0.673934


# Internal theo DAD Appendix D: lựa chọn của demo, không phải giá trị AUX.
DCE_AZIMUTH_BLOCK_SIZE_LINES = 6000
DCE_RANGE_BLOCKS = 20
DCE_RANGE_BLOCK_SIZE_SAMPLES = 1000
DCE_RANGE_ROI_STOP_SAMPLE = 17507
DCE_UNWRAP_FFT_LENGTH = 4096
DCE_OUTLIER_SIGMA = 2.5
# Ambiguity index N_amb such that f_abs = f_fine + N_amb*PRF (DAD §5.4).
# NOTE: mặc định 0 là *shortcut đã kiểm chứng* trên chính scene này, không phải
# hệ quả của maxFdc (100 Hz đó chỉ là giới hạn biến thiên giữa các azimuth block,
# dùng cho overlap, chứ KHÔNG chứng minh |f_DC| < PRF/2). Để pipeline L0 -> L1
# độc lập, phải truyền geometry_dc_provider tính N_amb từ orbit/attitude thay vì
# phụ thuộc maxFdc. Giữ bằng 0 tạm thời vì đã khớp annotation của scene S6 này;
# khi nối geometry DC (cần solver quỹ đạo/attitude theo doppler_centroid_estimation
# GeometryDcProvider) thì bỏ hằng này.
DCE_AMBIGUITY_NUMBER = 0
SRC_SEGMENT_SAMPLES = 1024
RCMC_KERNEL_LENGTH = 16
RCMC_PHASES = 64
AZIMUTH_TIME_CORRECTION_S = 0.0


__all__ = [name for name in globals() if name.isupper()]
