# Internal time delay từ PG L0 vào RRF — S1A/S4/VV

Ngày 21/10/2026, tiếp nối [chuẩn hóa PG](sentinel1_pg_normalization.md).
Đã chuyển delay từ một giá trị chẩn đoán thành đầu vào thực sự của RRF
trong demo DCE S4. CLI mặc định đo từ L0/AUX, kể cả khi không bật PG drift.

## 1. Công thức, nguồn và validity

Theo DAD Issue 2/5 §4.2.1.4, Eq. 4-34:

```text
delay_m = lag_peak_m / Fs_PG - (deltaTGuard1 - deltaTSuppr)
delay   = mean(delay_m của các sequence có correlation peak hợp lệ)
```

- Lag đo bằng correlation giữa replica tái dựng và nominal PG chirp 1/N
  từ header L0, FFT zoom/parabola như báo cáo PG. Không lấy lag L1.
- Fs_PG từ Range Decimation L0; không dùng bandwidth 100 MHz thay Fs.
- `deltaTGuard1`, `deltaTSuppr` đọc từ root AUX_INS; đơn vị giây.
- DAD yêu cầu trung bình trên PG sequence có peak hợp lệ. **Validity của
  correlation peak khác PG amplitude/phase QC**: không dùng mask chọn PG
  correction hay source extracted/model để loại delay theo amplitude/phase.
- Adapter hiện kiểm finite/nonzero và peak nằm trong cửa sổ đủ support
  nominal chirp; sequence lỗi làm adapter báo lỗi, chưa repair/drop tự động.
- DAD §4.2.2 bước 1 chuyển delay trung bình sang SLC processor để dịch
  chirp/RRF. Tái dựng/dịch **image replica extracted** là nhánh khác, chưa
  được triển khai ở thay đổi này.

Không trừ thêm `AUX_INS/internalCalibrationParams/timeDelay` hoặc delay L1
khỏi Eq. 4-34. Công thức đã trừ mốc guard/suppression. L1 và AUX timeDelay
không được dùng để fit hoặc hiệu chỉnh kết quả đo L0.

Các nguồn đối chiếu:

- DAD trong `references/`, §4.2.1.4, §4.2.2, §6.1.1 và §9.20.
- [Calibration and Characterisation Plan 8.1](https://sentiwiki.copernicus.eu/__attachments/a_50b520d3188da65898086d651f828368627281b555bdfe2be7575e78fbea067d/S1-PL-ASD-PL-0001-Sentinel-1A-1B-SAR-Instrument-Calibration-and-Characterisation-Plan-8.1.pdf),
  §4.6.3, §9.2.4: bù instrument delay qua replica của range compression.

## 2. Dấu phase ramp và range axis

Với quy ước Fourier DAD §9.20, thiết bị làm tín hiệu trễ dương `d`:

```text
X_delayed(f) = X_ideal(f) * exp(-2j*pi*f*d)
H_corrected(f) = H_zero_delay(f) * exp(+2j*pi*f*d)
```

RRF advance tín hiệu đúng một lần. Dấu `+` được kiểm tra bằng mô hình
thuận tạo raw delayed độc lập, sau đó so range compression với convolution
không delay. Không sao chép máy móc dấu `-` của tham số misregistration Δt
trong DAD Eq. 6-2 sang biến **positive hardware delay** có định nghĩa khác.

Range axis vẫn chỉ bỏ implementation delay `M-1` của matched filter theo
convention SLC hiện có. Không cộng/trừ `d` lần nữa vào axis hoặc SWST bias.
Fractional SWST alignment vẫn dùng `range_time_shift_s` riêng; nó dịch cả
grid và phase ramp như trước. Đây không phải hardware delay thứ hai.

## 3. API và đường đi dữ liệu

Pure core:

```python
delay = mean_internal_delay(
    peak_lags_samples,
    sample_rate_hz=...,
    delta_t_guard1_s=...,
    delta_t_suppr_s=...,
    valid=correlation_peak_valid_mask,
)
```

`valid` phải là mask bool cùng shape. Chỉ lags được chọn phải hữu hạn;
không có peak hợp lệ thì báo lỗi, không trả 0 hoặc đoán fallback.
Fs phải dương; guard/suppression hữu hạn không âm. Cả hai timing được
caller truyền vào, không chôn hằng số trong thuật toán.

`utils/pg_calibration_input.analyze_calibration` thêm report `internal_delay`:
mean, per-sequence delay, lags, Fs, guard/suppression, valid indices,
FFT zoom factor và denominator floor. Calibration chỉ đọc L0/AUX.

RRF và workflow:

- `create_freq_domain(..., internal_delay_s=0.0)` nhận delay cho nominal
  chirp từ header; chuyển tới builder chung `from_coefficients`.
- `ReferenceInput.range_reference(..., internal_delay_s=None)` cho phép
  caller thay riêng delay L1. Override **thay thế**, không cộng thêm.
- `compress_echoes(..., internal_delay_s=None,
  minimum_far_range_discard_samples=0)` nhận giá trị thực sự cần áp dụng.
  `None` giữ hành vi cũ cho audit/check scripts: reference có sẵn thì lấy
  delay của reference, nếu không thì 0. **CLI luôn truyền delay tường minh.**
- Main CLI chỉ gọi calibration một lần khi cần delay L0 và/hoặc PG L0.
  Bật delay L0 không tự bật PG drift. RRF default vẫn dùng chirp header;
  reference mode vẫn dùng nominal coefficients AUX và gain L1 như trước.

`internal_delay_s` áp vào RRF hiện được giới hạn không âm; delay đo ra âm,
không hữu hạn hoặc không đủ range support đều báo lỗi. Không fallback sang
L1 hoặc constant AUX để giữ lượt chạy tiếp tục.

## 4. Range support và ROI chung

Advance cần mẫu tương lai ở mép xa. Với `a` là fractional alignment [s]:

```text
required_far_discard = max(0, ceil((d+a)*Fs_echo))
far_discard = max(required_far_discard, minimum_far_range_discard_samples)
stop = (raw_range_samples - tx_pulse_samples) - far_discard
```

Code trừ epsilon số học **1e-8 sample** trước ceil để tránh một shift gần
zero do phép trừ SWST sinh ra ~1e-11 sample rồi bị làm tròn thành một sample.
Đây là tolerance floating point, không phải threshold hiệu chỉnh thiết bị.
Guard xét **tổng** hai phase ramp: ví dụ delay 1,8 sample + alignment
0,4 sample cần bỏ 3 sample, không phải 2. Caller không thể dùng minimum
nhỏ để vô hiệu guard bắt buộc. Nếu còn dưới hai mẫu thì báo lỗi trước FFT.

Đây là guard support theo vị trí chirp hữu hạn. Fractional shift bằng FFT
có sinc tails; không khẳng định đáp ứng có compact support tuyệt đối hoặc
không còn edge residual của IPF Extended Tapered chưa biết.

Cảnh S4: Fs_echo=56.302.083,36 Hz; delay khoảng 24,396 sample nên bỏ
**25 mẫu**, ROI chung còn **18.145** thay baseline chưa delay 18.170.
Để so ảnh hưởng delay riêng, cả cặp none/L0 đều đặt minimum=25; range
centres, polynomial origin và Fine DC L1 nội suy tại cùng grid. Không so
hai ROI khác nhau rồi quy tất cả chênh lệch cho delay.

## 5. CLI, default và provenance

```text
--internal-delay-source l0    # default: đo từ PG L0/AUX
--internal-delay-source l1    # chỉ cùng --l1-input-corrections
--internal-delay-source none  # delay=0, ablation
--minimum-far-range-discard-samples 25  # giữ ROI chung khi đối chứng
```

Default output thêm suffix `_delay_l0`, `_delay_l1` hoặc `_delay_none`
để không ghi đè baseline cũ. `--internal-delay-source l1` thiếu reference
mode bị parser từ chối. Các tham số PG FFT zoom/floor hiện có cũng điều
khiển phép đo peak cho delay; giá trị ghi trong report.

JSON DCE có `internal_delay_compensation`: source, `applied_s`, số đo L0,
delay L1 tham chiếu nếu có và minimum guard. `input_corrections` ghi đè
delay thực sự áp dụng lên summary reference, giữ giá trị L1 ở field riêng.
Mỗi segment ghi `internal_delay_s`, required guard và total guard. Không
nhầm số đo đang tham chiếu với giá trị thực tế đã áp.

## 6. Số liệu thực và đối chứng DCE

| Đại lượng | Giá trị [s] |
|---|---:|
| PG đầu | 4,333081233451953e-7 |
| PG cuối | 4,333096006705459e-7 |
| Mean L0 dùng cho RRF | **4,333088620078706e-7** |
| L1 `referenceReplica/timeDelay`, chỉ đối chứng | 4,333092e-7 |
| L0 − L1 | −3,379921294e-13 (−0,338 ps) |

Hai node đều peak valid. Các cặp dưới giữ cùng raw/bias L0, receiver,
PG nominal L0, layout/estimator; reference pair còn giữ nguyên EAP/RSL và
RRF gain/AUX coefficients. Tất cả dùng ROI 18.145, 20 range block ×1.000,
azimuth [0,6000), [10500,16500).

| Input / block | Fine DC/L1 RMSE [Hz] | Polynomial/L1 RMSE [Hz] | Fit RMS [Hz] |
|---|---:|---:|---:|
| Default, none / 1 | 1,369272109 | 1,341847112 | 1,439333385 |
| Default, L0 / 1 | 1,379920646 | 1,350746815 | 1,419224021 |
| Default, none / 2 | 1,444490852 | 1,400545292 | 1,215455972 |
| Default, L0 / 2 | 1,483952311 | 1,439163593 | 1,207396981 |
| Reference, L1 / 1 | 1,381625485 | 1,352814681 | 1,419865713 |
| Reference, L0 / 1 | 1,381625624 | 1,352814805 | 1,419865659 |
| Reference, L1 / 2 | 1,487421013 | 1,443421129 | 1,208617096 |
| Reference, L0 / 2 | 1,487420879 | 1,443420987 | 1,208617138 |

Fine DC L0−none: max **0,146240050 / 0,148243972 Hz**, RMS
**0,073087518 / 0,075749234 Hz**. Fine DC L0−L1 với reference inputs giữ
nguyên: max **7,21787e−7 / 6,22485e−7 Hz**, RMS
**3,10945e−7 / 2,74971e−7 Hz**. Tất cả chọn data, ambiguity 0, RMS<20 Hz.

Bù delay làm đúng gốc range và giảm phụ thuộc L1; **không** tuyên bố nó
làm giảm RMSE với L1. Cặp none/L0 cho RMSE tăng nhẹ; không đảo dấu hoặc
tune delay để làm RMSE giảm. Geometry và DCE residual còn các nguồn khác.

## 7. Tái chạy và kiểm thử

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/analyze_calibration.py \
  --output output/sentinel-1/internal_delay

# Hai lượt cùng default correction, cùng PG, chỉ đổi delay.
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py --l0-pg-drift \
  --internal-delay-source none --minimum-far-range-discard-samples 25 \
  --output output/sentinel-1/internal_delay/default_none

MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py --l0-pg-drift \
  --minimum-far-range-discard-samples 25 \
  --output output/sentinel-1/internal_delay/default_l0

# Reference pair: chạy lần lượt với source l1 rồi l0 và output tương ứng.
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py \
  --l1-input-corrections --l0-raw-analysis --l0-pg-drift \
  --internal-delay-source l1 --minimum-far-range-discard-samples 25 \
  --output output/sentinel-1/internal_delay/reference_l1
```

Thêm lượt `default_cli` chỉ truyền `--output`, kiểm đường mặc định đo delay
L0 nhưng **không** áp PG drift. Các output JSON/PNG ở thư mục trên, gitignored.
Lượt này Fine DC/L1 RMSE **1,374931908 / 1,487744452 Hz**, fit RMS
**1,418090178 / 1,205840656 Hz**; không so trực tiếp với các cặp đã bật PG
để quy khác biệt cho delay. `verification.json` lưu hash JSON, các gate
và kiểm tra cùng ROI/config/correction metadata cho các cặp đối chứng.

Kiểm thử thêm 7 test: mean/validity/config; forward integer/fractional
delay và convolution oracle; override thay delay L1; guard/common ROI và
guard kết hợp fractional SWST. Oracle calibration L0 thật bổ sung kiểm tra
delay bằng dense Fourier peak search độc lập cho cả hai sequence.
Suite: **132 test, 126 đạt, 4 skip, 2 lỗi thiếu XML S6 cũ**.

Giới hạn: adapter vẫn yêu cầu calibration đủ/đúng và PG QC hiện có hoàn tất;
chưa tách reader delay thành đường có thể tiếp tục khi PG correction QC
không đủ node. Chưa repair/drop sequence lỗi, chưa hỗ trợ hardware delay âm,
chưa chuyển notebook S6 hoặc audit/check scripts sang default nguồn mới;
chưa chứng minh full IPF parity hay cải thiện radiometry toàn pipeline.
