# Receiver gain compensation từ L0/AUX — DAD §9.2.3

Triển khai và đối chứng ngày 21/10/2026 trên `product_1`, S1A/S4/VV.
Đây là bước tiếp theo của [hiệu chỉnh I/Q](sentinel1_raw_data_correction.md).
Receiver compensation nay hoạt động trong default DCE, dùng timing L0,
hệ số AUX_INS và cờ AUX_PP1; không cần cờ hoặc gain từ annotation L1.
Layout của demo DCE vẫn giữ ngoại lệ suy từ L1 đã được người dùng cho phép.

## 1. Công thức đã kiểm tra trực tiếp trên PDF

Nguồn: DAD Issue/Revision 2/5, **PDF trang 126–127**, trang in **9-9–9-10**,
Figure 9-2, Eq. **9-9–9-16**. Không chỉ dựa vào text extraction.

```text
ts_us = sample_index / sample_rate_hz * 1e6
delta_us = (SWST_downlink_s + suppressed_s - pulse_length_s) * 1e6
t1_us = ts_us + delta_us

trend_dB(t1) = (p0 + p1*t1 + p2*t1^2 + p3*t1^3 + p4*t1^4)
               / (q0 + q1*t1 + q2*t1^2)
overshoot_dB(ts) = a*exp(-b*ts)*sin(d*ts + phi) + c
GV(ts) = 10^((trend_dB(t1) + overshoot_dB(ts))/10)

complex_corrected(ts) = complex_after_iq(ts) / sqrt(GV(ts))
```

`ts=0` là mẫu đầu tiên còn lại sau suppression, không phải mốc bắt đầu
TX hoặc SWST downlink. Trend lấy mốc cuối TX, overshoot lấy mốc RX.
SWST truyền vào API là giá trị header **chưa cộng suppression**; cộng đúng
một lần. Tính lại vector theo SWST từng chunk; không dùng chung vector
chunk 13 cho chunk 14. Không đưa `Rank*PRI` hoặc SWST bias dùng cho trục
slant range vào thời gian tương đối của mô hình receiver này.

Theo convention L0 hiện tại, suppression là `320/(8*F_REF)` =
**1,065679925490 µs**; DAD Figure 9-2 minh họa khoảng 1,06 µs. Đây là
convention giải mã/timing của demo đang có, không suy từ kết quả DCE L1.
Core nhận `suppressed_s` làm đầu vào, không chôn giá trị này trong mô hình.

### Lỗi mẫu số cũ đã sửa

Eq. 9-16 trong PDF có **căn bậc hai của GV ở mẫu số**. Bản pdftotext chỉ
hiện GV, khiến nhánh tham chiếu cũ chia complex raw trực tiếp cho GV.
Điều đó áp dụng sai hệ số biên độ dù các vector gain vẫn hữu hạn và các
test Doppler có thể vẫn pass.

`calculate_gain` trả **power GV** theo Eq. 9-14–9-15, dùng mẫu số 10 khi
đổi dB. Pipeline chia cho **sqrt(GV)**, tương đương amplitude divisor
`10^((trend_dB+overshoot_dB)/20)`. Không biến hai cách chuẩn hóa thành
option tuning. Test 10 dB yêu cầu power gain 10 và amplitude divisor
sqrt(10), đồng thời kiểm tra khôi phục từng mẫu complex.

Các bảng reference/audit/RRF cũ có RX dùng công thức sai đã được gắn
đính chính và giữ nguyên để không mất lịch sử. Không coi những RMSE đó
là kết quả của pipeline hiện tại.

## 2. API và cấu hình

Core: `src/sentinel1_processing/common/receiver_gain_compensation.py`.
Export qua `sentinel1_processing.common.receiver_gain_compensation`.

```python
gv = calculate_gain(
    n_samples,
    sample_rate_hz=sample_rate_hz,
    swst_s=swst_downlink_s,
    pulse_length_s=pulse_length_s,
    suppressed_s=suppressed_s,
    gain_trend_coefficients=trend_coefficients,
    gain_overshoot_coefficients=overshoot_coefficients,
    enabled=correct_rx_variation,
)
corrected = iq_corrected / np.sqrt(gv)[None, :]
```

Hàm thuần không đọc L0, AUX, L1 hoặc trạng thái notebook. Trả vector
float64; không sửa hệ số đầu vào. Mọi tham số đều bắt buộc, bằng keyword
trừ `n_samples`. Flag tắt trả vector 1 mà không đánh giá các hệ số; có thể
truyền None cho hệ số khi tắt. Timing/count vẫn phải hợp lệ.

| Tham số | Nguồn trong demo |
|---|---|
| `enabled` | AUX_PP1 `productList/product[productId=SM_SL1__1]/commonProcParams/correctRxVariationFlag` |
| 8 hệ số trend | AUX_INS `swathParamsList/swathParams[swath=S4]/rxVariationCorrectionParamsList/rxVariationCorrectionParams[rxPolarisation=V]/gainTrendCoefficients`, thứ tự p0…p4,q0…q2 |
| 5 hệ số overshoot | Cùng record AUX_INS, `gainOvershootCoefficients`, thứ tự a,b,c,d,phi |
| `sample_rate_hz` | Range Decimation trong header L0 qua decoder utilities |
| `swst_s`, `pulse_length_s` | Header L0 SWST/Tx Pulse Length, đơn vị seconds |
| `suppressed_s` | Convention L0 nêu trên, caller truyền tường minh |
| `n_samples` | Hai lần Number of Quads của chunk |

Đã đọc XSD AUX_PP1 đi kèm: `correctRxVariationFlag` yêu cầu thực hiện
correction khi và chỉ khi flag true. Với `SM_SL1__1` hiện tại flag là
**true**. Parser chọn đúng product, yêu cầu literal true/false; không dùng
default ẩn khi thiếu cờ. Hệ số AUX_INS chọn S4 và RX V, không dùng bộ H.

Core kiểm tra flag boolean, count nguyên dương, timing scalar thực hữu
hạn, sample rate/pulse length dương, SWST/suppression không âm. Khi bật,
yêu cầu đúng 8/5 hệ số thực hữu hạn; từ chối mẫu số zero/non-finite,
gain không dương, overflow và underflow về zero. Pipeline còn kiểm tra
raw sau RX/PG để bắt overflow theo complex dtype trước khi chạy FFT.

## 3. Tích hợp và provenance

Thứ tự xử lý giữ đúng:

```text
BAQ decode → I/Q correction → chia sqrt(GV) → PG nếu reference mode
           → range compression → EAP/RSL nếu reference mode → DCE
```

`compress_echoes(..., correct_rx_variation=None)` lấy cờ AUX; boolean
tường minh ghi đè cho đối chứng. Core gain dùng chung ở default và
reference, tránh áp RX hai lần. Một vector cho mỗi chunk được dùng lại
qua các batch; batch size không đổi mốc thời gian receiver.

`ReferenceInput.receiver_gain()` nay delegate cùng core và trả power GV.
Cờ receiver của helper lấy từ AUX; cờ L1 được giữ ở
`rx_enabled_annotation` chỉ để chẩn đoán. `audit_doppler_centroid.py`
truyền override false cho case `no_rx`, thay vì chỉ đổi thuộc tính object
tham chiếu mà pipeline không còn dùng để quyết định RX.

Đã bỏ chặn `correct_iq_gain`/`correct_iq_quadrature` cũ trong constructor
`ReferenceInput`. Helper giữ bias tham chiếu nguyên giá trị; caller quyết
định có áp bias qua cờ I/Q. Test bảo vệ trường hợp AUX bật gain/quad và
tắt bias mà constructor vẫn hoạt động, cũng như AUX/L1 có cờ RX khác nhau.

Mỗi `segments[*].receiver_compensation` trong JSON ghi enabled, nguồn cờ,
nguồn hệ số, SWST header, pulse length, suppression, delta từ cuối TX,
power gain first/last/min/max và amplitude divisor min/max. AUX gốc và
override được lưu riêng, không sửa AUX để giả rằng cấu hình gốc khác đi.
`post_iq_correction_analysis` vẫn là thống kê sau **chỉ I/Q**, trước RX.

## 4. Các lượt chạy S4/VV

Ba lượt đều chạy toàn bộ **16.500 echo × 21.154 mẫu**, cùng raw analysis
L0 toàn acquisition, bias `0,290804748523 + j*0,361787643809`. I/Q gain
và quadrature tiếp tục tắt theo AUX; không bật thêm để tìm RMSE thấp hơn.

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py \
  --output output/sentinel-1/receiver_compensation/aux_default

MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py --no-correct-rx-variation \
  --output output/sentinel-1/receiver_compensation/rx_disabled

MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py --l1-input-corrections --l0-raw-analysis \
  --output output/sentinel-1/receiver_compensation/reference_l0_bias
```

Có cả `--correct-rx-variation` để bật tường minh. Mỗi thư mục chứa
`doppler_centroid.json` và PNG. Memmap tạm tự xóa; `output/` gitignored.

### Timing và gain tính từ AUX/L0

TX pulse length chung **51,498982399290 µs**.

| Chunk | SWST header [µs] | Delta từ cuối TX [µs] | Power GV min…max | Amplitude divisor min…max |
|---|---:|---:|---:|---:|
| 13 | 90,289731687115 | 39,856429213315 | 0,943958764337…1,026669397644 | 0,971575403320…1,013246957875 |
| 14 | 91,781683582801 | 41,348381109001 | 0,944980300180…1,026921561208 | 0,972100972214…1,013371383654 |

Khi tắt RX, vector gain/divisor bằng 1; vẫn ghi timing thật để đối chiếu.

### Doppler centroid

Hai lượt RX bật/tắt dùng cùng ROI 18.170 mẫu, 20 block range × 1.000,
azimuth `[0,6000)` và `[10500,16500)`, cùng Fine DC L1 nội suy và cùng
cấu hình estimator. Reference mode có delay guard nên ROI 18.145 mẫu;
không dùng chênh lệch giữa reference/default để quy riêng cho receiver.

| Chế độ / block | Fine DC/L1 RMSE [Hz] | Polynomial/L1 RMSE [Hz] | Fit RMS [Hz] |
|---|---:|---:|---:|
| RX tắt / 1 | 1,362085958 | 1,330690448 | 1,425097498 |
| RX bật / 1 | 1,362630627 | 1,331278012 | 1,424903422 |
| RX tắt / 2 | 1,451192372 | 1,409946472 | 1,213540713 |
| RX bật / 2 | 1,451727677 | 1,410532531 | 1,213715114 |
| Reference + bias L0 / 1 | 1,371052731 | 1,341372647 | 1,417464328 |
| Reference + bias L0 / 2 | 1,494700036 | 1,451162999 | 1,205662463 |

Tác động RX riêng lên Fine DC: max **0,001890749 Hz** / **0,002561995 Hz**,
RMS **0,001032906 Hz** / **0,001320982 Hz** cho block 1/2. Lượt RX tắt
khớp từng Fine DC với default trước khi tích hợp RX. Cả ba lượt đều có
ambiguity 0, RMS dưới ngưỡng AUX 20 Hz và chọn data estimate.

Reference có bias L0 trước sửa dấu căn cho Fine RMSE 1,371552666 Hz /
1,495292833 Hz; sau sửa là các số trong bảng. Đây là thay đổi công thức,
không phải tuning estimator, range layout hoặc bias theo L1.

**Chốt cấu hình:** bật RX theo AUX_PP1. RMSE với L1 tăng rất nhẹ so với
RX tắt, nhưng việc hiệu chỉnh được quyết định bởi DAD/cấu hình, không
bởi tối thiểu hóa RMSE của một sản phẩm. Vẫn chưa có bằng chứng parity
bit-exact với IPF hoặc calibration radiometric tuyệt đối.

## 5. Kiểm thử

`test/test_receiver_gain_compensation.py` có **10 test**:

| Đối chứng | Nội dung |
|---|---|
| 10 dB và amplitude | Power GV=10, complex divisor sqrt(10), khôi phục mẫu complex |
| Công thức đầy đủ | Rational bậc 4/2 và overshoot đủ năm hệ số, khớp công thức scalar độc lập |
| Đơn vị/suppression | Tính µs và cộng suppression đúng một lần |
| SWST | Đổi SWST chỉ dịch trend; overshoot vẫn bắt đầu từ mẫu RX đầu tiên |
| Cờ tắt | Identity kể cả hệ số thiếu/singular; không đánh giá mô hình khi tắt |
| Một mẫu/input read-only | Shape đúng và không sửa hệ số đầu vào |
| Timing/count/flag lỗi | Từ chối kiểu, miền giá trị và non-finite |
| Hệ số lỗi | Từ chối shape sai, complex, NaN/Inf |
| Mẫu số | Zero ở bất kỳ sample hoặc polynomial overflow đều bị từ chối |
| Gain overflow/underflow | Không trả Inf hoặc zero làm mẫu số correction |

Thêm **3 integration test** trong `test/test_raw_data_analysis.py`:
hai chunk SWST khác nhau, so với convolution độc lập, batch 2/3 và
override hai chiều; parser đúng product và cờ bắt buộc; raw overflow phải
bị chặn trước FFT. Test thứ tự I/Q → RX → PG → FFT hiện có được sửa dùng
power gain 4/amplitude divisor 2 để phát hiện lỗi chia GV hoặc áp hai lần.

Thêm **1 test reference** trong `test/test_sentinel1_dce_demo.py`: không
chặn cờ gain/quad; cờ receiver theo AUX kể cả khi khác L1. Tone test hiện
có cũng đổi sang amplitude sqrt(GV), không tạo input giả theo mẫu số sai.

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python -m unittest \
  discover -s test -p 'test_receiver_gain_compensation.py'
```

Tổng cộng **14 test mới**. Full suite: **100 test, 94 đạt, 4 skip, 2 lỗi**
do thiếu XML S6 cũ; không đổi các lỗi fixture đó thành skip. File raw
analysis có 23 test, DCE demo 16 test. Không thêm dependency mới.

## 6. Phạm vi còn lại

Receiver compensation đã độc lập L1 trong default DCE, nhưng PG drift,
EAP/RSL ở default vẫn chưa có; các bước đó còn ở nhánh tham chiếu. Chưa
bổ sung spurious filtering §9.2.2, chưa xác minh chính xác RRF Extended
Tapered của IPF, chưa đổi notebook focus S6. Chưa tái chạy toàn bộ ma trận
audit/RRF cũ sau sửa căn; các báo cáo đó được đánh dấu lịch sử rõ ràng.
