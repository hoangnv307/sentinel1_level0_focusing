# Hiệu chỉnh raw I/Q theo DAD §9.2.1

Triển khai và đối chứng ngày 21/10/2026 trên S1A/S4/VV `product_1`.
Đây là bước tiếp theo của [raw data analysis](sentinel1_raw_data_analysis.md):
dùng thống kê đã chọn để hiệu chỉnh từng mẫu BAQ-decoded trước receiver
variation, PG, range compression và DCE.

Cập nhật tiếp theo: [receiver compensation §9.2.3](sentinel1_receiver_gain_compensation.md)
đã được đưa vào default L0/AUX. Các bảng DCE trong tài liệu này là lượt
raw-I/Q trước khi bật RX mặc định; có thể tái chạy bằng cờ tắt RX.

## Công thức, dấu và đơn vị

Đã đọc trực tiếp DAD Issue 2/5, PDF trang **124–125**, trang in **9-7–9-8**,
Eq. **9-1–9-6**, và đối chiếu định nghĩa thống kê ở §4.1.

Với `mu_I`, `mu_Q` là mean đo được, `rho = sigma_I / sigma_Q`, và
`theta` là quadrature departure tính bằng **radians**:

```text
I1 = Iin - mu_I                 Q1 = Qin - mu_Q
I2 = I1                        Q2 = rho * Q1
Iout = I2                      Qout = Q2/cos(theta) - I2*tan(theta)
```

Thực hiện đúng thứ tự **bias → gain → quadrature**. Nếu tắt một bước,
dùng identity của bước đó: bias 0/0, gain 1, theta 0. Ví dụ bật quadrature
nhưng tắt gain thì dùng Q1 trực tiếp trong Eq. 9-6; không tự bật gain.

DAD Eq. 9-1–9-2 in dấu cộng trong khi §4.1 định nghĩa mu là mean và
§9.2.1 yêu cầu output zero-mean. API nhận **mean**, vì vậy trừ mu; không
diễn giải mu là offset bù đã đổi dấu. Eq. 9-4 được đọc là `rho * Q1`.
Dấu trừ trong Eq. 9-6 được kiểm tra bằng cả theta dương và âm.

Mô hình thuận dùng để kiểm thử độc lập là:

```text
Iin = Iideal + mu_I
Qin = (Qideal*cos(theta) + Iideal*sin(theta))/rho + mu_Q
```

Với Iideal/Qideal zero-mean, cùng variance và không tương quan, mô hình
này tạo đúng rho và theta mà §4.1 ước lượng. Áp công thức nghịch phải
khôi phục cả hai kênh, thay vì chỉ làm một chỉ số gần giá trị mong muốn.

DAD §4.1 và §9.2.1 nói gain/non-orthogonality không cần mặc định cho
Sentinel-1 do demodulation số, nhưng vẫn phải hỗ trợ qua cấu hình.
Vì vậy thống kê lệch khỏi 1/0 không tự bật correction.

## API và nguồn cấu hình

Core nằm tại `src/sentinel1_processing/common/raw_data_correction.py`.

`apply(radar_data, *, i_bias, q_bias, gain_imbalance, quadrature_rad,
correct_iq_bias, correct_iq_gain, correct_iq_quadrature)` nhận đầy đủ
statistic và cờ tường minh; không đọc AUX/L1 hoặc tự đo lại trên batch.

| Đầu vào | Quy ước / nguồn |
|---|---|
| `radar_data` | Array complex 2D hữu hạn, không rỗng; echo hợp lệ đã BAQ decode |
| `i_bias`, `q_bias` | Mean I/Q của analysis scope hoặc giá trị predefined do caller chọn |
| `gain_imbalance` | Tỷ số sigma_I/sigma_Q, scalar thực dương |
| `quadrature_rad` | Scalar thực, radians, trị tuyệt đối nhỏ hơn pi/2 |
| `correct_iq_bias` | Cờ boolean; demo mặc định từ AUX_PP1 `commonProcParams/correctIQBiasFlag` |
| `correct_iq_gain` | Cờ boolean; demo mặc định từ `commonProcParams/correctIQGainImbalanceFlag` |
| `correct_iq_quadrature` | Cờ boolean; demo mặc định từ `commonProcParams/correctIQOrthogonalityFlag` |

Các XML path trên thuộc product `SM_SL1__1` trong AUX_PP1.
Operator trả array độc lập, giữ complex dtype và không sửa input, kể cả
input read-only. Từ chối statistic không hữu hạn, vector, gain không dương,
theta ngoài miền hoặc output overflow. Statistic `None` chỉ được phép khi
bước tương ứng tắt. Không thêm in-place API vì demo xử lý từng batch nhỏ.

`raw_data_analysis.correction_parameters(...)` tiếp tục chịu trách nhiệm
chọn measured/predefined và xuất `selected_statistics`, `sources`,
`enabled`, `applied_parameters`. Caller cung cấp các predefined value;
không suy ra từ L1 trong chế độ L0 hoặc tự thay theo significance flag.

`compress_echoes(...)` nhận thêm ba cờ correction và bốn predefined value
bằng keyword arguments. Cờ `None` nghĩa là giữ AUX; các cờ boolean ghi đè
chỉ dùng cho thử nghiệm. Một bộ statistic được chọn cho toàn analysis
scope, dùng chung hai chunk và mọi batch. Thứ tự thực thi:

```text
BAQ decode → I/Q correction → receiver variation (L0/AUX)
           → PG (reference mode) → range compression → EAP/RSL (reference mode) → DCE
```

Sau operator I/Q, `compression.compress` nhận `iq_bias=0j` để không trừ
bias lần thứ hai. Chế độ L1 input giữ ngoại lệ cũ: bias từ L1 trừ khi có
`--l0-raw-analysis` hoặc predefined bias tường minh. Gain/theta vẫn từ L0
hoặc predefined, không tự đọc gain/theta L1. Nếu bật gain/theta trong
reference mode mà chưa truyền raw analysis, API tự phân tích L0 và vẫn
giữ nguồn bias L1. Test riêng bảo vệ sự phân biệt này.

CLI lưu các override, tham số thực dùng và nguồn trong JSON. Trường
`raw_data_analysis` là trước correction; `post_iq_correction_analysis`
phân tích lại đúng subset sau **chỉ I/Q correction**, trước RX/PG/range
compression. Lượt phân tích sau cần decode lại subset, không giữ toàn raw
scene trong RAM. Hai lần phân tích dùng cùng selection và batch size;
`batch_lines` không đổi phạm vi đo.

## Tái chạy trên S4/VV

Giữ mặc định AUX (bias=true, gain=false, quadrature=false):

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py \
  --no-correct-rx-variation \
  --output output/sentinel-1/raw_correction/aux_default
```

Bật cả ba hiệu chỉnh để đối chứng:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py \
  --no-correct-rx-variation \
  --correct-iq-gain --correct-iq-quadrature \
  --output output/sentinel-1/raw_correction/all_enabled
```

CLI có cặp bật/tắt `--correct-iq-bias`/`--no-correct-iq-bias`, tương tự
cho gain và quadrature. Có thể truyền `--predefined-i-bias`,
`--predefined-q-bias`, `--predefined-gain-imbalance`,
`--predefined-quadrature-rad`. Predefined không tự bật correction; theta
luôn radians. Dùng thư mục output khác nhau cho các thử nghiệm.

Mỗi lượt xuất `doppler_centroid.json` và `doppler_centroid.png`, đồng thời
tự dọn memmap tạm. Kết quả số bên dưới được giữ trong tài liệu vì
`output/` bị gitignore.

## Kết quả trước–sau

Cùng 16.500 echo × 21.154 range samples = **349.041.000 mẫu complex**,
chunk 13/14, không loại mẫu chỉ vì giá trị zero. Cùng L0/AUX, cùng subset,
layout DCE, reference L1 và cấu hình estimator giữa hai lượt.

| Thống kê | Trước correction | Sau mặc định AUX | Sau bật cả ba |
|---|---:|---:|---:|
| Mean I | 0,290804748523 | −3,81383e−8 | −3,81383e−8 |
| Mean Q | 0,361787643809 | −2,49452e−8 | −6,22211e−8 |
| Std I | 2,932061407956 | 2,932061407284 | 2,932061407284 |
| Std Q | 2,985025167386 | 2,985025164257 | 2,932061361934 |
| Gain imbalance | 0,982256846606 | 0,982256847410 | 1,000000015467 |
| Quadrature [rad] | 0,001826341952 | 0,001826341890 | −4,75778e−8 |

Residual nhỏ sau correction phù hợp với làm tròn complex64 và các
estimator khác nhau: gain dùng global std, theta dùng mean Fisher-z theo
dòng. Không yêu cầu chính xác bitwise zero cho cảnh thật không đồng nhất.

| Chế độ / block azimuth | Fine DC/L1 RMSE [Hz] | Polynomial/L1 RMSE [Hz] | RMS fit [Hz] |
|---|---:|---:|---:|
| AUX / `[0,6000)` | 1,362085958 | 1,330690448 | 1,425097498 |
| Cả ba / `[0,6000)` | 1,362995716 | 1,331785531 | 1,425468634 |
| AUX / `[10500,16500)` | 1,451192372 | 1,409946472 | 1,213540713 |
| Cả ba / `[10500,16500)` | 1,451261704 | 1,410120760 | 1,213608422 |

Sai khác Fine DC giữa hai chế độ có max lần lượt **0,009424646 Hz** và
**0,009716648 Hz**; RMS sai khác **0,004347965 Hz** và **0,003516521 Hz**.
Hai block đều ambiguity 0 và chọn data estimate vì RMS dưới ngưỡng AUX
20 Hz. Default sau tích hợp có Fine DC **trùng từng giá trị** với output
`raw_analysis/dce_basic_global_bias` trước thay đổi này.

**Chốt cấu hình:** tiếp tục theo AUX: bật bias, tắt gain/quadrature.
Operator hoạt động đúng trên dữ liệu biết trước và đưa thống kê cảnh
thật về gần 1/0, nhưng RMSE với L1 không cải thiện. Không lấy thống kê
đẹp hơn hoặc độ khớp L1 để thay cờ cấu hình của sản phẩm.

## Kiểm thử và giới hạn

`test/test_raw_data_correction.py` có **10 test**:

| Test | Bằng chứng |
|---|---|
| Khôi phục từ statistic đo | Mô hình thuận → §4.1 → §9.2.1 trả đúng từng mẫu ideal; mean 0, gain 1, theta 0 |
| Tám tổ hợp cờ | Mỗi bước chỉ áp khi bật; đúng thứ tự và không tự bật bước khác |
| Dấu theta/gain | Theta −0,7/0/+0,7 rad, gain 0,4/1/2,5; đảo đúng mô hình thuận |
| Doppler biết trước | Khôi phục tín hiệu 7 Hz và lag-1 ACCC theo từng range bin, không dùng L1 |
| Predefined | Bốn giá trị predefined thay measured đúng nguồn và phép tính |
| Tắt hết | Chấp nhận statistic undefined, trả bản sao identity |
| Input/dtype | Giữ input read-only, không alias/mutate, giữ complex64/complex128 |
| Batch | Kết quả không đổi khi chia batch |
| Input/config lỗi | Từ chối array không complex/không hữu hạn/rỗng, scalar sai, gain/theta/cờ không hợp lệ |
| Overflow | Báo lỗi thay vì trả NaN/Inf |

Thêm **3 integration test** vào `test/test_raw_data_analysis.py`, nâng file
đó từ 17 lên **20 test**: correction bằng predefined phải đi trước RX/PG/
range FFT và chỉ trừ bias một lần; thống kê sau correction dùng đúng
scope và khớp công thức độc lập; reference mode giữ bias L1 khi bật gain
và cần tự đo statistic L0.

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python -m unittest \
  discover -s test -p 'test_raw_data*.py'
```

30 test analysis/correction đều đạt, bao gồm NumPy oracle trên echo S4
thật. Full suite: **86 test, 80 đạt, 4 skip, 2 lỗi** vì thiếu XML tham chiếu
S6 cũ; hai lỗi fixture giống trước triển khai. Các test DCE/RRF hiện có
vẫn đạt. Không sửa test cũ thành skip để che lỗi fixture.

Phạm vi hoàn thiện là **§9.2.1**. Chưa bổ sung spurious filtering §9.2.2,
chưa tái tạo chính xác IPF Extended Tapered RRF, và chưa chuyển notebook
focus S6 sang luồng raw analysis S4 này. RX nay đã có ở default L0/AUX;
PG/EAP/RSL vẫn chỉ có ở nhánh reference input đã mô tả trong
[audit DCE](sentinel1_dce_audit.md). L1 tiếp tục là đối chứng sau tính toán,
ngoại trừ ngoại lệ nghiên cứu được ghi rõ cho layout/reference mode.
