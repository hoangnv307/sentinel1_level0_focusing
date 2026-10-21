# Raw data analysis từ L0 — DAD §4.1–4.1.1

Triển khai ngày 21/10/2026 cho S1A/S4/VV `product_1`. Đây là bước 1 của
đề xuất sau [audit DCE](sentinel1_dce_audit.md): tạo thống kê và bias từ
L0, với API cấu hình tường minh và các test có đối chứng.

## Luồng và phạm vi

Đọc header → chọn ECHO S4/RX V → loại packet Error Flag → chọn subset
theo cấu hình → decode BAQ → thống kê I/Q → chọn tham số correction →
range compression/DCE. Calibration/noise không được đưa vào thống kê;
chưa có blackfill do ghép range ở giai đoạn này. Core không suy diễn mẫu
zero là blackfill; caller phải cung cấp echo hợp lệ.

Mặc định chọn toàn bộ echo hợp lệ của acquisition, cùng range width,
receiver gain, range decimation và instrument configuration. Hai chunk
13/14 có SWST khác nhau nhưng cùng một analysis scope, nên dùng cùng
bias. Nếu các thuộc tính trên đổi, API từ chối và yêu cầu chia scope;
không âm thầm trộn các chế độ thu.

Batch chỉ giới hạn RAM. Batch 256 dòng **không** có nghĩa chỉ phân tích
256 dòng; scope được xác định bằng các tham số chọn mẫu riêng.

## Đối chiếu công thức với DAD

Đã đọc trực tiếp PDF trang 29–31, trang in 4-2–4-4 để kiểm tra dấu căn
và dấu cận, thay vì chỉ dùng text extraction.

| Nội dung | Cách thực hiện | DAD |
|---|---|---|
| Mean I/Q | Tổng trên các mẫu thuộc subset; tích lũy float64 | Eq. 4-1–4-4 |
| Standard deviation | Population std, ddof=0; merge moments giữa batch bằng công thức Chan | Eq. 4-5–4-6 |
| Gain imbalance | σI/σQ; hai cận 1±3/sqrt(NM) | Eq. 4-7–4-9 |
| Correlation từng dòng | Center riêng mỗi dòng rồi tính covariance chuẩn hóa | Eq. 4-10–4-14 |
| Fisher transform | zk=atanh(ck), mean và population std của z; mỗi dòng valid cùng trọng số | Eq. 4-15–4-17 |
| Quadrature | θ=asin(tanh(mean(z))); trả radians | Eq. 4-18, 4-21 |
| Cận quadrature | asin(tanh(mean(z)±std(z))), là cận dispersion theo mô tả DAD, không tự chia std(z) cho sqrt(N) | Eq. 4-19–4-23 |
| Kiểm tra null | Bias ±3σ/sqrt(NM), gain quanh 1; correlation quadrature trong [−3σminus,+3σplus] | §4.1 bước 5, với diễn giải dấu ghi dưới đây |

Các điểm cần diễn giải rõ:

1. Cận trên I-bias trong PDF in dấu âm như cận dưới; dùng cặp đối xứng
   quanh 0, nhất quán với Q-bias.
2. Eq. 4-12 đặt ngoặc khác Eq. 4-13 và công thức covariance. Dùng
   `Σ(I-mean_line(I))²`, tương đương `ΣI²−(ΣI)²/M`. Test có DC offset lớn
   kiểm tra lỗi này.
3. DAD in cận null quadrature với `3σminus` dương, dù Eq. 4-23 định nghĩa
   σminus không âm. Triển khai cận dưới âm; có test θ âm và I/Q lý tưởng.
4. DAD mô tả significance TRUE khi ngoài cận, trong khi
   [Product Specification, Table 6-26](https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Specification-18052021.pdf)
   mô tả TRUE khi trong acceptable range. Vì vậy API đặt tên rõ
   `outside_null_bounds`, không xuất giả cờ IPF cùng tên nhưng khác nghĩa.
5. Không tự thay số đo bằng mặc định chỉ vì `outside_null_bounds=True`.
   Phần chữ DAD về thay bằng predefined không nhất quán với bias lớn nhưng
   vẫn được L1 hiện có dùng để correction. AUX hiện có không chứa bộ
   predefined raw statistics để áp quy tắc đó. API nhận từng predefined
   value tường minh; nếu caller cung cấp thì dùng nó và ghi nguồn.

Các hệ số 3 và ddof=0 là lựa chọn cố định trong công thức DAD, không biến
thành tham số tuning theo L1. Các diễn giải trên đã được kiểm tra về toán
học và trường hợp cụ thể, nhưng chưa chứng minh trùng logic QC của IPF.

## API và các configurable parameters

Core: `src/sentinel1_processing/common/raw_data_analysis.py`.

`analyze_batches(batches)` nhận các array complex `(lines, range_samples)`
đã chọn. Trả mean/std, gain, quadrature radians, các cận, kiểm tra null,
số mẫu và trạng thái quadrature. Không đọc L1, AUX hoặc chọn subset ẩn.

`analyze_echoes(...)` trong demo là lớp chọn dữ liệu L0:

| Đầu vào API | Vai trò |
|---|---|
| `azimuth_start_line` | Bắt đầu trên chỉ số echo hợp lệ, không phải packet number |
| `azimuth_step` | Bước lấy dòng xuyên qua ranh giới chunk |
| `max_azimuth_lines` | Giới hạn số dòng, None là không giới hạn |
| `range_start_sample`, `range_stop_sample` | ROI range, stop exclusive; None là đến hết echo |
| `batch_lines` | Kích thước giải mã/tích lũy, không đổi estimator/scoping |

Các đầu vào chọn mẫu đều bắt buộc khi gọi trực tiếp API. CLI có mặc định
công khai: bắt đầu 0, bước 1, toàn range, không giới hạn dòng, batch 256.
DAD cho phép phân tích subset nhưng không công bố N/M dùng cho sản phẩm
L1 này; chọn toàn cảnh là cấu hình của demo, không nhận là lựa chọn IPF.

`correction_parameters(analysis, *, ...)` nhận:

- `correct_iq_bias`, `correct_iq_gain`, `correct_iq_quadrature`: bắt buộc,
  tương ứng cờ AUX_PP1. Hàm không bật correction từ kết quả significance.
- `predefined_i_bias`, `predefined_q_bias`, `predefined_gain_imbalance`,
  `predefined_quadrature_rad`: giá trị cấu hình theo §4.1 khi caller cung
  cấp; None nghĩa là chưa có, không tự coi là 0/1.

Kết quả phân biệt `selected_statistics`, `sources`, `enabled` và
`applied_parameters`. Correction tắt trả identity tương ứng 0/0/1/0,
nhưng số đo vẫn được báo. Pipeline S1 hiện chỉ áp bias; nếu bật gain/
quadrature correction thì demo DCE từ chối vì operator đó chưa triển khai.
Việc tính và chọn các thống kê vẫn có thể chạy riêng.

Dữ liệu non-finite, batch rỗng hoặc width thay đổi bị từ chối. Các dòng
variance 0 hoặc correlation hoàn hảo ±1 không có Fisher z hữu hạn được
đếm riêng. Nếu không đủ hai dòng correlation valid thì θ/cận/flag là
None và có status, không ghi NaN hay giả θ=0. Nếu cần correction từ số đo
không xác định, caller phải cung cấp predefined value hợp lệ.

## Các test cụ thể

`test/test_raw_data_analysis.py` có **17 test**, trong đó 1 test integration
dùng raw S4/VV thật (skip có lý do nếu máy khác không có dữ liệu ngoài repo).

| Test / nhóm đối chứng | Kết quả phải chứng minh |
|---|---|
| Tín hiệu tạo bằng hai vector trực giao | Mean .25/−.125, std 2/4, gain .5, θ=asin(.2), cận dùng đúng sqrt(NM) |
| Các dòng có correlation −.4/.1/.2/.8 | Fisher mean đúng, khác lấy trung bình correlation trực tiếp |
| Offset theo dòng cỡ 10⁶ | Quadrature không đổi sau line centering |
| Chia cùng dữ liệu thành batch khác nhau | Mean/std/gain/θ và flag không đổi trong sai số float64 |
| I/Q lý tưởng | Không báo departure khỏi null |
| Quadrature âm | Cận null dưới mang dấu âm; phát hiện departure đúng |
| Dòng constant / correlation ±1 | Có số dòng bị loại, không phát sinh Fisher infinity/NaN |
| Input rỗng, real-only, non-finite, width khác | Từ chối rõ ràng |
| Cờ AUX tắt / bật, predefined override | Áp identity hoặc số đo/giá trị cấu hình đúng; không tự bật gain/phase |
| Số đo không xác định | Chỉ yêu cầu predefined khi correction đó bật |
| Cờ sai kiểu, predefined NaN/list/gain 0/θ=π/2 | Từ chối |
| Echo giả có noise/calibration/Error Flag | Không đưa vào mẫu; chọn đúng dòng và ROI qua hai chunk |
| Batch 2 so với 3 trong workflow | Không làm đổi analysis scope |
| Cấu hình selector sai / receiver gain thay đổi | Từ chối, yêu cầu chia scope |
| Start/step/line limit có cấu hình | Thực sự chọn đúng số dòng yêu cầu |
| DCE giả lập hai chunk | Cùng bias trên cả hai chunk, AUX tắt bias thì không trừ |
| 16 echo thật rải qua hai chunk | Đối chiếu trực tiếp mean/std với NumPy, θ với corrcoef+Fisher; trừ bias cho mean residual <10⁻¹² trong float64 |

## Kết quả chạy thật

<!-- RDA_RESULTS_START -->
| Cấu hình | N dòng × M range samples | Mean I | Mean Q | Gain σI/σQ | θ [rad] |
|---|---|---|---|---|---|
| Toàn cảnh, batch 256 | 16500 × 21154 | 0.2908047485 | 0.3617876438 | 0.9822568466 | 0.0018263420 |
| Toàn cảnh, batch 128 | 16500 × 21154 | 0.2908047485 | 0.3617876438 | 0.9822568466 | 0.0018263420 |
| 256 dòng đầu | 256 × 21154 | 0.2883251961 | 0.3642456462 | 0.9824662762 | 0.0024988482 |
| 256 dòng rải, step 64 | 256 × 21154 | 0.2910976960 | 0.3629951165 | 0.9821531993 | 0.0018763612 |

Toàn cảnh gồm **349,041,000 complex samples**, std I/Q = 2.9320614080/2.9850251674. Không có Error Flag, constant line hoặc Fisher-invalid line bị loại trong dữ liệu thật này.
Sai khác lớn nhất giữa batch 128 và 256 của sáu thống kê là **4.441e-16**; cờ ngoài null giống nhau.
Cận bias toàn cảnh là ±0,0004708213 cho I và ±0,0004793260 cho Q; gain bounds [0,9998394231; 1,0001605769]. Bias/gain ngoài null, quadrature trong null theo quy ước đã nêu.
AUX bật bias, tắt gain/quadrature: tham số thực áp là bias đo được, gain=1 và θ=0. Không áp θ đo được chỉ vì có số thống kê.

L1 tham chiếu có bias 0,2893930078 + 0,3621337116j và gain 0,9822294712. Chênh bias L0−L1 là +0,0014117408 −0,0003460678j. Không biết chính xác subset IPF nên không dùng các chênh lệch này làm ngưỡng pass/fail.

| DCE | Fine RMSE/L1 B1 / B2 [Hz] | Fit RMS B1 / B2 [Hz] |
|---|---|---|
| Cơ bản, một bias L0 chung | 1.362086 / 1.451192 | 1.425097 / 1.213541 |
| Hiệu chỉnh tham chiếu, thay riêng bias bằng L0 | 1.371553 / 1.495293 | 1.417343 / 1.205824 |

Cả hai lần chạy giữ 20/20 điểm, ambiguity 0 và chọn data DC. So với control AUX-RRF dùng bias L1 ở báo cáo RRF trước, thay bias bằng L0 làm Fine DC đổi tối đa khoảng 4,1×10⁻⁵ Hz trên cùng tọa độ. Đây là tiến bộ về nguồn dữ liệu và tính nhất quán scope, chưa phải giảm residual với L1.
Layout/time alignment của DCE và PG/EAP/delay của chế độ tham chiếu vẫn phụ thuộc L1 như trước; không gán việc thay bias cho toàn bộ tiền xử lý độc lập L0/AUX.
<!-- RDA_RESULTS_END -->

Không dùng mức khớp L1 để đổi subset hoặc bật correction bị AUX tắt.
Product Spec không ghi rõ đơn vị quadrature ở bảng metadata đã đối
chiếu, nên JSON báo cả hai cách diễn giải literal L1 là radian/degree;
không chọn cách diễn giải đó để đưa số L1 vào pipeline.

## Tái chạy

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/analyze_raw_data.py
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  -m unittest discover -s test -p 'test_raw_data_analysis.py'
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py \
  --output output/sentinel-1/raw_analysis/dce_basic_global_bias
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py --l1-input-corrections \
  --l0-raw-analysis --output output/sentinel-1/raw_analysis/dce_reference_l0_bias
```

CLI raw analysis cung cấp đầy đủ các option cùng tên selector, viết bằng
dấu gạch nối. Demo DCE cung cấp `--analysis-start-line`, `--analysis-step`,
`--analysis-lines`, `--analysis-range-start`, `--analysis-range-stop`,
`--batch-lines`. Với `--l1-input-corrections` mà không có
`--l0-raw-analysis`, bias L1 vẫn được giữ như một control nghiên cứu;
thống kê L0 vẫn được tính và ghi riêng. JSON luôn ghi bias thực sự đã áp.

Kết quả tại `output/sentinel-1/raw_analysis/` (gitignored), gồm thống kê,
đối chiếu L1, cờ cấu hình, SHA-256 input/source, log test và JSON/PNG DCE.
Full suite: **73 test, 67 qua, 4 skip, 2 lỗi thiếu annotation S6 cũ**.

Notebook focus S6 vẫn dùng helper mean cũ; bước này tích hợp vào demo DCE
S4, không tự chuyển scene của notebook. Internal calibration, PG và EAP
tự dựng từ L0/AUX vẫn là các bước tiếp theo; không tuyên bố pipeline đã
độc lập hoàn toàn L1 hoặc tái hiện IPF bit-exact.
