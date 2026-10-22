# Tách availability của internal delay và PG correction

Ngày 21/10/2026, tiếp nối [internal delay L0](sentinel1_internal_delay.md).
Thay đổi này xử lý trường hợp PG correction không đủ node nhưng calibration
vẫn có peak tốt để đo delay. Không đổi công thức delay, normalization PG,
threshold AUX hoặc default source của CLI.

## 1. Quy tắc xử lý

Theo DAD §4.2.1.4, sequence không tìm được correlation peak hợp lệ phải
được loại khỏi phép đo delay. PG amplitude/phase được kiểm tra riêng ở
§4.2.1.8, với lựa chọn extracted/model và ngưỡng invalid fraction từ AUX.
Nội suy PG theo §9.4 cần ít nhất hai node tăng theo thời gian.

| Tình trạng | Delay | PG interpolation |
|---|---|---|
| Hai peak/PG hợp lệ | Mean của hai delay | Sẵn sàng |
| Chỉ một sequence, peak hợp lệ | Dùng delay đó | Chưa đủ hai node |
| Hai peak hợp lệ, chỉ một PG qua QC | Mean của **cả hai** delay | Chưa đủ hai node extracted |
| Peak hợp lệ nhưng tất cả PG fail QC, AUX chưa yêu cầu model | Vẫn đo bình thường | Không sẵn sàng; không tự fallback |
| Một peak bị loại, một peak tốt | Chỉ dùng peak tốt | Tùy QC và chính sách model AUX |
| Toàn bộ peak hỏng | Không sẵn sàng, `mean_s=null` | Không có extracted PG; model chỉ dùng nếu AUX yêu cầu |

Một node PG có thể qua QC khi chỉ còn một phép đo: mean bằng chính nó,
std bằng zero. Điều này không chứng minh đủ dữ liệu nội suy. Gate hai node
được kiểm riêng, không đưa thêm threshold hoặc node giả để làm lượt chạy qua.

Model PG có thể cung cấp các node correction nếu đúng chính sách AUX,
nhưng **không tạo được peak đo delay**. CLI mặc định cần delay L0 nên vẫn
dừng nếu mọi peak đều hỏng, kể cả khi model đủ node cho PG correction.

## 2. Những gì đã đổi trong API

`pre_processing/internal_calibration.validate_pg`:

- Nhận thêm keyword `measurement_valid=None`. Mặc định coi tất cả phép đo
  đầu vào là có dữ liệu; adapter truyền mask tường minh.
- Cho phép một phép đo. Số node không còn là điều kiện để thực hiện QC.
- NaN/zero ở vị trí được đánh dấu invalid không tham gia thống kê. NaN/zero
  ở vị trí được caller đánh dấu valid vẫn là lỗi dữ liệu, không âm thầm loại.
- Relative mean/std và phase unwrap chỉ tính trên các phép đo hợp lệ.
  Invalid fraction vẫn tính trên **toàn bộ sequence đã nhận dạng**, gồm
  sequence không đo được peak; không thu nhỏ mẫu số để che sequence hỏng.
- Trả `rejection_reasons` cho từng node, `interpolation_available` và
  `interpolation_unavailable_reason`; không raise chỉ vì còn dưới hai node.
- Lựa chọn model vẫn theo `use_model` hoặc invalid fraction **>** maximum.
  Nếu fraction bằng maximum thì không thêm fallback.

`interpolate_pg` giữ nguyên: ít nhất hai node, times hữu hạn tăng nghiêm
ngặt, correction hữu hạn khác zero. Đổi hợp đồng QC không nới gate này.

`utils/pg_calibration_input.analyze_calibration` giữ signature và tuple return
`times, relative_corrections, report`. Caller phải kiểm tra availability;
các array trả về có thể có **0 hoặc 1 node**, không mặc nhiên nội suy được.

Adapter xử lý reconstruction/correlation theo từng sequence và tính delay
**trước** PG QC. Một peak bị loại không làm mất các peak tốt còn lại. Audit
legacy raw-first-replica chỉ là diagnostic, không còn được chặn delay/PG.
Với dữ liệu đầy đủ, nominal reference và số học PG/delay giữ nguyên.

## 3. Validity, lý do loại và JSON

Mỗi `nodes[i]` có:

| Field | Ý nghĩa |
|---|---|
| `reconstructed_replica_valid` | Reconstruction/correlation thành công, peak hữu hạn khác zero, nằm trong support chứa trọn chirp |
| `replica_rejection_reason` | Lỗi reconstruction, không có peak hoặc peak ngoài support; null khi hợp lệ |
| `pg_measurement_valid` | Đo được PGcal và chuẩn hóa inverse/PGref thành giá trị hữu hạn khác zero |
| `pg_measurement_rejection_reason` | Chi tiết lỗi đo/chuẩn hóa PG; riêng với QC amplitude/phase |
| `pg_rejection_reasons` | Các tiêu chí PG QC thất bại |
| `pg_selected` | Node được chọn cho correction theo extracted/model policy |

Mã QC: `measurement_invalid`, `relative_amplitude`, `relative_phase`,
`absolute_amplitude`, `absolute_phase`. Có thể có nhiều lý do cùng node.
`pg_selected=true` với source model không có nghĩa peak measured hợp lệ.

Trạng thái tổng hợp:

- `internal_delay.available`: cần ít nhất một peak hợp lệ.
- `internal_delay.mean_s`: null nếu không đo được; không thay bằng 0/L1/AUX.
- `internal_delay.valid_sequence_indices`: index trong danh sách **đầy đủ**,
  không đánh lại số sau khi loại; per-sequence delay/lag giữ vị trí tương ứng.
- `quality.interpolation_available`: ít nhất hai node được chọn cho PG.
- `quality.source`: extracted/model; chỉ tên source không chứng minh sẵn sàng.
- `quality.absolute_scale_verified`: có PG measurement đúng normalization
  units để đối chiếu model; false khi không còn measurement. Không chứng minh
  radiometry IPF hoặc đủ node, và không thay cho hai cờ availability.

Các số đo không có dùng JSON **null**, không NaN/Infinity. Report vẫn ghi
threshold thực tế, invalid fraction, source và các sequence bị loại.

Các lỗi toàn acquisition vẫn dừng: không có PG sequence; cấu hình sai;
model LUT không hằng chưa có epoch; header/BAQ/RX/timeline/packet lỗi;
PCC order hoặc bộ loại pulse thiếu/trùng; chirp parameters khác giữa các
sequence. Đây chưa phải cơ chế sửa header/packet hay đổi chirp tự động.
Spectral denominator không đủ support và correlation không có peak là lỗi
sequence được ghi lại để tiếp tục với các sequence còn tốt.

## 4. Hành vi CLI

`estimate_doppler_centroid.py` chỉ yêu cầu những nhánh người dùng chọn:

1. Nếu source delay là L0, kiểm `internal_delay.available`.
2. Nếu bật `--l0-pg-drift`, kiểm `quality.interpolation_available`.
3. Chỉ khi các yêu cầu đó đạt mới dựng provider và xử lý echo.

CLI lưu report L0/AUX vào `<output>/calibration.json` **trước** khi báo lỗi
do thiếu peak/node, và lỗi chỉ rõ đường dẫn. Không gọi raw echo analysis
hoặc range compression sau gate thất bại. Khi chỉ cần delay và PG QC fail,
pipeline tiếp tục và không tự bật PG drift/model correction.

CLI kiểm kê `analyze_calibration.py` xuất JSON/PNG cả khi không đủ PG node
hoặc không có peak. Plot không dùng `times[0]` của array rỗng, hiển thị lý
do PG chưa sẵn sàng; các cột đối chứng L1 không có số đo được ghi null.
Report L0/AUX được lưu trước cả bước đối chứng L1. L1 không tham gia đo hay
quyết định availability.

## 5. Kiểm chứng dữ liệu thực

Output dưới `output/sentinel-1/calibration_availability/` (gitignored):

| Case | Thay đổi diagnostic trong bộ nhớ | Delay [s] | PG interpolation |
|---|---|---:|---|
| `healthy` | Không | 4,333088620078706e-7 | Có |
| `qc_failed` | `max_pg_amp_error=0`, max invalid fraction vẫn 1 | 4,333088620078706e-7 | Không |
| `one_sequence` | Chỉ đưa PG sequence đầu vào adapter | 4,333081233451953e-7 | Không |
| `all_peaks_invalid` | Đặt các PG TxCal decoded samples thành zero | null | Không |

Các case diagnostic có `configuration.fault_injection` giải thích thay đổi;
không sửa file raw/AUX hoặc dùng L1 tạo dữ liệu. `one_sequence` giữ nguyên
image calibration/echo và chỉ ẩn các chunk 15–20 khỏi danh sách đầu vào.

Hai lượt DCE toàn bộ 16.500 echo đã đối chứng:

- `dce_healthy_pg`: bật PG drift, delay L0, cùng minimum far guard 25 như
  baseline `internal_delay/default_l0`. **Tất cả** Fine DC, polynomial data/
  geometry, selected DC và RMS fit bằng chính xác baseline.
- `dce_qc_failed_delay_only`: PG QC bị làm fail trong bộ nhớ, không bật
  drift. Kết quả DC/polynomial/RMS bằng chính xác baseline `default_cli`
  có PG QC tốt nhưng cũng không bật drift. Điều này kiểm chứng lỗi QC
  không còn chặn hoặc làm đổi nhánh delay-only.

Hai lượt vẫn chọn data, ambiguity 0, RMS dưới AUX 20 Hz ở cả hai block.
`verification.json` lưu trạng thái các case và kết quả đối chiếu array.

## 6. Kiểm thử và tái chạy

`test_internal_calibration.py`: **35 test**, thêm 9 test adapter PCC tổng
hợp không cần dataset ngoài repo. Mô hình có đường TX chứa pulse tại lag
biết trước, RX/TA/EPDN/APDN là transfer functions delta; adapter thực hiện
PCC, reconstruction và tìm peak thật. Các test kiểm một sequence, một peak
hỏng, mọi peak hỏng, singular denominator, ngoài support, một/tất cả PG
fail QC, mẫu số invalid fraction, model policy và config lỗi.

Test QC cũ đổi sang kiểm report thiếu node, không tự fallback và
`interpolate_pg` vẫn từ chối. Test configurable input bổ sung mask sai
shape/không phải bool. `test_raw_data_analysis.py`: **28 test**, thêm test
caller CLI cần đúng nhánh đã bật, lưu report trước lỗi và không gọi echo
analysis khi gate thất bại. Real-L0 oracle hiện có vẫn đạt.

Full suite: **142 test, 136 đạt, 4 skip, 2 lỗi thiếu XML S6 cũ**.

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  -m unittest discover -s test -p 'test_*.py'

MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/analyze_calibration.py \
  --output output/sentinel-1/calibration_availability/healthy

MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py --profile l1-regression --l0-pg-drift \
  --minimum-far-range-discard-samples 25 \
  --output output/sentinel-1/calibration_availability/dce_healthy_pg
```

Giới hạn còn lại: parser vẫn cần header/PCC/timeline đúng và AUX model
hiện hỗ trợ. Valid peak chưa có tiêu chí SNR/leakage riêng; relative phase
vẫn unwrap theo thứ tự các measurement hợp lệ. Chưa xác nhận bit-exact IPF,
chưa thêm fallback ngoài AUX hoặc chuyển pipeline notebook S6.
