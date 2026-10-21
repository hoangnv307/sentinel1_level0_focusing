# Sentinel-1 Level-0 Decoding Demo

Dự án mô phỏng/tái hiện quá trình lấy nét ảnh Sentinel-1 từ dữ liệu thô
**Level-0 (L0) lên Level-1 (L1)**, trước mắt là ảnh **SLC** trong hình học
slant range. Pipeline phát triển từ dự án của
[Rich Hall](https://github.com/Rich-Hall/sentinel1decoder).

## Dữ liệu và tài liệu tham khảo

Dữ liệu nguồn nằm tại **`~/sar_data/focusing_data/`**, bên ngoài repo. Tại lần
khảo sát 20/10/2026, có một bộ dữ liệu `product_1` với ba phần:

```text
~/sar_data/focusing_data/product_1/
├── level-0/       # Raw packet VV .dat và manifest.safe
├── level-1/       # XML L1 để tham chiếu kết quả
└── auxiliary/     # Các gói SAFE AUX_INS, AUX_PP1 và AUX_CAL
```

| Phần | Vai trò |
|---|---|
| `level-0/` | Dữ liệu thô để giải mã I/Q và lấy metadata packet phục vụ lấy nét. File hiện có: `s1a-s4-raw-s-vv-20250903t183628-20250903t183638-060827-0792bb.dat` (khoảng 224 MiB). |
| `level-1/` | `manifest.safe`, annotation chính, calibration, noise và RFI XML; dùng đối chiếu DCE, FM rate, timing, hình học và chất lượng. Các XML đặt phẳng; hiện chưa có ảnh measurement TIFF để so sánh pixel. |
| `auxiliary/` | AUX_INS mô tả thiết bị, nominal chirp, timing, LUT giải mã và hiệu chỉnh; AUX_PP1 chứa cấu hình bộ xử lý; AUX_CAL chứa mẫu anten và hệ số hiệu chuẩn. Mỗi gói gồm manifest, payload XML trong `data/` và XSD trong `support/`. |

`product_1` là cảnh **Sentinel-1A, Stripmap/S4, VV**, thu ngày **03/09/2025**,
absolute orbit `60827`, relative orbit `30`, chiều bay descending. L1 tham
chiếu là SLC do IPF `003.92` tạo, annotation mô tả ảnh 14.628 × 18.322 pixel.
Manifest gốc còn liệt kê VH và nhiều file không có trong bản trích cục bộ;
không mặc định bộ dữ liệu này là SAFE đầy đủ.

**Quy tắc nguồn dữ liệu:** dữ liệu và tham số tạo ảnh phải lấy từ L0/AUX.
Metadata L1 chỉ dùng trong test/chẩn đoán, không đưa giá trị tham chiếu trở lại
pipeline. Bản AUX_INS hiện có (`G20251015T084505`) khác bản manifest L1 ghi
đã sử dụng (`G20250116T092837`); đây là provenance, không có nghĩa bản hiện
tại không áp dụng được. Xem quy tắc chọn AUX trong [AGENTS.md](AGENTS.md).
AUX_PP1/AUX_CAL trùng tên phiên bản. File
orbit AUX_RESORB được L1 ghi nhận chưa có trong bộ dữ liệu.

**Cấu hình hiện hành:** notebook vẫn dùng cảnh S6 tại `data/sao_paulo/`,
`SCENE_CHUNKS=(13, 14)` và `s6_parameters.py`. Chưa chuyển pipeline sang S4;
khi chuyển cần xác định lại chunk và cấu hình từ L0/AUX của `product_1`.

Thư mục [`references/`](references/) chứa tài liệu tham khảo thuật toán,
hiện có **Sentinel-1 Level 1 Detailed Algorithm Definition (DAD)**,
SEN-TN-52-7445 / MPC-0307, Issue/Revision **2/5**, ngày **17/11/2022**.
Đây là cơ sở đối chiếu luồng xử lý và công thức; cấu hình cụ thể lấy từ AUX.

- [Workflow L0 → L1 theo DAD](docs/sentinel1_l0_to_l1_workflow.md): luồng chi
  tiết, nguồn tham số, miền dữ liệu, chuẩn hóa và checklist kiểm tra.
- [Context dành cho các lần phát triển sau](AGENTS.md): danh mục dữ liệu,
  nội dung XML, khác biệt phiên bản và quy tắc sửa notebook.
- [Báo cáo kiểm tra tham số và thử nghiệm DCE S4/VV](docs/sentinel1_dce_audit.md):
  kiểm kê nguồn tham số, thử bỏ từng hiệu chỉnh, độ nhạy estimator và giới hạn xác nhận.
- [Bước RRF tiếp theo](docs/sentinel1_rrf_followup.md): nominal chirp từ AUX,
  Parseval/support và hai phương án phổ mở rộng được đề xuất để thử nghiệm.
- [Raw data analysis từ L0](docs/sentinel1_raw_data_analysis.md): API cấu hình
  theo DAD §4.1, significance, 17 test cụ thể và kết quả chạy dữ liệu thật.

## Cài đặt

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Mở notebook

Notebook dùng **marimo** và được lưu dưới dạng file Python:

```bash
marimo edit workflows/sentinel-1/focus_scene.py
```

Marimo tự chạy cell theo quan hệ phụ thuộc và lưu các kết quả tốn thời gian
vào `.cache/sentinel1/`. Không cần cấu hình phiên bản hoặc policy cache thủ
công.

Các kết quả được lưu gồm:

- Decode dữ liệu I/Q của các segment trong scene.
- I/Q bias của từng segment.
- Range-compressed data của từng segment.
- Doppler centroid estimation.
- Focused SLC.

## Các trường hợp sử dụng

| Thao tác | Điều xảy ra |
|---|---|
| Mở lại notebook, không đổi gì | Các bước nặng được đọc từ ổ đĩa; bảng và biểu đồ vẫn có thể được dựng lại. |
| Sửa cell DCE | DCE được tính và lưu lại. Focus chỉ tính lại nếu dữ liệu DCE thay đổi. Decode và Range Compression dùng dữ liệu đã lưu. |
| Sửa source DCE trong `src/sentinel1_processing/doppler_centroid/` | Tương tự sửa cell DCE. Marimo tự reload module và chạy lại phần phụ thuộc. |
| Sửa Range Compression | Range Compression được tính lại. DCE và Focus chỉ tính lại nếu dữ liệu đầu vào của chúng thay đổi. Decode vẫn dùng dữ liệu đã lưu. |
| Sửa Focus | Chỉ Focus chạy lại. |
| Đổi tham số hoặc file đầu vào | Cell dùng giá trị đó và các cell phía sau chạy lại; các cell phía trước không liên quan được giữ nguyên. |
| Chỉ sửa cell `print`, bảng hoặc biểu đồ | Chỉ phần hiển thị liên quan chạy lại; dữ liệu xử lý không được tính lại. |

Khi một cache mới thay thế cache cũ của cùng công đoạn, notebook tự xóa bản
cũ sau khi Focus hoàn tất để thư mục cache không tăng mãi.

## Demo Doppler Centroid cho S4/VV

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python workflows/sentinel-1/estimate_doppler_centroid.py
```

Demo mặc định đọc `~/sar_data/focusing_data/product_1`; có thể đổi bằng
`--product <thư-mục>` và `--output <thư-mục>`. Kết quả là
`output/sentinel-1/product_1_dce/doppler_centroid.json` và biểu đồ
`doppler_centroid.png`. Dữ liệu range-compressed tạm dùng memmap, tự xóa sau
khi chạy; cần khoảng 2,3 GiB dung lượng tạm cho bộ S4 hiện tại.

**Ngoại lệ nghiên cứu theo yêu cầu người dùng:** demo này suy layout DCE từ
annotation L1 rồi kiểm tra trên lưới L0; pipeline focus chính vẫn giữ quy tắc
L0/AUX. Fine DC được tính từ tín hiệu L0 đã giải mã, sửa I/Q bias và range
compression; ambiguity được giải bằng orbit/attitude L0. Tần số và polynomial
L1 chỉ dùng để vẽ/đo sai lệch sau ước lượng, không thay thế kết quả tính.

| Tham số | Cấu hình demo / căn cứ |
|---|---|
| Azimuth block | 6.000 dòng; từ duration Fine DCE L1 và PRI L0. |
| Vị trí block | Dòng L0 `0` và `10.500` (0-based), mỗi block 6.000 dòng; lịch riêng của cảnh này, không phải bước đều 3.000. |
| Quy đổi thời gian để suy lịch | `zeroDopMinusAcqTime=0,335899 s` từ L1; geometry provider vẫn nhận acquisition time L0. |
| Range block | 20 block × 1.000 mẫu, phân bố đều với start index lấy floor; overlap khoảng 96–97 mẫu. |
| Range ROI / polynomial origin | Vùng giao support range-compressed hai chunk: 18.170 mẫu, origin `0,005546845595867 s` suy từ L0/AUX; khớp t0 và các tâm Fine DC L1. |
| Polynomial / fit weights | Bậc hai, không trọng số; fit lại 20 điểm L1 tái tạo polynomial và RMS annotation trong sai số làm tròn. |
| ACCC | Power weighting theo DAD; không suy được duy nhất weighting IPF từ annotation. |
| Ngưỡng RMS | 20 Hz từ AUX_PP1 `SM_SL1__1/dcProcParams`. |
| Unwrap / outlier | FFT 4.096 điểm, 3,5σ, tối đa 5 vòng fit; lựa chọn demo, chưa suy được từ L1. |

Giả thuyết range layout là block đầu bắt đầu ở near range của vùng giao;
demo kiểm tra toàn bộ tâm block khớp annotation trong 0,02 sample. Không
khẳng định đây là cấu hình IPF duy nhất có thể tạo các tâm đó. JSON lưu cấu
hình, nguồn file, I/Q bias, các ước lượng, coherence, valid mask, RMS và sai
lệch với L1. Nếu fit RMS vượt ngưỡng, `selected_source` chuyển sang geometry;
data estimate vẫn được giữ để chẩn đoán.

Chế độ mặc định chưa phải tái hiện DCE đầy đủ của IPF: chưa áp PG drift,
EAP/RSL hoặc RFI mitigation; từ cập nhật 21/10/2026, bias lấy từ
raw analysis trên toàn bộ echo hợp lệ, dùng chung hai chunk. Batch 256
không còn giới hạn số echo phân tích. Phạm vi chọn mẫu có thể cấu hình
qua `--analysis-start-line`, `--analysis-step`, `--analysis-lines`,
`--analysis-range-start`, `--analysis-range-stop`, `--batch-lines`.
Xem [kết quả và các test raw analysis](docs/sentinel1_raw_data_analysis.md).
Demo hỗ trợ đầy đủ bias → gain → quadrature theo DAD §9.2.1 qua
`common.raw_data_correction.apply`; mặc định vẫn theo cờ AUX_PP1.
Các cặp `--correct-iq-bias`/`--no-correct-iq-bias`, tương tự cho gain và
quadrature, cho phép ghi đè để thử nghiệm. Bốn tùy chọn
`--predefined-i-bias`, `--predefined-q-bias`, `--predefined-gain-imbalance`,
`--predefined-quadrature-rad` thay statistic đo được, không tự bật correction.
JSON ghi tham số/nguồn thực dùng và thống kê trước–sau trên cùng subset.
Xem [công thức, API, 13 test mới và đối chứng S4/VV](docs/sentinel1_raw_data_correction.md).
Khi bật cả ba, gain về `1,000000015`, theta về `−4,76e−8 rad`, nhưng
RMSE DCE với L1 không cải thiện; tiếp tục giữ bias bật, gain/quadrature tắt.
Receiver gain compensation §9.2.3 hiện chạy độc lập từ L0/AUX theo
`commonProcParams/correctRxVariationFlag=true`; dùng
`--no-correct-rx-variation` để đối chứng với luồng cũ chưa có RX.
**Eq. 9-16 chia complex signal cho `sqrt(GV)`**, trong đó
`GV=10^((trend_dB+overshoot_dB)/10)` là power gain. Nhánh tham chiếu cũ
đã chia nhầm cho GV vì text extraction mất dấu căn; số liệu cũ có RX cần
đọc cùng [báo cáo sửa receiver compensation](docs/sentinel1_receiver_gain_compensation.md).
Đã có [kiểm kê calibration và PG drift từ L0/AUX](docs/sentinel1_pg_calibration.md):
800 calibration packet, hai PG sequence bao echo, API PCC/replica/QC và
đối chứng cùng layout DCE. Dùng `--l0-pg-drift` để áp drift **tương đối**;
đã [sửa chuẩn hóa PG bằng nominal chirp `1/N`](docs/sentinel1_pg_normalization.md),
đối chiếu đúng thang `PGref/PGcal` với model AUX. Giữ opt-in vì chưa kiểm
chứng radiometry toàn pipeline/IPF. Kiểm kê riêng:
`python workflows/sentinel-1/analyze_calibration.py` với `MPLBACKEND=Agg`.
Geometry dùng orbit/attitude L0 và ellipsoid. Vì vậy RMSE với L1 là kết quả đánh giá, không phải cam
kết parity hay ngưỡng pass/fail của demo.

Kết quả chạy trên toàn bộ 16.500 echo của `product_1`:

| Block (0-based, stop exclusive) | RMS fit | RMSE Fine DC so với L1 | RMSE polynomial so với L1 |
|---|---|---|---|
| `[0, 6000)` | 1,425 Hz | 1,363 Hz | 1,331 Hz |
| `[10500, 16500)` | 1,214 Hz | 1,452 Hz | 1,411 Hz |

Cả hai block giữ đủ 20 điểm fit, ambiguity bằng 0, RMS dưới 20 Hz nên
`selected_source=data`. RMS fit đo độ phù hợp polynomial với điểm DC của
demo; RMSE với L1 là phép so sánh khác. Kết quả nằm trong `output/` và không
được commit cùng code.

### Input DCE dùng hiệu chỉnh tham chiếu L1

Theo yêu cầu dùng các giá trị tính sẵn để chuẩn bị input đầy đủ hơn, chạy:

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python workflows/sentinel-1/estimate_doppler_centroid.py --l1-input-corrections
```

Kết quả được ghi riêng tại `output/sentinel-1/product_1_dce_l1_input/`,
giữ nguyên kết quả demo cơ bản. Các tham số và phép biến đổi được lưu trong
`doppler_centroid.json/input_corrections`:

| Hiệu chỉnh | Nguồn / cách áp dụng |
|---|---|
| I/Q bias | L1 `rawDataAnalysis/support/*BiasUsedForCorrection`: trừ `0,2893930077552795 + j·0,3621337115764618` khỏi echo trước mọi gain correction. |
| Receiver variation | Hệ số S4/RX V trong AUX_INS, cờ từ AUX_PP1; DAD §9.2.3, trend tính từ cuối TX pulse, overshoot từ `t_RX` sau suppressed samples; đơn vị µs, chia `sqrt(GV)` theo Eq. 9-16. |
| PG drift tương đối | Hai node inverse-PG trong L1 đã mang chiều correction; chuẩn hóa trực tiếp `value(t)/value(first)` rồi nội/ngoại suy complex §9.4. `--l0-pg-drift` thay node L1 bằng kết quả calibration L0/AUX. |
| Range replica | Nominal amplitude/phase S4/VV từ AUX_INS (cập nhật 21/10/2026); L1 `referenceReplica/gain` và delay tham chiếu vẫn được giữ. Builder dùng whitening và normalization theo Parseval; coefficient L1 chỉ để đối chiếu độ làm tròn. Không thêm Hamming. |
| Internal delay | L1 `referenceReplica/timeDelay=4,333092e-7 s`: advance output của matched filter để bù positive hardware delay; tone/chirp test kiểm tra chiều dịch. |
| EAP | Bảy vector L1 `antennaPattern`, mỗi vector 574 giá trị complex theo thứ tự I/Q; nội suy theo range, căn magnitude giữ nguyên phase rồi nghịch đảo, sau đó nội suy correction theo thời gian (§9.5). |
| Range spreading loss | Nhân amplitude `(R/800000 m)^(3/2)` theo DAD §9.6; reference range đọc từ annotation. |

Định nghĩa PG và complex EAP đã đối chiếu với
[Sentinel-1 Product Specification, Issue 3/9, Table 6-57 và 6-82](https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Specification-18052021.pdf).
Annotation đã biểu diễn inverse instrument PG cùng hệ số giữa swath;
nhánh tham chiếu nay chuẩn hóa trực tiếp, sửa lỗi nghịch đảo thêm một lần.
Xem [đối chiếu Product Specification và L0](docs/sentinel1_pg_calibration.md).
Chỉ phục hồi drift tương đối; chưa khôi phục absolute radiometric scale. Một hệ số complex hằng
chung toàn scene không làm thay đổi pha lag-1 hay DCE.

Bù internal delay cần mẫu tương lai tại mép xa: demo bỏ thêm **25 mẫu** mỗi
chunk, nên ROI chung còn **18.145 mẫu**. Vẫn dùng 20 block × 1.000 mẫu nhưng
phân bố lại trên ROI hợp lệ; Fine DC L1 được nội suy tới tâm block mới khi
tính RMSE. Không giữ các mẫu thiếu chirp support chỉ để khớp layout L1.
Bank EAP được kiểm tra range/time coverage, không âm thầm clamp ngoài biên.

Kết quả audit ngày 20/10/2026 trên toàn bộ 16.500 echo (trước khi chuyển
nominal coefficients sang AUX; xem [kiểm tra RRF ngày 21/10](docs/sentinel1_rrf_followup.md)):

Đây là số liệu lịch sử trước khi sửa mẫu số RX và chiều PG;
xem [kết quả PG hiện tại](docs/sentinel1_pg_calibration.md).

| Block | RMS fit | RMSE Fine DC/L1: cơ bản → hiệu chỉnh | RMSE polynomial/L1 sau hiệu chỉnh |
|---|---|---|---|
| 1 | 1,417 Hz | 1,362 → 1,372 Hz | 1,342 Hz |
| 2 | 1,206 Hz | 1,451 → 1,495 Hz | 1,452 Hz |

Cả hai block vẫn giữ 20 điểm fit và chọn data DC. Sai số với L1 **chưa giảm**;
đây là báo cáo thực nghiệm, không chỉnh bias/phase để ép khớp. Hai chế độ
có ROI khác 25 mẫu để bảo đảm support sau bù delay, nên Fine DC được so tại
tâm range thực tế của từng chế độ.

Không áp gain imbalance/quadrature vì cờ AUX tắt; không áp Hamming/AAP của
hậu xử lý, không trừ thermal-noise power khỏi complex input và không thêm
RFI mitigation khi annotation ghi `Never`. Geometry DC vẫn dùng ephemeris L0.

**Giới hạn:** đây là input chẩn đoán có dùng L1, không phải kết quả L0/AUX
độc lập. PG chỉ là relative drift,
và RRF nominal này chưa chứng minh tái tạo đúng mọi chi tiết phổ
`Extended Tapered` của IPF. Không gọi input là tương đương IPF chỉ vì có thêm
hiệu chỉnh. Các thông số unwrap/outlier/weighting được giữ nguyên, không
chỉnh theo RMSE để làm kết quả khớp L1.

## RADARSAT-1 Level-0

Giải mã một cửa sổ CEOS thành I/Q `complex64`:

```bash
/home/xiaoxin/python_envs/sentinel1/bin/python workflows/radarsat-1/decode_level0.py
```

Chạy thêm Chirp Scaling Algorithm và ghi ảnh xem nhanh:

```bash
/home/xiaoxin/python_envs/sentinel1/bin/python workflows/radarsat-1/decode_level0.py --focus
```

Mặc định script đọc file `.raw` duy nhất dưới `data/radarsat-1`, lấy 1.536 dòng
x 2.048 mẫu range và ghi kết quả vào `output/`. Dùng `--help` để chọn cửa sổ khác.

Ước lượng Doppler centroid từ L0, tính Geometry DC từ orbit và so sánh với
CRT polynomial trong metadata L1:

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python workflows/radarsat-1/estimate_doppler_centroid.py
```

Trích xuất metadata CEOS leader thành JSON:

```bash
/home/xiaoxin/python_envs/sentinel1/bin/python workflows/radarsat-1/extract_leader.py <file.ldr>
```

## Cấu trúc dự án

- `workflows/sentinel-1/`: các quy trình focus Sentinel-1 bằng marimo.
- `workflows/radarsat-1/`: các lệnh xử lý và focus RADARSAT-1.
- `src/`: thuật toán xử lý và tiện ích dùng lại.
- `test/`: kiểm thử tự động.
- `references/`: tài liệu thuật toán, hiện có DAD Sentinel-1.
- `docs/`: workflow chi tiết để triển khai theo DAD.
- `~/sar_data/focusing_data/`: bộ dữ liệu Sentinel-1 L0/L1/AUX, ngoài repo.
- `data/`, `output/`, `.cache/`: dữ liệu vào, kết quả và cache cục bộ; không commit.

## Quy tắc khi sửa notebook

Tách tính toán và hiển thị thành hai cell:

```python
# Cell tính toán
doppler_centroid_estimates = estimator.estimate_segments(...)
```

```python
# Cell hiển thị
print(doppler_centroid_estimates)
```

Nếu đặt `print()` hoặc vẽ biểu đồ bên trong block `cache.persistent(...)`,
phần hiển thị sẽ bị bỏ qua khi cache được dùng. Vì vậy các block này chỉ chứa
tính toán tạo dữ liệu; cấu hình kỹ thuật của marimo cache nằm trong
`src/notebook_support/cache.py`.

Cache dùng pickle; chỉ mở cache do chính dự án tạo ra. Muốn tính lại toàn bộ,
xóa thư mục `.cache/sentinel1/` rồi mở lại notebook.
