# AGENTS.md — sentinel1_level0_decoder_demo

Sentinel-1 Level-0 → focused SLC (Stripmap) processing demo, derived from
[Rich Hall's sentinel1decoder](https://github.com/Rich-Hall/sentinel1decoder).

## Overview

The repo contains **marimo notebooks** rendered as Python files under
`workflows/sentinel-1/`, backed by pure-processing modules
under `src/sentinel1_processing/`. The notebook decodes raw Level-0 I/Q data,
corrects I/Q bias, range-compresses, estimates Doppler centroid, and focuses an
SLC image. Expensive stages use the `cache.persistent(...)` notebook-support
context, which hides the marimo cache configuration from the workflow.

All user-facing docs/commits are in Vietnamese.

## Mục tiêu và tài liệu cần đọc trước

Dự án nhằm **mô phỏng/tái hiện quá trình lấy nét ảnh Sentinel-1 từ dữ liệu thô
Level-0 (L0) lên Level-1 (L1)**, trước mắt là ảnh Single-Look Complex (SLC)
trong hình học slant range. Đây là dự án nghiên cứu/tái hiện thuật toán; ảnh đã
focus chưa mặc nhiên tương đương một sản phẩm SAFE L1 đầy đủ của IPF.
Các workflow RADARSAT-1 là phần thử nghiệm riêng; context dữ liệu dưới đây dành
cho Sentinel-1.

- Đọc [workflow L0 → L1 theo DAD](docs/sentinel1_l0_to_l1_workflow.md) trước
  khi sửa thuật toán hoặc chuyển sang cảnh mới. Tài liệu ghi thứ tự xử lý,
  nguồn tham số, các nhánh theo mode/product, chuẩn hóa và điểm kiểm tra.
- Đọc [kiểm tra RRF ngày 21/10/2026](docs/sentinel1_rrf_followup.md) khi sửa
  range compression. `from_coefficients` là builder chung; reference-input
  nominal chirp dùng AUX_INS. Các phổ mở rộng trong `check_range_reference.py`
  là giả thuyết nghiên cứu, không tự coi là công thức IPF Extended Tapered.
- Raw analysis S4 đã triển khai tại `common/raw_data_analysis.py`; xem
  [API và các diễn giải DAD](docs/sentinel1_raw_data_analysis.md).
  Các tham số được DAD gọi configurable phải đi qua đầu vào của tác vụ,
  không chôn trong thân hàm. Batch size không được làm đổi phạm vi thống kê.
- `common/raw_data_correction.apply` hỗ trợ đủ bias → gain → quadrature
  theo §9.2.1; nhận statistic/cờ tường minh, không sửa input. Xem
  [công thức, kiểm thử và so sánh S4/VV](docs/sentinel1_raw_data_correction.md).
  Demo DCE áp trước RX/PG/range compression; truyền `iq_bias=0j` cho range
  compressor để không trừ hai lần. Giữ mặc định AUX: bias bật, gain/quad
  tắt; thử nghiệm bật cả ba không cải thiện RMSE DCE. JSON có thống kê
  trước–sau cùng subset và nguồn tham số; theta luôn radians.
- `references/` chứa **tài liệu tham khảo thuật toán**. Tại lần khảo sát
  20/10/2026, thư mục chỉ có
  `ESA2022_Sentinel-1 Level 1 Detailed Algorithm Definition_bookmarked.pdf`:
  **Sentinel-1 Level 1 Detailed Algorithm Definition (DAD)**, SEN-TN-52-7445,
  DI-MPC-IPFDPM, MPC-0307, Issue/Revision **2/5**, ngày **17/11/2022**, 166 trang PDF.
- DAD là nguồn đối chiếu luồng/công thức; XML AUX là nguồn cấu hình cụ thể.
  Luôn ghi mục/phương trình DAD và XML path của tham số khi triển khai.
  Các tham chiếu cũ đến `references/s1a-aux-*.xml` trong code không phản ánh
  nội dung thư mục hiện tại; AUX thực tế nằm trong bộ dữ liệu bên ngoài repo.

## Dữ liệu Sentinel-1 cục bộ

Nguồn dữ liệu của dự án là **`~/sar_data/focusing_data/`**, hiện tương ứng
`/home/xiaoxin/sar_data/focusing_data/`. Đây là dữ liệu bên ngoài Git, không phải
thư mục `data/` trong repo. Khi lập đường dẫn Python, dùng
`Path("~/sar_data/focusing_data").expanduser()` hoặc `Path.home()`.

Cấu trúc đã khảo sát ngày 20/10/2026: có **một bộ dữ liệu `product_1`**, gồm
ba phần của cùng cảnh, không phải ba sản phẩm độc lập:

```text
~/sar_data/focusing_data/
└── product_1/
    ├── level-0/                  # Raw L0: một file VV .dat và manifest.safe
    ├── level-1/                  # Metadata L1 tham chiếu, các XML đặt phẳng
    └── auxiliary/                # Ba gói SAFE AUX_INS, AUX_PP1, AUX_CAL
        └── <tên-gói>.SAFE/
            ├── manifest.safe
            ├── data/s1a-aux-<loại>.xml
            └── support/*.xsd
```

### Nhận dạng cảnh `product_1`

| Thuộc tính | Giá trị quan sát từ tên file, manifest và annotation |
|---|---|
| Vệ tinh / mode / swath đang có | Sentinel-1A / Stripmap (`SM`) / `S4` |
| Kênh dữ liệu cục bộ | `VV`; manifest gốc còn liệt kê `VH`, nhưng file VH không có ở đây |
| Absolute / relative orbit | `60827` / `30`, chiều bay `DESCENDING` |
| Datatake trong tên file | `0792BB` (annotation biểu diễn thập phân `496315`) |
| Khoảng thu nhận theo manifest L0 | `2025-09-03T18:36:28.308510Z` → `2025-09-03T18:36:38.958931Z` |
| Khoảng ảnh theo annotation L1 VV | `2025-09-03T18:36:28.964370` → `2025-09-03T18:36:37.828315` (UTC) |
| Loại sản phẩm L1 tham chiếu / bộ xử lý | `SLC`, Sentinel-1 IPF `003.92` |

Khoảng thời gian raw và ảnh hợp lệ khác nhau; không cắt input theo thời gian
output L1. Các giá trị L1 trong mục này chỉ nhận dạng/đối chiếu, không phải
tham số đầu vào để tái tạo ảnh.

### (1) `level-0/` — dữ liệu nguồn để xử lý

- `s1a-s4-raw-s-vv-20250903t183628-20250903t183638-060827-0792bb.dat`:
  raw packet L0 VV, **235.089.092 byte** (khoảng 224 MiB). Dùng decoder đọc
  packet/header và giải mã I/Q; không coi đây là ma trận complex lưu thẳng.
- `manifest.safe`: XML mô tả sản phẩm L0, thời gian thu, vệ tinh, mode, orbit,
  footprint, lịch sử xử lý và danh mục data object.
- Manifest gốc liệt kê cả file VH, `*-annot.dat`, `*-index.dat`; các file đó
  **không có** trong bản trích cục bộ này. Kiểm tra file thực có trước khi dùng
  đường dẫn trong manifest; không mặc định đây là gói SAFE hoàn chỉnh.
- Khi đổi cảnh, phải đọc lại packet để xác định echo/calibration/noise,
  acquisition chunk, PRI, SWST/SWL, Range Decimation, chirp, orbit/PVT và
  attitude có sẵn. Khảo sát metadata bằng decoder cho bộ S4 hiện tại có
  28 chunk: echo ở chunk 13 (11.550 dòng) và 14 (4.950 dòng), có SWST đổi;
  các chunk khác chứa noise/calibration. Đây là kết quả riêng `product_1`,
  không dùng làm mặc định cho cảnh khác.

### (2) `level-1/` — bộ tham chiếu kết quả

Các XML được đặt **phẳng**, không giữ cấu trúc `annotation/calibration/` của
SAFE gốc. Hậu tố chung của bốn file annotation là
`s1a-s4-slc-vv-20250903t183628-20250903t183637-060827-0792bb-002.xml`:

| File | Nội dung và mục đích tham chiếu |
|---|---|
| `manifest.safe` | Nhận dạng SLC, IPF, tài nguyên AUX/orbit đã dùng và danh mục file của SAFE gốc. |
| File hậu tố chung, không thêm tiền tố | Annotation chính: `adsHeader`, quality, cấu hình/downlink, orbit (15 state vector), attitude (11 bản ghi), raw-data analysis, replica, noise, terrain height, azimuth FM rate (5), image/processing information, Doppler centroid (2 estimate), antenna pattern, geolocation grid (462 điểm). Các danh sách burst, coordinate conversion và swath merge đang rỗng. |
| `calibration-<hậu-tố-chung>` | 11 calibration vector; mỗi vector có 460 vị trí pixel và các LUT `sigmaNought`, `betaNought`, `gamma`, `dn`. Đây là calibration annotation của L1, khác với AUX_CAL. |
| `noise-<hậu-tố-chung>` | 6 noise-range vector, mỗi vector có 460 vị trí pixel và `noiseRangeLut`; `noiseAzimuthVectorList` rỗng. |
| `rfi-<hậu-tố-chung>` | Header và trạng thái `rfiMitigationApplied=None`. |

Annotation mô tả ảnh **14.628 dòng azimuth × 18.322 mẫu range**, pixel complex,
16-bit signed integer, spacing khoảng **4,142846 m azimuth × 2,662357 m range**.
Đây là kích thước/spacing **tham chiếu**, không dùng để ép kích thước output.
Thư mục hiện **không có measurement TIFF**, preview hoặc kênh VH dù manifest
liệt kê chúng. Có thể đối chiếu metadata, DCE/FM rate, timing, chất lượng và
geolocation; chưa thể so sánh trực tiếp pixel ảnh chuẩn từ riêng bộ này.

**Ranh giới bắt buộc:** L1 chỉ phục vụ test/chẩn đoán. Không lấy I/Q bias,
chirp, Doppler polynomial, FM rate, orbit/attitude, terrain height, calibration
LUT, timing hoặc kích thước ảnh từ L1 đưa ngược vào pipeline tạo SLC.

**Ngoại lệ có phạm vi:** người dùng yêu cầu suy cấu hình DCE từ L1, sau đó
dùng các hiệu chỉnh input tính sẵn cho demo
`workflows/sentinel-1/estimate_doppler_centroid.py`. Demo mặc định chỉ suy
layout/time alignment; bias mặc định đo từ toàn echo L0 hợp lệ, dùng chung
hai chunk. Cờ `--l1-input-corrections` còn dùng bias, replica gain/delay,
PG tương đối và bank EAP từ L1; nominal chirp và receiver variation từ AUX_INS, RSL theo
DAD. DC vẫn đo từ echo L0, geometry từ ephemeris L0; không dùng polynomial
Doppler L1 thay cho ước lượng. Chế độ này không phải kiểm chứng L0 độc lập
với L1. Không chuyển ngoại lệ sang pipeline focus chính. README/JSON ghi
nguồn, giả định, support và kết quả riêng của từng chế độ.
Thêm `--l0-raw-analysis` để thay riêng bias L1 bằng bias L0 trong chế độ
tham chiếu; JSON phải ghi rõ giá trị reference và giá trị thực sự áp dụng.

### (3) `auxiliary/` — cấu hình và đặc tính hệ thống

| Gói đang có | Payload / nội dung cần tra |
|---|---|
| `S1A_AUX_INS_V20231201T000000_G20251015T084505.SAFE` | `data/s1a-aux-ins.xml`, root `auxiliaryInstrument`, schemaVersion `3.16`: radar frequency, các thời gian guard/suppression/latch, roll steering, 23 swath parameter set, 88 internal calibration set, 30 timeline và các LUT giải mã BAQ/FDBAQ. Có nominal chirp, receiver variation, on-board decimation filter, internal delay, PG model/reference, SWST/azimuth bias theo swath/kênh. |
| `S1A_AUX_PP1_V20190228T092500_G20250515T131849.SAFE` | `data/s1a-aux-pp1.xml`, root `l1AuxiliaryProcessorParameters`, schemaVersion `3.15`: 33 product configuration và 12 application LUT; các cờ hiệu chỉnh, pre-processing, RFI, DCE, SLC, post-processing, bandwidth, cửa sổ, look, spacing và block. Với cảnh này tra `productId=SM_SL1__1`, sau đó `swath=S4`. |
| `S1A_AUX_CAL_V20190228T092500_G20240327T102320.SAFE` | `data/s1a-aux-cal.xml`, root `auxiliaryCalibration`, schemaVersion `2.10`: 88 calibration parameter set theo swath/polarisation, elevation/azimuth antenna pattern, azimuth antenna element pattern, absolute calibration constant và noise calibration factor. Chọn `swath=S4`, `polarisation=VV`. |

Mỗi gói có `manifest.safe` ghi định danh/hiệu lực và `support/` chứa schema
riêng cùng `s1-object-types.xsd`. Đọc XSD để xác định đơn vị, kiểu, cấu trúc
vector và quy ước complex; không suy đoán từ tên field. Việc parse XML được
kiểm tra khi khảo sát; chưa xác nhận toàn bộ bằng XSD (payload PP1 ghi 3.15,
schema PP1 đi kèm ghi 3.14).

Một số giá trị **tra được từ AUX cho S4**, hữu ích để kiểm tra chọn đúng nhánh:
`aziProcBandwidth=1429 Hz`, `aziBlockSize=4096`,
`extraAziProcBlockOverlap=50`, `maxFdc=100 Hz`; range processing bandwidth
`50.600.000 Hz`, Hamming coefficient `0.75`; SWST bias `-8.2256909e-9 s`;
AUX_CAL S4/VV có absolute calibration constant `1.0`, noise factor `0.817658`.
Đọc lại XML khi thực thi, không dùng danh sách tóm tắt này thay parser.

**Khác biệt phiên bản cần giữ trong context:** manifest L1 ghi đã dùng
`S1A_AUX_INS_V20231201T000000_G20250116T092837.SAFE`, còn bản cục bộ là
`...G20251015T084505.SAFE`. AUX_PP1 và AUX_CAL trùng tên bản ghi trong
manifest. Đây là provenance, không phải bằng chứng AUX không áp dụng được
hoặc nguyên nhân sai số DCE. Payload hai bản chưa được so sánh.

Theo [Auxiliary Product Specification §3.2.1, Issue 3/16](https://sentiwiki.copernicus.eu/__attachments/a_ac101207bbd6069e56e3922c261317dac89cb2f1bc1b44e0696bad24def0df0f/S1-RS-MDA-52-7443%20-%20Sentinel-1%20IPF%20Auxiliary%20Product%20Specification%202024%20-3-16.pdf),
chọn `V` gần nhất nhưng không sau job start; nếu trùng `V`, chọn `G` mới nhất.
`V` là ngày bắt đầu áp dụng, `G` là ngày tạo. Giữa hai bản INS trên, bản
`G20251015` được ưu tiên khi xử lý lại; generation sau ngày thu không loại
nó khỏi lựa chọn. Chưa kiểm kê toàn bộ catalog AUX để xác nhận bản phù hợp
nhất trên toàn kho. Khi cần tái lập đúng lần chạy L1 cũ, giữ thông tin bản
`G20250116` mà manifest ghi nhận.

File orbit
`S1A_OPER_AUX_RESORB_OPOD_20250903T194555_V20250903T160523_20250903T192253.EOF`
được manifest L1 nhắc tới nhưng hiện không có trong bộ dữ liệu.

### Phân biệt bộ dữ liệu mới với cấu hình code hiện hành

Notebook `workflows/sentinel-1/focus_scene.py` hiện trỏ vào
`data/sao_paulo/s1a-s6-raw-s-vv-20251226t214356-20251226t214427-062491-07d496.dat`,
chọn `SCENE_CHUNKS=(13, 14)` và dùng `src/sentinel1_processing/s6_parameters.py`.
Đó là cấu hình **S6**, khác `product_1` **S4**. Cập nhật tài liệu không có
nghĩa là pipeline đã chuyển sang bộ dữ liệu mới. Không chỉ đổi `INPUT_PATH`
rồi giữ nguyên swath, chunk, LUT, bandwidth hoặc các ngưỡng regression của S6;
phải xác định lại từ L0/AUX tương ứng trước khi xử lý S4.

## Commands

- **Install**: `pip install -r requirements-lock.txt && pip install -e . --no-deps`
- **Open notebook**: `marimo edit workflows/sentinel-1/focus_scene.py`
- **Run tests** (unittest — pytest is *not* installed):
  ```bash
  MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python -m unittest discover -s test -p "test_*.py"
  ```
  (`MPLBACKEND=Agg` is required on headless boxes; a plain run errors with a
  Tk "couldn't connect to display" — an environment issue, not a code failure.)
- **Check notebook**: `marimo check workflows/sentinel-1/focus_scene.py`
  (demo DCE là CLI Python thường, không đưa vào lệnh kiểm tra notebook).
- The project interpreter lives at `/home/xiaoxin/python_envs/sentinel1/bin/python` (repo is not under that env's site-packages path).

## Layout

- `workflows/sentinel-1/` — Sentinel-1 marimo workflows (cells as `@app.cell`).
- `workflows/sentinel-1/estimate_doppler_centroid.py` — demo CLI S4/VV, chạy
  bằng Python thường; không phải notebook marimo. Suy layout theo ngoại lệ L1
  nêu trên, xuất JSON/PNG; dùng memmap tạm để giới hạn RAM.
- `src/sentinel1_processing/utils/dce_reference_input.py` — chuẩn bị input
  DCE chẩn đoán từ L1/AUX theo cờ CLI; không tự áp vào pipeline focus.
- `workflows/radarsat-1/` — RADARSAT-1 command-line workflows.
- `src/sentinel1_processing/` — pure processing package:
  - `pre_processing/downlink_header_validation/` — kiểm tra header downlink theo DAD §4.3.
  - `common/raw_data_correction.py` — bias/gain/quadrature theo DAD §9.2.1;
    `estimate_iq_bias` giữ tương thích notebook S6, `apply` là operator chung.
  - `common/raw_data_analysis.py` — thống kê streaming §4.1 và chọn tham số
    correction với cờ/predefined values nhận tường minh từ caller.
  - `range_processing/` — `range_reference_function`, `swst_bias`, `dependent_gain`.
  - `azimuth_pre_processing/range/compression.py` — range compression
    (`compress`, `zero_pad`, `forward_fft`, `multiply_reference_function`,
    `inverse_fft`, `extract_valid_samples`).
  - `azimuth_pre_processing/` — zero padding and forward FFT.
  - `azimuth_processing/` — focus pipeline: RCMC, secondary range compression,
    azimuth compression, processing blocks.
  - `doppler_centroid/estimation.py` (~2k lines) — DCE từ dữ liệu.
  - `doppler_centroid/geometry.py` — DCE hình học từ orbit/PVT và attitude.
  - `common/effective_velocity.py` — effective velocity theo DAD §9.10.
  - `utils/dce_plotting.py` — DCE diagnostic plots.
  - `utils/geotiff.py` — export SLC GeoTIFF với GCP.
- `src/notebook_support/cache.py` — fingerprint, cache và memmap dành riêng cho notebook.
- `test/` — unittest suite for Sentinel-1, RADARSAT-1, cache, and effective velocity.
- `data/`, `output/` — raw input / produced output (gitignored, see `.gitignore`).
- `~/sar_data/focusing_data/` — bộ dữ liệu Sentinel-1 nguồn hiện tại, ngoài repo;
  xem mục dữ liệu ở trên.
- `references/` — tài liệu thuật toán, hiện có DAD Sentinel-1; không phải kho AUX.
- `docs/sentinel1_l0_to_l1_workflow.md` — workflow và checklist triển khai theo DAD.
- `.cache/sentinel1/` — marimo persistent cache (gitignored).

## Notebook conventions

- Mọi tham số và dữ liệu dùng để tạo SLC phải xuất phát từ sản phẩm Level-0
  hoặc các file AUX (`AUX_INS`, `AUX_PP1`, `AUX_CAL`). Metadata Level-1 chỉ
  được dùng trong test/chẩn đoán để tham khảo và đối chiếu; không được đưa giá
  trị suy ra từ Level-1 trở lại pipeline xử lý.
  Ngoại lệ demo DCE dùng layout/input corrections L1 ở trên không áp dụng cho việc tạo SLC.
- Cells are `@app.cell`; each cell declares its global deps via its `def _(...)` params.
- **Separate computation from display**: heavy compute goes in its own cell, then a
  separate display cell (`print`, `md`, `plt`) reads its result. Do NOT put
  `print()`/plots inside a `cache.persistent(...)` block —
  they're skipped on cache hits.
- Persistent blocks wrap only data-producing computation:
  ```python
  with cache.persistent(stage_name, scene_cache_directory):
      cache_inputs
      result = build_result()
  ```
- Keep numeric/array constants (sample rates, pulse params) as named vars in early cells.

## Gotchas (marimo `persistent_cache`)

- **Don't pass Python `complex` scalars into values hashed by `persistent_cache`.** marimo's
  content hasher (`primitive_to_bytes`) can't serialize a plain `complex`:
  `TypeError: cannot convert 'complex' object to bytes` at `__enter__`, before the block runs.
- Use a numpy complex instead — `np.complex128(real, imag)` is a *data primitive* and hashes fine.
  Example: `iq_bias=np.complex128(*iq_bias_components)` instead of `complex(*...)`.
  (Add `np` to the cell's `def _(...)` params if not already a dep.)
- **A Python `complex` can also leak into notebook scope and poison a downstream cache's hash.** If a
  cache block (or a dependent display cell) returns/exposes a bare `complex` (e.g. from
  `estimate_iq_bias()`), any later `persistent_cache` block that transitively depends on it will
  fail at `__enter__` with the same `cannot convert 'complex' object to bytes`. Wrap the scalar as
  `np.complex128(value)` before it leaves the block, so it's hashed as a data primitive.
- The notebook auto-clears stale cache entries for a stage after the later Focus stage completes;
  to recompute everything, delete `.cache/sentinel1/` and restart the notebook.
- Don't delete names bound inside a cache block; marimo restores those names on
  cache hits.

## Testing expectations

- Báo cáo kiểm tra luồng DCE S4/VV: [docs/sentinel1_dce_audit.md](docs/sentinel1_dce_audit.md).
  Tái chạy bộ thử bằng `workflows/sentinel-1/audit_doppler_centroid.py`;
  JSON/CSV tại `output/sentinel-1/dce_audit/`. Không đổi cấu hình chính chỉ
  vì một ablation hoặc cửa sổ quan sát khác cho RMSE với L1 thấp hơn.

- Tests use `unittest` classes (not pytest fixtures), với `MPLBACKEND=Agg`.
  Lần chạy raw correction ngày 21/10/2026: 86 test, 80 đạt, 4 skip, 2 lỗi do thiếu
  annotation S6 cũ `references/s1a-s6-slc-vv-20251226t214357-20251226t214426-062491-07d496-002.xml`.
  15 test trong `test/test_sentinel1_dce_demo.py` và 5 test trong
  `test/test_range_reference.py` đều đạt. Không nhầm các
  lỗi thiếu fixture S6 này với sai số của demo S4.
  20 test `test/test_raw_data_analysis.py` và 10 test
  `test/test_raw_data_correction.py` đều đạt, gồm một test echo S4
  thật so với NumPy oracle; test này chỉ skip nếu thiếu dữ liệu ngoài repo.
- New pure-processing functions should get a `test_*` unittest; keep `np.testing.assert_*` for array comparisons.
