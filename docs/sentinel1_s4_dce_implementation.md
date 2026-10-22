# Triển khai kế hoạch S4 L0/AUX → DCE

Cập nhật ngày 22/10/2026: nhánh research L0/AUX → DCE đã chạy hết trên
cảnh S4 thật, có EAP/RSL và orbit EOF. Đối chiếu với
[kế hoạch](sentinel1_s4_dce_improvement_plan.md); strict vẫn dừng đúng gate
Extended Tapered, chưa tuyên bố tái hiện IPF đầy đủ.

## Luồng đang có

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py \
  --config configs/s4_dce_research.json --preflight-only

MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py \
  --config configs/s4_dce_research.json
```

`--product` mặc định là `~/sar_data/focusing_data/product_1`; output mặc định
`output/sentinel-1/s4_dce_job`. Không mở L1 trong profile mặc định `l0-aux`.
`--compare-annotation <XML> --comparison-max-time-delta-s <giây>` chỉ được
đọc sau khi đã lưu kết quả processing. Comparison nằm trong JSON riêng;
không ghép bằng index, giới hạn sai lệch thời gian, chỉ so trên giao range.

Profile mẫu là **research**: coi support echo L0 liên tục như một internal
slice để thử nghiệm, không tuyên bố đây là slice context IPF. Giữ B=6.000,
20 × 1.000 mẫu range, nominal spacing 10 s, tail `after_last_line`. RRF
`aux_nominal_fft` dùng polynomial nominal AUX_INS, chưa là Extended Tapered.
EAP/RSL tắt tường minh; I/Q, RX, PG theo cờ AUX. PG dùng absolute PGref/PGcal
đã kiểm units; không tuyên bố toàn chuỗi radiometry tương đương IPF.

Với `layout.policy=slice_context`, cung cấp `context`:

```json
{
  "slice_start_times_gps_s": [1440959806.628471],
  "last_slice_stop_time_gps_s": 1440959816.627,
  "current_slice_index": 0,
  "source": "L0/job configuration, mô tả nguồn cụ thể"
}
```

Đây là ví dụ hình thức interface, **không phải context IPF đã xác nhận cho
S4**. Không sao chép các số ví dụ làm cấu hình thực tế. Context và line times
cùng acquisition GPS epoch; index slice là 0-based. Lịch H/K giữ boundary
window chung, không lọc theo output product center. `mapping_tolerance_s` và
`continuity_tolerance_s` độc lập tolerance 5 µs của checker L1; profile S4
chọn 40 µs để chấp nhận lượng tử hóa timestamp packet, nhỏ hơn nửa PRI.

## Đầu vào bổ sung và correction

- Orbit: `--orbit-eof <file.EOF>`, bắt buộc EARTH_FIXED/UTC, mission đúng,
  đơn vị m/m/s, đủ OSV và coverage. File chỉ định lỗi không fallback về PVT.
- DEM: cài tùy chọn `pip install -e '.[dem]'`. Bản đầu hỗ trợ một band
  EPSG:4326, height datum ellipsoid, đơn vị mét được khai báo; ngoài coverage,
  nodata, datum không rõ hoặc mâu thuẫn đều lỗi.
- `corrections.apply_rsl=true` cần `reference_range_m` tường minh từ job/AUX.
  Amplitude gain là `(R/Rref)^1.5`.
- `corrections.apply_eap=true` cần object `eap` đầy đủ. Các key:
  `update_interval_s`, `boresight_update_interval_s`, `antenna_axis_body`
  (unit vector trong SAR body frame), `boresight_source` (`attitude` hoặc
  `attitude_or_nominal`), `minimum_magnitude`, `terrain`, `solver`.
- Terrain DEM: `{"kind":"dem","path":"...","height_datum":"ellipsoid","height_unit":"m"}`.
  Terrain hằng `{"kind":"constant","height_m":0}` chỉ dùng research.
- Solver cần `range_tolerance_m`, `height_tolerance_m`,
  `doppler_tolerance_mps`, `max_nfev`, `look_side`. Đây là yêu cầu độ chính xác
  tường minh, không lấy từ L1.
- EAP dùng LUT AUX_CAL đúng S4/VV, grid quanh tâm LUT với angular increment,
  không dùng nominal near/far làm support. Không chuẩn hóa LUT. Nội suy
  complex rồi dùng `sqrt(abs(EAP))/EAP`; zero/ngoài support là lỗi.
- Bank EAP/RSL nhân sau range compression trên trục đã hiệu chỉnh timing,
  trước ACCC. Khi EAP bật, coverage được kiểm cả tại mọi bank node; PVT L0
  đủ cho DCE center chưa chắc đủ cho đầu/cuối bank.
- Nominal roll chỉ được dùng khi policy cho phép và thiếu attitude tại node;
  dùng AUX_INS reference angle/height và sensitivity với đơn vị deg/m.
- `corrections` có thể override các cờ I/Q, RX, internal calibration trong
  research. `internal_delay_source=none` và statistic predefined cũng chỉ
  cho research; profile độc lập không nhận nguồn L1.
- `calibration.output_normalization=absolute` giữ scale PGref/PGcal.
  `relative` giữ hành vi regression cũ và chỉ dùng research trong runner mới.

Strict giữ cờ AUX và từ chối ablation/research candidate. Với AUX hiện có yêu
cầu Extended Tapered, strict **phải báo chưa hỗ trợ công thức RRF**. Đây là
điều kiện dừng đúng của kế hoạch; không tự chuyển sang candidate rồi ghi strict.

## DAD → implementation → test → giới hạn

| Yêu cầu | Implementation | Test/evidence | Giới hạn hiện tại |
|---|---|---|---|
| §4.3 header/preflight | `utils/dce_job.py`, RGDEC5 sample-count helper | AUX selection, config/preflight không có thư mục L1, PDU oracle | Tách scope theo PRI/config/gap, tách piece theo SWST/SWL; S4 RGDEC5 là subset được kiểm PDU, filter khác báo chưa hỗ trợ |
| §5.6 block layout | `doppler_centroid/layout.py`, core `slice_timeline` | K=1/2/3, hai tail, shared boundary, gap/thiếu support, không clamp; core start schedule khớp 787/787 metadata | 10 s inferred; tail chưa có một default IPF chung |
| Layout cũ | `legacy_slice_timeline`, `--profile l1-regression` | Các test regression cũ giữ lại | Không dùng L1 regression như processing độc lập |
| §5.1 geometry/coverage | `geometry.py`, `utils/orbit_input.py` | EOCFI-NOTE-052 2.1 §3.5 matrix/axes/quaternion; frame-bias check; EOF units/frame/coverage | EOF POEORB thật đúng ngày đã chạy cùng toàn chuỗi S4/EAP; không extrapolate orbit |
| §9.4 PG/delay | Adapter calibration nhận selected AUX paths và normalization | Oracle peak L0 kiểm cả absolute/relative; delay/QC test cũ | Units PG không chứng minh radiometry toàn chuỗi |
| §9.5 EAP | `dependent_gain.py`, AUX LUT reader, terrain/boresight providers | Phase/scale, LUT support, bank interpolation, terrain target oracle, DEM nodata/datum | Đã chạy toàn range S4 thật với terrain hằng research; chưa kiểm DEM thật/radiometry IPF |
| §9.6 RSL | `rsl_correction`, independent range gain factory | Amplitude exponent, test factory và vị trí áp sau compression | Rref phải có nguồn job/AUX |
| §5 source/fallback | `_geometry_for_block`, `_result` dùng chung hai đường mảng/segments | Zero input, RMS cao, invalid config/provider, predefined support, nguồn/reason/null | Comparison vẫn là diagnostic, không chứng nhận DC của IPF |
| Provenance | Job/AUX/input/source hashes, correction ledger | Config/reader tests, JSON không có reference trong processing | CLI chưa có persistent cache; không sửa cache notebook S6 |

`preflight.json` hiện là kiểm header/config/coverage trước decode echo nặng.
QC calibration cần đo từ cal packets khi thực thi và được lưu `calibration.json`
trước lỗi availability. Khi calibration hoặc support/gain bank không đạt,
runner cập nhật lại `ready=false` cùng lỗi trước khi decode echo nặng. Không diễn giải `ready=true` như chứng nhận mọi dữ
liệu tín hiệu đều đạt QC.

Kết quả chạy thử đầu tiên (header chirp, PG relative, EAP/RSL tắt): hai block
0/10.500, RMS khoảng 1,419/1,207 Hz. Đây là artifact giai đoạn đầu ở
`output/sentinel-1/s4_dce_job_verified/`; không lấy nó làm kết quả cuối cho
profile AUX nominal/PG absolute đã cập nhật.

## Kiểm chứng đã chạy

- Toàn bộ unittest: **200 test: 194 đạt, 6 integration test skip** vì thiếu
  dữ liệu ngoài repo. Bao gồm optional DEM với rasterio đã cài để kiểm fixture.
- Core `context_starts` kiểm riêng trên timing của **787/787 annotation**:
  khớp ít nhất một tail, start residual lớn nhất 3,300 µs. Artifact:
  `output/sentinel-1/s4_dce_core_layout_regression.json`.
- Chạy thật với input directory **chỉ có symlink L0 và AUX, không có L1**:
  `output/sentinel-1/s4_without_l1_input/`. Job ở
  `output/sentinel-1/s4_without_l1_research_rsl.json` dùng AUX nominal FFT,
  PG absolute, RSL bật với Rref=1.000.000 m do job khai báo để thử nghiệm;
  EAP tắt. Không diễn giải reference range này là cấu hình IPF đã suy ra.
- Kết quả tại `output/sentinel-1/s4_without_l1_verified/`: start 0/10.500,
  data source cả hai, RMS 1,419013/1,207441 Hz. Processing JSON không dùng
  L1; comparison được ghi sau, ở file riêng. RMSE selected DC với data
  polynomial L1 khoảng 1,358/1,436 Hz trên giao range; time delta khoảng
  −4,292/+4,053 µs. Đây là diagnostic, không phải tiêu chí tinh chỉnh cấu hình.
- Strict preflight thật đã dừng trước decode echo với lý do RRF chưa xác
  nhận và RSL thiếu reference range, không tự fallback sang candidate/L1:
  `output/sentinel-1/s4_strict_gate/preflight.json`.
- Toàn bộ 40 source fingerprint trong lần chạy thật trùng source tại lúc
  đối chiếu kết quả. Các artifact/data này nằm ngoài Git.

## Những phần vừa hoàn thiện

- Header-only segmentation tách scope khi PRI, chirp, rank, RX gain,
  instrument configuration, số mẫu hoặc continuity đổi. SWST/SWL đổi tạo
  piece để range alignment trong scope tương thích. Packet index gốc được
  giữ cho decoder. Block context cắt qua boundary được báo thiếu support,
  không clamp, rút ngắn hay nối hai PRF. `missing_support_policy=report`
  cho phép xử lý các cửa sổ còn đủ; danh sách thiếu vẫn nằm trong preflight.
- Đo delay từ calibration rồi tính ROI giao sau compression/timing/guard
  bằng cùng `compression.slc_support` với đường xử lý thật. Kiểm block range
  và dựng gain bank trước thống kê/decode echo nặng. Support sai hoặc thiếu
  làm `ready=false`, có nguyên nhân lưu trong preflight.
- Calibration chỉ chạy khi PG hoặc delay được yêu cầu. Delay thiếu và PG QC
  không đạt được ghi thành preflight thất bại; `calibration.json` giữ kết quả
  đo nếu có. Ledger còn yêu cầu EAP/RSL khi resolve correction thất bại.
- Mỗi scope ghi config đã resolve và inventory nguồn/đơn vị/trạng thái của
  tham số; input gồm hash DEM khi dùng. AUX inventory lưu riêng phiên bản
  payload và XSD đính kèm cùng hash: PP1 hiện là payload 3.15 nhưng XSD 3.14,
  không loại payload được hỗ trợ chỉ vì XSD đính kèm cũ. Processing ghi scope ID cùng window
  ID và index toàn acquisition để không đánh đồng index cục bộ của hai PRI.
- Test runner dùng echo tổng hợp với **hai PRF**, chạy FFT/compression/DCE
  thật và khôi phục DC biết trước 7 Hz. Ba lần chạy, gồm hai annotation có
  polynomial khác nhau, cho processing JSON **giống từng byte**, trong khi
  comparison thay đổi. Không có thư mục L1 trong input test.
- Audit, plot và export regression đã kiểm với geometry fallback do echo
  zero: data polynomial, fine DC và RMS không đo được giữ `null`; plot không
  lỗi. Lỗi config/provider vẫn truyền ra, không bị đổi thành fallback.

## Đối chiếu nghiệm thu theo kế hoạch

| Nhóm | Kết quả và bằng chứng |
|---|---|
| L0/AUX độc lập | Runner mặc định không đọc L1; test hai PRF và thay comparison giữ nguyên bytes processing; lần chạy S4 thật không có thư mục L1 |
| Header và support | Segmentation PRI/config/gap, PDU RGDEC5, ROI/guard/delay trước echo decode; test từ chối ROI thiếu và lưu lỗi |
| Layout/context | Core H/K cùng hai tail, K=1/2/3, shared boundary, không clamp; 787 metadata start schedule; không suy context IPF chưa biết |
| Geometry | ESA numerical oracle, ERFA/frame test, EOF reader/coverage, terrain analytic oracle, DEM fixture và nodata/datum |
| Correction | PG absolute/relative oracle, delay độc lập QC, EAP complex LUT/scale/support, RSL amplitude, gain bank trước DCE |
| Source/fallback | Data/Geometry/Predefined; zero/NaN/RMS/invalid provider; JSON null và audit/plot/regression caller |
| Provenance | Hash raw/AUX/config/source/EOF/DEM khi dùng, inventory đơn vị/nguồn và ledger yêu cầu/thực áp dụng |
| Regression | Legacy profile giữ hành vi cũ; comparison riêng theo time/range giao, bounded time; không dùng RMSE để chọn cấu hình |

## Phạm vi kiểm chứng còn lại

- EAP/RSL toàn cảnh S4 đã chạy đạt; terrain và các tham số research không
  được diễn giải thành cấu hình IPF đã suy ra.
- Chưa có DEM ellipsoid thật cho cảnh; terrain hằng chỉ kiểm chuỗi research.
- Extended Tapered chưa có công thức được xác nhận, strict phải dừng rõ.
- Internal slice context IPF của L0 hiện có và nguyên nhân hai tail lệch một
  PRI chưa xác nhận; không suy ngược từ comparison vào processing.
- Chưa chạy decoder với bộ raw/decoded reference SAR-MPC; đây là kiểm chứng
  tích hợp tùy chọn, không tự tải dữ liệu trong pipeline.
- Chưa sửa pipeline focus SLC, mở rộng S6/TOPS hoặc noise/RFI ngoài DCE.

Nguồn frame: [EOCFI-NOTE-052 issue 2.1](https://forum.step.esa.int/uploads/default/original/2X/3/315f415284743cf42be03a464121df1ca337a26d.pdf).

## Lần chạy EAP/RSL S4 thật sau cùng

Artifact: `output/sentinel-1/s4_eap_verified/`. Input directory
`output/sentinel-1/s4_without_l1_input/` chỉ có L0/AUX. Dùng cấu hình
`output/sentinel-1/s4_eap_research.json`, orbit ESA POEORB có validity
02/09/2025 22:59:42 đến 04/09/2025 00:59:42 UTC. Không dùng L1 khi
resolve config, calibration, gain bank hoặc DCE. Chỉ đọc annotation ở bước
comparison sau khi lưu processing JSON.

| Start line (0-based) | Source | Fit RMS (Hz) | RMSE với polynomial data L1 (Hz) |
|---|---|---|---|
| 0 | data | 1,419896 | 1,360397 |
| 10.500 | data | 1,208924 | 1,439180 |

So sánh trên giao range, 19 điểm/block; sai lệch thời gian khoảng
−4,292/+4,053 µs. Không chỉnh tham số để giảm các RMSE này. EAP đổi trọng
số complex theo range, nên kết quả có thể khác nhánh EAP tắt; sai số với
L1 tăng/giảm không tự chứng minh một correction đúng/sai.

Các correction thực áp dụng: I/Q bias, RX variation, internal calibration
PG absolute, internal delay, AUX nominal FFT RRF, EAP và RSL. Gain/quadrature
I/Q giữ tắt theo AUX. ROI cuối có **18.145 mẫu**; guard far-range **25 mẫu**,
delay đo **4,333089 × 10⁻⁷ s**. Có hai piece SWST cùng scope liên tục.

Giá trị job research: Rref=1.000.000 m, terrain ellipsoid height=0 m, bank
update=20 s (hai node đầu/cuối trong cảnh khoảng 10 s), boresight update=1 s,
antenna body axis=[0,0,1], policy `attitude_or_nominal`. Node boresight cuối
ngoài attitude L0 dùng AUX_INS nominal roll, được ghi rõ nguồn; không
extrapolate attitude. Các giá trị này phục vụ kiểm luồng, chưa phải bộ
tham số vận hành IPF được xác nhận.

Toàn bộ **40 hash source** của lần chạy trùng code tại lúc kiểm kết quả.
Các file processing/comparison/calibration/preflight là JSON riêng. Dữ liệu,
EOF và output vẫn nằm ngoài Git.
