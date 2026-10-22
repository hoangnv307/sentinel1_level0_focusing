# Kế hoạch cải thiện luồng S4 từ L0/AUX tới Doppler Centroid

Cập nhật ngày 22/10/2026, thay cho kế hoạch có stride azimuth 3.000 dòng.
Đây là kế hoạch triển khai, chưa phải các tính năng đã hoàn thành.

## 1. Mục tiêu, hiện trạng và bằng chứng

Hoàn thiện luồng S4/VV tới DCE chạy độc lập từ **L0, AUX và cấu hình job
được cung cấp tường minh**. L1 chỉ dùng đối chiếu sau xử lý hoặc trong profile
regression tách biệt. Kết nối lại pipeline focus SLC là bước tiếp theo.

Core đã có I/Q correction, receiver compensation, PG/internal delay, range
compression, fine DC, unwrap, ambiguity resolution, polynomial fitting và
geometry DC. Tái sử dụng các phần này và các test hiện có; không triển khai lại.

Các khoảng thiếu cần xử lý:

| Hiện trạng đối chiếu source | Hướng cải thiện |
|---|---|
| `estimate_doppler_centroid.py` vẫn mở annotation S4 và suy layout từ L1 | Tách xử lý độc lập và comparison; nhận layout context từ L0/job |
| `build_azimuth_blocks(... slice_timeline ...)` dùng start/midpoint, lọc center theo output product và yêu cầu ZD offset | Thay bằng lịch theo H/K trong acquisition time; giữ block chung, kiểm support thực |
| `custom` đang clamp start ngoài miền vào biên hợp lệ | Không dùng clamp để che lỗi context; kiểm tra trước và từ chối cấu hình sai |
| EAP/RSL trong nhánh `ReferenceInput`, EAP lấy từ L1 | Tạo correction từ AUX_CAL và hình học |
| Fine estimation/unwrap chạy trước geometry trong `estimate_block` | Tính geometry trước, fallback có lý do khi data-DCE không đạt |
| Attitude đã có gate thời gian nhưng chưa kiểm chứng đủ frame/coverage toàn chuỗi | Kiểm coverage ở mọi điểm truy vấn, xác nhận quaternion và frame |
| `Config` còn default RMS của S6 | S4 resolver luôn truyền threshold AUX S4; không thừa hưởng default S6 |
| Tham số nghiên cứu chưa phân biệt đủ với tham số AUX | Lưu nguồn, đơn vị, mức bằng chứng và trạng thái áp dụng |

Bằng chứng mới từ [khảo sát 787 annotation](sentinel1_dce_datatake_patterns.md):

- 2.243 record đều có 20 điểm fineDce; duration/PRI phù hợp block 6.000 dòng.
- Lịch với khoảng danh nghĩa 10 s theo slice context khớp cả 787 annotation
  khi xét hai tail convention, sai số start/stop/center lớn nhất 3,300 µs.
- Có 193 cặp cửa sổ acquisition chung giữa các product kề nhau.
- Có hai tail khác nhau khoảng một PRI, kể cả cùng mission/swath/pol/IPF/PRI/span.
- Có datatake qua ranh absolute orbit; orbit không phải khóa nhóm datatake.
- Width range khoảng 1.000 mẫu được suy dưới giả định block đầu bắt đầu tại
  polynomial t0; chưa xác nhận duy nhất mọi chi tiết range support của IPF.

DAD §5.6 xác định block size, azimuth spacing và số block range là configurable.
**6.000 / 20 / khoảng 1.000 / 10 s là profile nghiên cứu có bằng chứng metadata,
không phải bộ hằng số được DAD công bố.**

## 2. Thứ tự triển khai

### Bước 1 — Tách L1, dựng cấu hình và preflight

- Tách đọc L0/AUX, resolve configuration, processing và comparison trong CLI.
  Chỉ mở L1 khi người dùng yêu cầu comparison hoặc regression có nhãn riêng.
- Lưu kết quả DCE độc lập reference trước; bỏ cách ghép estimate/reference
  bằng `zip` trong output chính. Comparison nhận hai tập kết quả riêng.
- Acquisition time và trục slant-range xuất phát từ L0 cùng các correction
  timing đã áp dụng. ZD offset L1 thuộc comparison; không truyền ngược vào core.
- Trước decode I/Q nặng, kiểm header: mode/polarization, packet continuity,
  PRI, SWST, range-decimation và số mẫu theo PDU tương ứng S4. Segment khi
  thông số đổi; không nối vượt gap hoặc tạo block qua đoạn không liên tục.
- Kiểm AUX theo loại sản phẩm và hợp đồng phiên bản, mission/mode/polarization,
  cùng quy tắc validity của từng loại. Ghi thời điểm tham chiếu dùng để chọn AUX;
  không loại AUX chỉ vì ngày phát hành cũ, cũng không mặc định mọi loại AUX
  phải phủ từng dòng theo cùng một quy tắc. Xác nhận selection time theo Aux Spec.
- Phân biệt validity để chọn AUX với coverage của bảng orbit/attitude/LUT thực
  sự được nội suy. Hai phép kiểm đều cần, không thay thế nhau.
- Resolve mọi cờ correction; lưu XML path, đơn vị, payload/schema version.
  Phân biệt lỗi XSD đi kèm cũ với hợp đồng payload đã xác nhận.
- Nếu job thiếu context hoặc tham số bắt buộc, báo thiếu trước tính nặng;
  không lấy giá trị từ L1 hoặc một swath khác để tiếp tục.

**Đầu ra:** preflight và cấu hình đã resolve, gồm correction yêu cầu/hỗ trợ,
nguồn dữ liệu, missing inputs và profile xử lý. Profile regression L1 giữ
khả năng chạy kết quả cũ nhưng không được ghi là L0/AUX độc lập.

### Bước 2 — Sửa layout DCE theo acquisition context

Đặt `a[i]`, `b[i]` là start/stop của internal slice, `B=6000`,
`S=nominal_spacing_s=10` trong profile nghiên cứu:

```text
H = a[i+1] - a[i]       nếu còn slice kế tiếp
H = b[i]   - a[i]       nếu là slice cuối hoặc một slice
K = max(1, floor(H/S))
regular_start[j] = a[i] + j*H/K, j=0,...,K-1

Chưa cuối: thêm block bắt đầu ở a[i+1].
Cuối: thêm block B dòng sát cuối theo support/convention được khai báo.
```

- **Bỏ preset stride 3.000 dòng.** Spacing chính là tham số giây danh nghĩa;
  bước thực tế H/K thay đổi theo context. Không thay bằng bước cố định 10 s
  hoặc midpoint trong mọi trường hợp. Profile line-stride chỉ còn cho nghiên cứu
  được chọn rõ; không dùng làm lịch IPF mặc định.
- Context phải có nguồn: L0 metadata nếu đủ, hoặc job/internal-layout config.
  Không mặc định SAFE slice, echo chunk, internal slice và output product là
  cùng một miền. L0 hiện có chưa được chứng minh đủ để suy toàn bộ context IPF.
- Nếu thiếu context, không tuyên bố tái tạo lịch IPF. Có thể chạy profile
  nghiên cứu được chọn rõ trên support L0 liên tục; báo phạm vi và sự khác biệt.
  Việc dựng internal slicing theo DAD §8.2.1 là tác vụ riêng khi cần, không
  suy internal start bằng cách cộng 25 s từ output product.
- Dùng thời gian tương đối với một acquisition origin chung để hạn chế mất
  chính xác float. Ánh xạ start sang line time thực; kiểm sai số ánh xạ và gap.
  Rounding/index convention phải tường minh, được test; không fit offset theo L1.
- Block dùng index `[start, stop)` với `stop-start=B`. Timestamp của ranh stop
  và thời gian dòng cuối là hai đại lượng khác nhau; lưu convention rõ ràng.
- Với metadata diagnostic, giữ cả `last_line` và `after_last_line` như báo cáo.
  Với xử lý L0, lấy số dòng/support thực và convention đã khai báo để quyết định
  tail; không chọn model bằng residual L1 hoặc theo tên vệ tinh/IPF.
- Không rút ngắn block, zero-pad dữ liệu thiếu hoặc clamp start để làm đủ
  support. Phân biệt context sai (lỗi cấu hình) với thiếu dữ liệu thu nhận
  (trạng thái không khả dụng). Block cuối chưa đủ B dòng phải có lý do loại;
  không làm mất block đầy đủ sát biên bằng một quy tắc bỏ tail chung chung.
- Không lọc block theo điều kiện center nằm trong output product. Giữ cửa sổ
  chung ở ranh slice, gán định danh acquisition window để tái sử dụng tính toán.
- Nhóm metadata theo mission/datatake/swath/polarization/context origin;
  absolute orbit là thuộc tính. Không nối datatake khác epoch chỉ vì trùng ID.
- Range ROI lấy giao support sau range compression/timing/guard. Profile dùng
  20 block × 1.000 mẫu khi support đủ; start/rounding là tham số/quy tắc rõ.
  Không tự giảm width hoặc thay ROI để khớp tâm fineDce L1.

**Đầu ra:** layout không cần L1, có start/stop/center, index, support status,
context provenance và tail convention; checker reference độc lập tiếp tục
kiểm 787 annotation. Ngưỡng 5 µs của checker metadata không thay thế ngưỡng
ánh xạ line của processing.

### Bước 3 — Kiểm chứng geometry, frame và coverage

- Cho phép orbit EOF được chỉ định; validate mission, thời gian, frame/units,
  độ đầy đủ rồi dùng file đó. Nếu không chỉ định thì dùng PVT L0.
- Kiểm coverage tại mọi thời điểm thực sự truy vấn: DCE geometry và EAP bank,
  kể cả cửa sổ ở biên/context. Không dùng `orbitModelMargin` để cho phép
  extrapolation tùy ý. Attitude Slerp đã có gate; bổ sung gate orbit và kiểm
  sorting, duplicate/conflicting timestamps, finite values.
- Kiểm chứng quaternion Q0 scalar, chiều phép quay, trục anten và frame bằng
  ví dụ số EOCFI-NOTE-052. Frame geocentric mean of 2000 cần được đối chiếu
  với GCRS hiện dùng; chưa coi hai tên là tương đương chỉ vì đều inertial.
- Giữ geometry-DCE trên ellipsoid theo DAD §5.1. Terrain dùng riêng cho gain
  EAP; không âm thầm đổi bề mặt geometry-DCE khi thêm DEM.
- DEM: tái sử dụng dependency sẵn có nếu phù hợp; nếu thiếu thì dùng `rasterio`
  tùy chọn. Bản đầu hỗ trợ WGS84 địa lý, height ellipsoid; kiểm datum, nodata
  và coverage. Không coi độ cao orthometric là ellipsoid height.
- Chỉ dùng nominal roll law khi nhánh DAD cho phép, với AUX_INS và nguồn
  trạng thái tường minh; không dùng để che thiếu attitude không được phép.

**Đầu ra:** geometry kiểm được độc lập và báo thiếu coverage trước tính nặng.
Nếu geometry bắt buộc không khả dụng thì dừng rõ ràng, không fallback bằng 0.

### Bước 4 — Hoàn thiện correction trước DCE

- Đọc cờ/tham số AUX_PP1/INS/CAL theo đúng S4/VV và product type.
- Giữ chuỗi I/Q bias → gain → quadrature đã có; không trừ bias hai lần.
  Receiver correction dùng complex raw chia căn power gain, không chia power.
- Nối PG theo cờ/source AUX, tái sử dụng nominal reference normalization và
  inverse-response convention đã kiểm chứng. Delay availability độc lập với
  PG amplitude/phase QC; không bắt triển khai lại phần đã có.
- Timing: áp internal delay, SWST bias, fractional alignment và guard đúng
  một lần; correction EAP/RSL nhận range axis cuối cùng, không range L1.
- Dựng EAP bank từ complex LUT AUX_CAL, angular grid, antenna frame và roll
  steering. Dùng tâm LUT/angular increment; không lấy nominal near/far angle
  làm biên LUT, không chuẩn hóa tùy ý amplitude.
- EAP dùng `sqrt(abs(EAP))/EAP` theo quy ước complex pattern DAD; kiểm zero,
  finite values và ngoài miền. Mọi floor/threshold cần thiết là input tường minh.
- RSL amplitude dùng `(R/reference_range_m)^1.5`. `reference_range_m` và EAP
  update interval phải có nguồn AUX/job; nếu chưa có thì báo thiếu, không lấy L1.
- Áp EAP/RSL sau range compression, trước DCE, theo cờ AUX và đúng range/time.
  Không coi cả hai bắt buộc cho mọi cấu hình bất kể cờ xử lý.
- Extended Tapered RRF vẫn là phần nghiên cứu chưa chốt: kiểm nominal chirp
  AUX, FFT/sign convention, support và normalization trước. Chế độ strict
  báo chưa hỗ trợ nếu cấu hình yêu cầu mà công thức chưa xác nhận. Research
  chỉ chạy candidate được chỉ định và lưu tên; không chỉnh để tối ưu RMSE L1.

**Đầu ra:** correction ledger gồm requested/applied/unavailable, nguồn và
support. Đủ preflight không tự chứng minh radiometry hoặc tương đương IPF.

### Bước 5 — Chọn nguồn DC và fallback có kiểm soát

- Hỗ trợ `Data Analysis`, `Geometry`, `Predefined` theo cấu hình AUX.
  Predefined polynomial cần origin/time/range support tường minh.
- Với Data Analysis, tính/kiểm geometry trước fine estimation, unwrap và fit.
- Khi data không đủ, ACCC không hợp lệ, unwrap/fit không đạt hoặc RMS vượt
  threshold S4, trả geometry theo chính sách DAD cho block đó.
- Chỉ bắt lỗi chất lượng dữ liệu được định nghĩa. Lỗi cấu hình, sai shape,
  geometry/coverage hoặc lỗi lập trình phải dừng; không catch mọi `ValueError`.
- Mỗi block lưu `selected_source`, `selection_reason`, geometry/data DC nếu
  có, RMS, masks và ambiguity status. Giá trị không tính được là `null`.
- 33/2.243 record reference có cờ RMS vượt threshold. Đây là case cần test;
  trường `dcMethod=Data Analysis` không chứng minh data DC luôn được chọn.

**Đầu ra:** nguồn DC cuối có thể giải thích và không mất geometry vì data lỗi.

## 3. Quy ước API và cấu hình

Mở rộng module/hàm hiện có; cập nhật caller, exports nếu cần, cache fingerprint
và test. Không tạo framework cấu hình hoặc thay NumPy pipeline.

| Tác vụ | Đầu vào phải tường minh |
|---|---|
| DCE azimuth layout | Block size, nominal spacing (s), slice starts/stops/current slice, acquisition line times, tail convention, mapping tolerance và support policy |
| Range layout | ROI trên range-compressed support, số block, width, quy tắc đặt start/rounding |
| Data-DCE | PRF, unwrap FFT, polynomial degree, outlier/iterations, weighting, threshold RMS S4, source method và fallback policy |
| Geometry | Orbit/attitude provider, ellipsoid, số control point, frame/time convention và coverage policy |
| EAP | Complex LUT, angular grid, roll steering/antenna transform, terrain provider, update interval và out-of-domain policy |
| RSL | Slant range và reference range (m) |
| Calibration/RRF | Cờ correction, PG/chirp source, nominal coefficients, delay, spectrum candidate nếu nghiên cứu |

Tham số configurable của DAD phải vào hàm thực hiện tác vụ qua argument hoặc
config được truyền vào; hàm thuần không tự mở AUX/L1/config file. Dùng JSON
và CLI hiện có. Nguồn mỗi giá trị là L0/AUX/job/research; lưu đơn vị, XML path
hoặc config key, version/hash và trạng thái áp dụng.

Phân biệt hai trục: **processing mode** strict/research và **bằng chứng tham số**
AUX/documented/inferred. Strict yêu cầu đủ input và thuật toán hỗ trợ, nhưng
không đồng nghĩa IPF-equivalent: lịch 10 s vẫn inferred. Research cho phép
candidate/ablation/terrain hằng được chọn rõ; không làm mềm validation shape,
continuity, datum hay coverage. Không nhập một cờ strict chung để che các lý do
thiếu độc lập; preflight phải liệt kê từng mục.

## 4. Kiểm thử và tiêu chí nghiệm thu

| Nhóm | Kiểm thử/nghiệm thu |
|---|---|
| Độc lập L1 | Chạy profile nghiên cứu tường minh khi thư mục L1 vắng mặt; thay annotation comparison không làm đổi processing result/cache |
| Context layout | K=1/2/3, trước/sau ngưỡng H/S, single/nonfinal/final, shared boundary, center ngoài output product, block thiếu support, PRI/gap thay đổi |
| Biên và range | Hai tail convention; stop exclusive/last line; không clamp; ROI sau correction; 20 × 1.000 khi đủ support; lỗi rõ khi thiếu |
| Metadata regression | Checker giữ start/stop/center của 787 XML và mọi lỗi; không coi record chung là mẫu độc lập; không biến 5 µs thành processing tolerance |
| Datatake | Qua ranh orbit, ID trùng khác epoch, thiếu slice/context bất nhất |
| Geometry | Ví dụ quaternion ESA, chiều quay, frame/time/units; biên coverage cho mọi điểm truy vấn; cấm extrapolation |
| Correction | EAP phức giữ pha, zero/ngoài LUT/DEM nodata, RSL mũ 1,5; delay còn khả dụng khi PG QC không đạt; không áp timing hai lần |
| DCE/fallback | DC tổng hợp biết trước, ambiguity ±PRF, dữ liệu zero/NaN, thiếu điểm fit, RMS cao; kiểm nguồn/reason/null; config lỗi không bị nuốt |
| Provenance/cache | S4 không dùng RMS S6; thay context/tail/spacing/correction/provider làm đổi fingerprint |

Giữ regression layout cũ dưới profile riêng để đo tác động từng correction.
Với layout mới, so trên cùng tọa độ time/range và miền giao, không ghép theo
index hoặc nearest block không giới hạn khoảng cách. Báo khoảng lệch thời gian
và không so ngoài miền. Không yêu cầu data DC trùng L1 để nghiệm thu đúng luồng.

Decoder reference SAR-MPC là kiểm chứng tùy chọn, không tự tải trong pipeline.
Fixture S6 phụ thuộc file thật phải được kiểm tra lại hiện trạng trước khi sửa;
logic thuần dùng fixture tổng hợp có nhãn, integration dùng đường dẫn tùy chọn.
Không tạo annotation giả để che lỗi dữ liệu thật.

37 test liên quan đã pass trước đợt triển khai này; không phải số lượng test
của toàn repo. Khi triển khai, chạy unittest phù hợp bằng interpreter dự án
với `MPLBACKEND=Agg`, rồi cập nhật README/AGENTS/workflow và bảng
DAD → implementation → test → giới hạn cho từng bước.

## 5. Phạm vi và điều kiện dừng

Bước triển khai kế tiếp nên là **Bước 1–2: tách L1 và layout theo acquisition
context**, giữ correction hiện có để đo thay đổi. Sau đó geometry/coverage,
correction còn thiếu, rồi fallback. Không tuyên bố strict pipeline hoàn chỉnh
cho tới khi các nhánh bắt buộc của cấu hình thực tế đều được hỗ trợ.

Chưa mở rộng TOPS/S6, chưa sửa pipeline focus SLC hoặc RFI/noise processing
ngoài nhu cầu DCE. Product nominal khoảng 25 s không phải DCE spacing hay độ
dài datatake cố định. Cụm sensing span bội 5 s A/B và 3 s C/D chỉ dùng trong
chẩn đoán; chưa có cơ sở đưa thành quy tắc dựng acquisition.

Các điểm chưa chốt: nguồn internal slice context cho L0 hiện có, nguyên nhân
hai tail lệch một PRI, frame attitude và công thức Extended Tapered. Khi thiếu
nguồn/coverage/thuật toán bắt buộc, kết quả đúng là báo thiếu trước tính nặng.

## Tài liệu đối chiếu

- [DAD 2.5 cục bộ](../references/ESA2022_Sentinel-1%20Level%201%20Detailed%20Algorithm%20Definition_bookmarked.pdf),
  §4–5, §5.6, §6.1, §8, §9.4–9.6, §9.20; [workflow đã trích](sentinel1_l0_to_l1_workflow.md).
- [Khảo sát 787 annotation và artifact](sentinel1_dce_datatake_patterns.md).
- [SentiWiki Products](https://sentiwiki.copernicus.eu/web/s1-products): nominal
  slicing SM/IW khoảng 25 s, phân biệt product slicing với acquisition.
- [ESA processor logical view](https://s1.pages.eopf.copernicus.eu/s1-l12-rp/0.0.0/sdd/view-logical.html):
  tách workflow khỏi thuật toán và truyền tham số processing tường minh;
  tham khảo thiết kế, không mặc định cùng implementation IPF đang đối chiếu.
- [EOCFI-NOTE-052 issue 2.1](https://forum.step.esa.int/uploads/default/original/2X/3/315f415284743cf42be03a464121df1ca337a26d.pdf):
  nguồn cho tác vụ kiểm chứng attitude/frame, chưa tuyên bố đã nghiệm thu ví dụ số.
- [Auxiliary Product Specification 3/16](https://sentiwiki.copernicus.eu/__attachments/a_ac101207bbd6069e56e3922c261317dac89cb2f1bc1b44e0696bad24def0df0f/S1-RS-MDA-52-7443%20-%20Sentinel-1%20IPF%20Auxiliary%20Product%20Specification%202024%20-3-16.pdf):
  hợp đồng AUX; áp đúng loại/version, không lấy ghi chú XSD cũ làm chuẩn payload mới.
- [SAR-MPC test data](https://sar-mpc.eu/test-data-sets/): nguồn integration tùy chọn.

## Cập nhật kết quả triển khai — 22/10/2026

Kế hoạch ở trên giữ mốc và lý do thiết kế. Theo dõi phần đã triển khai,
bằng chứng và giới hạn trong [báo cáo triển khai](sentinel1_s4_dce_implementation.md).
Đã kiểm ví dụ quaternion/matrix ESA và frame GM2000; frame attitude không
còn là điểm chưa kiểm bằng ví dụ số. Đã tách L1, thay lịch H/K, chia scope
PRI/config/gap, nối correction độc lập và fallback có lý do. Công thức
Extended Tapered, nguồn context IPF của bộ L0 hiện có và lựa chọn tail duy nhất
vẫn chưa xác nhận. Strict dừng khi thiếu thuật toán/nguồn bắt buộc, theo đúng
điều kiện nghiệm thu; chạy research không chứng minh tương đương IPF.
