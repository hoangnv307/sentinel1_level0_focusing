# Kiểm chứng khoảng cách azimuth block DCE từ annotation Stripmap

## Kết luận và phạm vi

**Không dùng bước 3.000 dòng làm preset mặc định.** Đây là giả định overlap
50% trong đề xuất trước, không khớp các annotation hiện có. Giữ block dài
6.000 dòng và 20 block range. **Cập nhật 22/10: phải xét slice context**:
slice chưa cuối chia khoảng từ start hiện tại đến start slice kế tiếp,
còn slice cuối mới dùng sensing span của nó. Khoảng danh nghĩa gần 10 giây
vẫn là giả thuyết; khoảng cách thực tế không cố định bằng 10 giây.
Quy ước biên cuối khác nhau giữa hai bộ mẫu, chưa có một default chung.

Khảo sát ngày 21/10/2026 gồm 5 annotation mới trong
`~/sar_data/focusing_data/l1_sm_annotation_metadata/` và annotation S4/VV của
`product_1/level-1/`: tổng cộng **6 cảnh, 17 estimate**, S1–S5, VV/HH.
Sáu cảnh ban đầu có một downlink record và một phần tử `sliceList/slice`, slice number 1.
Không có L0 tương ứng cho 5 cảnh mới, nên số dòng của chúng được suy từ
PRI và sensing span; S4 có thể đối chiếu 16.500 dòng echo L0 đã kiểm kê.

Ngày 22/10 bổ sung 3 sản phẩm S1C/S6/HH cùng datatake: **tổng cộng 9 XML,
25 estimate**, trong đó 2 cặp estimate trùng cửa sổ acquisition. Phần dưới
giữ kết quả khảo sát một slice để đối chiếu, sau đó cập nhật quy tắc nhiều slice.

Đây là bằng chứng timing/layout từ L1, không dùng giá trị Doppler để fit
và không đưa polynomial L1 vào pipeline. Chưa đủ bằng chứng tuyên bố đây
là quy tắc IPF cho mọi cảnh; mới có một datatake gồm nhiều slice, chưa kiểm TOPS/WV.

## Khảo sát ban đầu: sáu cảnh chỉ có một slice

Các XML path dùng trong phép kiểm:

- `generalAnnotation/downlinkInformationList/downlinkInformation/firstLineSensingTime`,
  `lastLineSensingTime`, `downlinkValues/pri`.
- `imageAnnotation/imageInformation/sliceList/slice/sensingStartTime`,
  `sensingStopTime`: kiểm tra extent trùng downlink; không dùng thời lượng ảnh output.
- `imageAnnotation/imageInformation/zeroDopMinusAcqTime`: ánh xạ acquisition
  time sang thời gian annotation trong phép chẩn đoán này.
- `dopplerCentroid/dcEstimateList/dcEstimate/`: `fineDceAzimuthStartTime`,
  `fineDceAzimuthStopTime`, `azimuthTime`, số phần tử `fineDceList/fineDce`.

Quy ước:

```text
a = firstLineSensingTime                     # acquisition time
b = lastLineSensingTime                      # thời gian dòng cuối, inclusive
PRI = downlinkValues/pri
T = b - a                                   # không cộng một PRI vào span này
N = round(T / PRI) + 1                       # số dòng được suy từ metadata
B = azimuth_block_size_lines = 6000
D = B * PRI                                 # độ dài block theo biên exclusive
z = zeroDopMinusAcqTime                      # chỉ dùng đối chiếu L1
S = nominal_spacing_s = 10                   # giả thuyết, configurable
K = max(1, floor(T / S))

regular_starts[j] = a + j * T / K            # j = 0,...,K-1
tail_start = a + (N - B) * PRI
```

Chỉ giữ regular start có đủ B dòng support, thêm tail start, sắp xếp và
loại trùng. Tập start không vượt `N-B`; `N<B` phải báo thiếu support.
Start/stop/center dự đoán trong annotation lần lượt bằng
`start+z`, `start+z+D`, `start+z+D/2`.

**Hai quy ước biên khác nhau là có chủ ý:** chia khoảng đều dùng `T=b-a`,
còn block cuối phủ tới biên exclusive `b+PRI`. Thay T bằng `N*PRI` làm lệch
các estimate giữa cảnh. Không lấy `numberOfLines` hay product first/last
line của ảnh SLC thay cho extent echo: ảnh đã loại support ở hai mép.

| Cảnh | T (s) | N suy ra | K | Start quan sát, tương đối với DCE start đầu (s) | Số estimate |
|---|---:|---:|---:|---|---:|
| S1/VV, 19/10 23:42:36 | 30,004974 | 56.130 | 3 | 0; 10,001658; 20,003316; 26,798079 | 4 |
| S1/VV, 21/10 02:07:19 | 25,004056 | 46.775 | 2 | 0; 12,502028; 21,797161 | 3 |
| S2/HH, 17/10 21:35:27 | 25,002461 | 40.325 | 2 | 0; 12,501231; 21,282847 | 3 |
| S3/HH, 18/10 15:35:48 | 14,999821 | 28.875 | 1 | 0; 11,883388 | 2 |
| S5/VV, 18/10 18:53:24 | 29,997166 | 56.970 | 2 | 0; 14,998583; 26,838379 | 3 |
| S4/VV, 03/09 18:36:28 | 9,998376 | 16.500 | 1 | 0; 6,362990 | 2 |

Mọi estimate có width tương ứng 6.000 PRI trong sai số timestamp và có
20 điểm fineDce. Sai số lớn nhất của **cả start, stop và center** so với
giả thuyết là **2,659 µs**, nhỏ hơn tolerance 5 µs, khoảng 0,0044 PRI ở S4.
Không fit thêm một time offset để giảm sai số.

S4 cho kết quả starts `[0, 10500]` trên lưới L0. Hai cửa sổ
`[0,6000)` và `[10500,16500)` có **khoảng trống 4.500 dòng**; không overlap.
DCE lấy mẫu thống kê theo thời gian, không đòi hỏi phủ kín mọi dòng raw.

Các start giữa cảnh có thể nằm ở tọa độ dòng phân số trong annotation:
ví dụ S5 gần 28.484,5 dòng và S1 ngày 19/10 gần 18.709,667 dòng.
Không thể suy duy nhất cách IPF lượng tử hóa cửa sổ thực tế từ timestamp
này; khi nối vào L0 phải chọn dòng gần nhất theo timeline thật và ghi
residual, không tuyên bố exact sample parity.

## Bằng chứng ban đầu cho khoảng danh nghĩa gần 10 giây

- Bước đều **3.000 dòng + tail** không khớp số lượng/vị trí của cả 6 cảnh.
- Bước đều **10 giây + tail** chỉ khớp S4 (chỉ có đầu/cuối); sai ở 5 cảnh mới.
- Quy tắc **đầu + midpoint + tail** giải thích một số cảnh nhưng không giải
  thích S1 30,005 s có bốn estimate và S3/S4 chỉ có hai.
- Chia theo `floor(T/S)` giải thích được sự khác nhau giữa S1 **30,004974 s**
  và S5 **29,997166 s**. S5 có nhiều dòng hơn S1 nhưng ít estimate hơn,
  nên số dòng đơn thuần không giải thích được số block trong hai mẫu này.

Dưới giả định mỗi cảnh giữ đủ K regular start và một tail riêng biệt,
số estimate quan sát ràng buộc cùng một S vào khoảng:

```text
9,999055333... s < S <= 10,001658 s
```

10 giây là ứng viên tự nhiên trong khoảng đó, **không phải giá trị được
xác định duy nhất**. Quy tắc khác vẫn có thể khớp tập mẫu nhỏ này.

[DAD §5.6](https://sentiwiki.copernicus.eu/__attachments/a_756fb6a7b0748da60a84b616db0d25be9f6a5115eeb185b38bfd75c6382d562e/DI-MPC-IPFDPM%20-%20Sentinel-1%20Level%201%20Detailed%20Algorithm%20Definition%202022%20-%202.5.pdf?cb=68b1b7de2b2941fa5cfc5745b54238dd)
quy định kích thước block và khoảng cách azimuth là configurable; cho phép
overlap nhưng không bắt buộc overlap 50%, không ấn định 3.000 dòng hay
10 giây. Slicing §8 phân biệt extent input, internal processing và output;
không lấy công thức overlap của azimuth focusing làm công thức spacing DCE.

## Bổ sung 22/10: datatake có ba sản phẩm liên tiếp

Thư mục thực tế:
`~/sar_data/focusing_data/l1_sm_datatake_annotation/sm_random_datatake_annotation_metadata/`.
Ba file là S1C/S6/HH, ngày 30/03/2025, absolute orbit 1674, datatake
`002C4C` (11340). Cả ba ghi cùng một `sliceList` gồm 3 slice và cùng
anchor time; `imageInformation/sliceNumber` lần lượt 1, 2, 3. Đây là dữ liệu
tham chiếu metadata, không xác nhận khả năng decode L0 S1C của pipeline hiện tại.

### Lịch theo context của slice, không reset theo downlink span

Thời gian trong bảng là acquisition UTC ngày 30/03/2025:

| Slice | Sensing start | Sensing stop | Span downlink | Start kế tiếp − start hiện tại |
|---|---|---|---:|---:|
| 1 | 18:04:11.826667 | 18:04:43.025753 | 31,199086 s | 25,356509 s |
| 2 | 18:04:37.183176 | 18:05:08.382262 | 31,199086 s | 25,356509 s |
| 3 | 18:05:02.539685 | 18:05:17.807092 | 15,267407 s | Không có |

Với hai slice chưa cuối, chia riêng span 31,199086 s như giả thuyết cũ
sẽ cho K=3, bốn estimate gồm tail, và các start sai. Thực tế mỗi sản phẩm
có **ba estimate**: start slice hiện tại, midpoint giữa hai slice start,
và start slice kế tiếp. Bước giữa các block là **12,6782545 s**, không phải
31,199086/3 và cũng không phải 10 s cố định.

Tập start acquisition dự đoán toàn datatake, tương đối với start slice 1:

```text
0; 12,6782545; 25,356509; 38,0347635; 50,713018; 62,373524724...
```

Chọn record cho từng sản phẩm theo context và giữ đầu chung ở ranh giới:

```text
slice 1: [0;          12,6782545; 25,356509]
slice 2: [25,356509;  38,0347635; 50,713018]
slice 3: [50,713018;  62,373524724...]
```

Khái quát ứng viên, với các slice start `a[i]` và stop `b[i]`:

```text
H[i] = a[i+1] - a[i]    nếu còn slice kế tiếp
H[i] = b[i] - a[i]      nếu là slice cuối
K[i] = max(1, floor(H[i] / nominal_spacing_s))
regular[j] = a[i] + j * H[i]/K[i], j=0,...,K[i]-1

slice chưa cuối: thêm start a[i+1], kiểm đủ support của block
slice cuối: thêm tail với quy ước biên được chọn tường minh
```

Ở datatake này, K=2,2,1. Ba XML khớp mô hình khi chọn biên cuối
`last_line`, sai số start/stop/center tối đa lần lượt **1,000; 1,000;
2,276 µs**. Giả thuyết chia riêng từng downlink span không khớp cả ba.
Hai slice chưa cuối mới chỉ kiểm được trường hợp K=2; chưa chứng minh
công thức floor đúng cho mọi chiều dài khoảng giữa các slice start.

### Estimate chung và thời gian của từng sản phẩm

Offset `zeroDopMinusAcqTime` của ba sản phẩm khác nhau: **0,358752;
0,345526; 0,359954 s**. Vì vậy không so timestamp DCE L1 trực tiếp.
Trừ đúng offset của từng sản phẩm cho cả start, stop và center cho kết quả:

| Cặp record (0-based) | Acquisition start | Acquisition stop | Acquisition center |
|---|---|---|---|
| Slice 1 record 2 / slice 2 record 0 | 18:04:37.183177 | 18:04:40.790076 | 18:04:38.986626 |
| Slice 2 record 2 / slice 3 record 0 | 18:05:02.539686 | 18:05:06.146585 | 18:05:04.343135 |

Hai cặp có **sai số timing 0 µs ở độ chính xác XML**. Có thể suy lịch
acquisition chung, nhưng không coi toàn bộ DC record là bản sao: polynomial
data có khác biệt nhỏ giữa các sản phẩm, dù geometry polynomial trong hai
cặp này giống nhau.

Đặc biệt, estimate đầu của slice 2/3 có center ZD **trước** product first
line time khoảng 1,136/0,765 s. Không lọc record chỉ bằng điều kiện center
nằm trong ảnh output. Nhánh `slice_timeline` hiện tại trong core có điều
kiện này và lịch midpoint/tail cố định: **chưa đủ để tái tạo bộ ba mới**.

### Phản ví dụ cho quy ước tail cũ

Slice cuối mới có DCE stop acquisition bằng `lastLineSensingTime + 1 µs`.
Tail khớp `b - 6000*PRI`, không khớp `b + PRI - 6000*PRI`. Giữ quy ước
`after_last_line` cũ làm sai tối đa **600,868 µs**, gần một PRI
(601,150 µs), trong khi `last_line` giảm sai số xuống 2,276 µs.

Ngược lại, sáu cảnh S1A ban đầu khớp `after_last_line`. **Chưa biết nguyên
nhân khác biệt**: không tự gán cho S1A/S1C, phiên bản IPF hay slicing khi
chưa có thêm bằng chứng. Script nhận `--tail-stop-convention` tường minh,
không tự chọn mode theo mức độ khớp annotation. Giả thuyết một quy ước tail
cho tất cả cảnh hiện bị phản chứng.

## Cách tái lập và thay đổi đề xuất

```bash
/home/xiaoxin/python_envs/sentinel1/bin/python workflows/sentinel-1/check_dce_azimuth_layout.py
/home/xiaoxin/python_envs/sentinel1/bin/python workflows/sentinel-1/check_dce_azimuth_layout.py \
  ~/sar_data/focusing_data/l1_sm_datatake_annotation \
  --tail-stop-convention last_line \
  --output output/sentinel-1/dce_random_datatake_last_line.json
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python -m unittest discover -s test -p 'test_dce_azimuth_layout.py'
```

Script nhận file/thư mục XML ở positional arguments, cùng các tham số
`--azimuth-block-size-lines`, `--nominal-spacing-s`, `--time-tolerance-s`,
`--tail-stop-convention`, `--output`. Thư mục input được duyệt đệ quy.
Mặc định vẫn kiểm sáu XML ban đầu, dùng convention `after_last_line`.
JSON tại `output/sentinel-1/dce_azimuth_layout.json` lưu SHA-256 từng input,
timing quan sát, tọa độ dòng phân số, slice context, các model, residual và
các cặp cửa sổ acquisition chung. Exit code khác 0 nếu có cảnh không khớp.
`adaptive_span` trong JSON là giả thuyết cũ theo downlink; `slice_context`
là mô hình mới theo context với convention người dùng chọn. Cả hai được
giữ lại để thấy phản ví dụ, không thay kết quả sai bằng model tự fit.

Điều chỉnh phần layout trong kế hoạch cải thiện:

1. Bỏ preset bước 3.000 dòng. Giữ B=6.000 và 20 block range như đã thống nhất.
2. Đề xuất `nominal_spacing_s=10` và policy theo **slice context** như trên
   cho research preset; tham số đi vào hàm dựng layout, nguồn ghi
   `annotation_inference`. Tail convention phải tường minh, chưa có default
   tổng quát đủ bằng chứng cho cả hai bộ mẫu.
3. Khi triển khai độc lập L1, lấy a/b/PRI từ echo L0 và slice/job context.
   Không đọc annotation để lấy start mỗi lần chạy. Giữ offset z ở comparator;
   timestamp xử lý nội bộ dùng acquisition time.
4. Đã kiểm một bộ annotation gồm slice đầu/giữa/cuối. Ưu tiên thêm datatake
   nhiều slice, khoảng giữa start khác 25 s và các trường hợp ở gần ngưỡng
   20/30 s để thử phản chứng quy tắc. **Không cần L0 để kiểm quy tắc spacing**;
   L0 chỉ cần khi muốn kiểm ánh xạ sang dòng raw thực và chạy estimator.

**Lượt này chỉ thêm phép chẩn đoán và tài liệu.** `spacing`, `custom` và
`slice_timeline` trong core chưa thay đổi. Đặc biệt `spacing` hiện tự thêm
tail; `slice_timeline` còn dùng start/midpoint và lọc center theo product,
chưa phải triển khai của giả thuyết được kiểm chứng ở đây. Không thay đổi kết quả DCE
hoặc kích hoạt phần còn lại của kế hoạch.
