# Khảo sát DCE và thời lượng product/datatake từ 2020 trở đi

## Phạm vi và dữ liệu kiểm chứng

Khảo sát ngày 22/10/2026, dùng **787 annotation Stripmap SLC** từ S1A/B/C/D,
S1–S6, gồm **2.243 DC estimate**. Có 729 sản phẩm thuộc các cửa sổ catalogue
2024+ đã chọn được tải đầy đủ annotation co-pol (HH/VV) cùng manifest;
58 sản phẩm 2020/2023 bổ sung độ phủ lịch sử. Tất cả file tải qua CDSE Nodes,
có UUID, URL, số byte, SHA-256 và provenance. Không tải measurement hoặc
dùng L1 làm input cho pipeline L0/AUX.

| Năm | Annotation trong cohort chính |
|---|---:|
| 2020 | 39 |
| 2023 | 19 |
| 2024 | 91 |
| 2025 | 393 |
| 2026 | 245 |
| Tổng | 787 |

Theo vệ tinh: S1A 402, S1B 19, S1C 253, S1D 113. Theo swath: S1 92,
S2 102, S3 159, S4 96, S5 84, S6 254. IPF từ 003.20 tới 004.03.
81 file trước 2020 đã có trên đĩa được loại khỏi cohort chính.

Các cửa sổ gần đây gồm tháng 09/2024, 03/2025, 10/2025, 09/2026 và ngày
30/03/2026. Có thêm tháng 06/2020, ngày 30/03/2020, tháng 09/2023.
Đây là toàn bộ kết quả trong những cửa sổ đã ghi nhận, **không phải toàn bộ
archive CDSE từ 2024** và không phải mẫu ngẫu nhiên toàn cầu. Nhiều vị trí
được thu lặp lại; không coi mỗi record là một phép kiểm độc lập.

Sau loại trùng UUID, catalogue trong phạm vi 2020+ có **968 sản phẩm / 764
nhóm datatake-swath**; annotation được phân tích có **593 nhóm
mission/datatake/swath/polarisation/context**. Trong đó 587 nhóm đủ slice,
6 nhóm S1C tháng 03/2025 còn thiếu một số product so với sliceList.
Đã tìm lại bằng seed ±2 giờ; catalogue không trả thêm SLC cho các điều kiện
đó. Không suy ra rằng L0 mất dòng hoặc datatake bị đứt từ việc thiếu product L1.

## 1. Quy luật chọn cửa sổ DCE

### Kích thước và range layout

- Cả 2.243 record đều có **20 điểm fineDce theo range**.
- Duration/PRI khớp **6.000 dòng azimuth**; sai lệch biểu diễn timestamp
  dưới 0,0033 dòng. Không lấy 6.000 làm số giây hoặc FFT azimuth focusing.
- Nếu block range đầu bắt đầu tại `t0`, công thức
  `2 * (firstFineDceSlantRangeTime - t0) * rangeSamplingRate` cho width
  khoảng **1.000 mẫu**, sai khác tối đa 0,000112 mẫu trong cohort này.
- Chênh lệch giữa bước range-center lớn nhất và nhỏ nhất trong mỗi record
  tối đa khoảng 1,013 mẫu, phù hợp cách rải các cửa sổ gần đều trên lưới
  mẫu nguyên. Width suy ra vẫn mang giả định block đầu tại t0; không phải
  bằng chứng duy nhất về mọi chi tiết đặt support của IPF.

### Khoảng cách azimuth theo slice context

Đặt `a[i]`, `b[i]` là acquisition start/stop trong sliceList; `B=6000`,
`PRI` theo downlink và `S` là khoảng danh nghĩa (giây):

```text
H = a[i+1] - a[i]       nếu còn slice kế tiếp
H = b[i]   - a[i]       nếu là slice cuối hoặc chỉ có một slice
K = max(1, floor(H / S))
regular_start[j] = a[i] + j*H/K, j=0,...,K-1

Slice chưa cuối: thêm start tại a[i+1].
Slice cuối: thêm một block đủ B dòng ở sát biên cuối; xem hai quy ước bên dưới.
```

Với **S=10 s**, tất cả 787 annotation khớp lịch khi xét đúng quy ước tail.
Phép kiểm giữ nguyên S=10, B=6000 và tolerance 5 µs; không fit thêm offset
để giảm residual. Kiểm riêng start, stop và center, cùng số lượng record; sai số lớn nhất
của model khớp là **3,300 µs**.

Từ số lượng record, dưới giả định `K = numberOfDcEstimates - 1`, S bị ràng
buộc trong khoảng **9,9999675 s < S ≤ 10,000054 s**. Điều này củng cố rất
mạnh giả thuyết khoảng danh nghĩa 10 giây, nhưng chưa thay thế được đặc tả
tham số IPF. Các H quan sát được tới khoảng 39,085 s, kiểm K=1–3; không
khẳng định cách xử lý mọi đoạn unsliced dài hơn từ tập này.

Bước thực tế là H/K và có thể thay đổi khi H qua một ngưỡng. Ví dụ H gần
25 s cho khoảng 12,5 s, H gần 27 s cho khoảng 13,5 s. Do đó API phải nhận
spacing theo giây và context các slice; không suy spacing từ riêng B/2.

### Biên cuối còn khác nhau một PRI

Cohort 2020+ khớp một trong hai phương án:

```text
last_line:       tail_start = b[i] - B*PRI
after_last_line: tail_start = (round((b[i]-a[i])/PRI) + 1 - B)*PRI + a[i]
```

Các slice chưa cuối có cùng lịch ở cả hai model, nên không dùng chúng để
đếm bằng chứng phân biệt tail. Không có single/final slice 2020+ ủng hộ
phương án đặt start block cuối tại b[i].

**Không thể chọn tail chỉ từ phiên bản IPF, swath, polarization, PRI và
sensing span.** Có 26 nhóm cùng các tham số nhìn thấy đó nhưng xuất hiện
cả hai loại tail. Một ví dụ S1A/S1/VV/IPF 003.92:

- `s1a-s1-slc-vv-20251010t025029-20251010t025043-061357-07a805-002.xml`:
  stop cuối gần b[i].
- `s1a-s1-slc-vv-20251005t114536-20251005t114551-061290-07a54e-002.xml`:
  stop cuối gần b[i]+PRI.

Cả hai có PRI `0.0005345716926237736 s`, span `15.002220 s`. Thử floor
span/PRI đơn thuần cũng không giải thích toàn bộ các trường hợp. Chưa
quy nguyên nhân cho rounding, phần raw được giữ lại hoặc một cấu hình
không được annotate. Pipeline dùng L0 phải dựng block theo số dòng/support
thực, giữ convention tường minh; không dùng chọn model khớp L1 như input.

### Method và chất lượng DC

Cả 787 sản phẩm ghi cấu hình **Data Analysis**. Có **33/2.243 record** gắn
cờ `dataDcRmsErrorAboveThreshold=true`; các record còn lại false, không
thiếu cờ. Median RMS khoảng 2,234 Hz, lớn nhất khoảng 240,626 Hz.
Cấu hình method không chứng minh data-derived DC đã được dùng cho mọi
record. Theo DAD §5, cần giữ đường geometry và fallback khi chất lượng
không đạt; không suy DC từ giá trị reference để làm xanh phép kiểm.

## 2. Xu hướng độ dài dải quét, datatake và product

### Product được chia theo lịch nominal 25 giây

| Vị trí product | Số mẫu | First-to-last line span, min / median / max (s) |
|---|---:|---|
| Chỉ một slice | 523 | 8,839681 / 23,679301 / 33,899448 |
| Slice đầu | 64 | 28,144043 / 28,264874 / 28,381480 |
| Slice giữa | 133 | 24,999009 / 24,999427 / 24,999658 |
| Slice cuối | 67 | 10,501491 / 23,935992 / 34,928021 |

Span ở đây là last-line minus first-line, chưa cộng một sample interval.
Các ranh output liền nhau cách một line interval, quan sát 0,000519–0,000620 s.
Do đó các slice giữa thể hiện lịch **xấp xỉ 25 giây**, phù hợp
[SentiWiki về product slicing](https://sentiwiki.copernicus.eu/web/s1-products).
Slice đầu/cuối chịu ảnh hưởng overlap, cắt support và quy tắc xử lý đoạn cuối
ngắn; không dùng duration của chúng để suy một nominal product length khác.

Theo lưới pixel azimuth L1, các product giữa dài khoảng **168,5–171,0 km**,
median 170,5 km. Đây là `(numberOfLines-1)*azimuthPixelSpacing`, không phải
phép đo geodesic của quỹ đạo hoặc bề rộng swath theo range.

### Internal slice start khác lịch product output

Trong context đã đọc:

- Khoảng giữa hai internal slice start: **23,750236–26,744772 s**,
  median khoảng 25,491768 s.
- Overlap theo stop[i]−start[i+1]: **4,447542–7,804660 s**,
  median khoảng 5,865420 s.

DAD §8.2.1 yêu cầu internal signal slices bám lưới block azimuth focusing,
phụ thuộc FFT length, overlap và origin chung. DAD §8.3 mô tả lịch output.
Vì vậy nominal product length 25 s không ấn định internal start step 25 s,
và cũng không ấn định DCE spacing 12,5 s trong mọi cảnh.

### Datatake có thể dài nhiều product và qua ranh orbit

Extent từ first slice start tới last slice stop trong metadata nằm trong
**9,998376–340,041285 s**, có nhóm tới **14 slice**. Đây là extent
signal/internal được annotate, chưa bao gồm chắc chắn toàn bộ preamble,
postamble hoặc sensing period của L0. Catalogue cho envelope sản phẩm,
cũng không thay thế được planned acquisition duration.

**Sáu nhóm datatake đi qua hai absolute orbit.** Do đó khóa nhóm đúng cho
phép kiểm này là mission + datatakeId + swath + polarization + context
origin; lưu danh sách orbit như thuộc tính. Bộ lọc seed cũng không khóa
orbit. Sửa này nhận diện được **193 cặp cửa sổ acquisition chung** giữa
các product kề nhau; so trực tiếp timestamp ZD hoặc khóa theo orbit sẽ bỏ
sót một phần các cặp.

### Thời lượng echo ở scene chỉ có một slice có tính cụm

| Vệ tinh | Single-slice scene | Các mốc sensing span sau làm tròn (s) | Sai lệch lớn nhất tới lưới |
|---|---:|---|---:|
| S1A | 272 | 10, 15, 20, 25, 30, 35 | 9,226 ms tới bội số 5 s |
| S1B | 17 | 15, 20, 25, 30 | 7,788 ms tới bội số 5 s |
| S1C | 160 | 15, 18, 21, 24, 27, 30 | 8,363 ms tới bội số 3 s |
| S1D | 74 | 12, 15, 18, 21, 24, 27, 30 | 8,363 ms tới bội số 3 s |

Đây là xu hướng mạnh trong cohort, không phải yêu cầu được hardcode vào
processor. Tập mẫu có nhiều acquisition lặp lại trên các vị trí nhất định;
chưa tách được ảnh hưởng của mission planning, timeline thiết bị và lưới
processing. [SentiWiki mission](https://sentiwiki.copernicus.eu/web/s1-mission)
có mô tả observation scenario và nguồn acquisition plans; cần đối chiếu
nguồn đó nếu muốn xác định vì sao một datatake được đặt thời lượng cụ thể.

## 3. Cấu hình đề xuất cho bước triển khai tiếp theo

1. Giữ B=6.000 dòng, 20 block range; width khoảng 1.000 mẫu theo phép suy
   đã nêu. Các giá trị đi qua đầu vào hàm, không chôn trong implementation.
2. Dùng `nominal_spacing_s=10` cùng policy theo slice context; nhận slice
   starts/stops tường minh từ L0/job context hoặc bước dựng internal layout.
3. Tách acquisition timeline dùng xử lý khỏi ZD timeline dùng đối chiếu.
   Không lọc DCE chỉ vì tâm nằm ngoài output product extent.
4. Tail dùng số dòng thực còn đủ support; giữ khác biệt một PRI trong
   chẩn đoán. Không tự chọn convention theo vệ tinh/IPF hoặc tối ưu residual L1.
5. Nominal product length 25 s thuộc slicing/output. Datatake acquisition
   length là input/context thu nhận; không suy một độ dài cố định từ DCE.
6. Giữ geometry fallback và các trạng thái chất lượng riêng. XML reference
   chỉ được dùng trong báo cáo/test, không đưa correction/DC từ L1 vào core.

Lượt khảo sát này chưa đổi thuật toán DCE/focusing. Công cụ được sửa cách
nhóm datatake qua orbit và thêm quản lý phiên token, báo cáo cohort, biểu đồ.

## 4. Tái lập và artifact

```bash
python workflows/sentinel-1/survey_dce_patterns.py \
  ~/sar_data/focusing_data/copernicus_metadata \
  --min-year 2020 \
  --catalogue-directory output/sentinel-1/copernicus/survey \
  --output output/sentinel-1/copernicus/survey/patterns.json \
  --figure output/sentinel-1/copernicus/survey/patterns.png
```

- [JSON chi tiết](../output/sentinel-1/copernicus/survey/patterns.json): từng
  annotation và SHA-256, hypotheses/residual, catalogue, subset và nhóm.
- [Biểu đồ](../output/sentinel-1/copernicus/survey/patterns.png): product
  duration, DCE spacing, datatake extent và cụm sensing span.
- `recent_catalogue.json`, `recent_download_index.json`: toàn bộ 729 product
  trong các cửa sổ 2024+; các file `fill_*.json` ghi truy vấn bổ sung cho nhóm thiếu.
- `session_refresh_check.json`: đã refresh token thật, tải lại manifest qua
  authenticated GET và xác nhận SHA-256 giống cache. Không có token trong report.
- 37 test liên quan pass; fixture tổng hợp kiểm cả grouping qua orbit,
  quy ước biên, cohort, cache và authentication. Artifact đầu ra và dữ liệu
  ngoài repo không được commit.

Nguồn thuật toán: [DAD 2.5](https://sentiwiki.copernicus.eu/__attachments/a_756fb6a7b0748da60a84b616db0d25be9f6a5115eeb185b38bfd75c6382d562e/DI-MPC-IPFDPM%20-%20Sentinel-1%20Level%201%20Detailed%20Algorithm%20Definition%202022%20-%202.5.pdf?cb=68b1b7de2b2941fa5cfc5745b54238dd),
§5.6 và §8. DAD xác định các tham số configurable và quan hệ slicing;
các số 6.000 / 20 / 1.000 / 10 s ở trên được kiểm bằng metadata, không
gán ngược thành nội dung mà DAD đã công bố.
