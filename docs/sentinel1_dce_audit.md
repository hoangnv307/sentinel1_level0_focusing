# Kiểm tra luồng và tham số DCE — Sentinel-1A S4/VV

Ngày kiểm tra: 20/10/2026. Dữ liệu: `~/sar_data/focusing_data/product_1`.
Cập nhật bước RRF ngày 21/10/2026 nằm trong
[báo cáo tiếp theo](sentinel1_rrf_followup.md); giữ nguyên số liệu lịch sử bên dưới.
**Đính chính RX:** các lượt cũ có receiver correction đã chia complex raw
cho GV thay vì **sqrt(GV)** của Eq. 9-16 (dấu căn bị mất khi trích text).
Xem [triển khai và kết quả RX đã sửa](sentinel1_receiver_gain_compensation.md).
Các số RMSE lịch sử bên dưới không đại diện cho công thức RX hiện tại.
Đính chính tiếp theo: nhánh PG reference cũ cũng đã nghịch đảo thêm một
lần giá trị inverse-PG trong annotation. Nay chuẩn hóa trực tiếp; xem
[PG L0/AUX và đối chứng cùng layout](sentinel1_pg_calibration.md).
Phạm vi là đường xử lý **L0 → input range-compressed → Doppler centroid**
của demo S4/VV. Các tham số dùng sau DCE được phân loại để tránh đưa nhầm
vào estimator; báo cáo này không xác nhận chất lượng ảnh SLC sau focus.

## 1. Cách đọc kết quả

Có hai đại lượng khác nhau:

- **Fit RMS:** độ lệch giữa các Fine DC và polynomial do cùng một lần ước
  lượng tạo ra. Đây là đại lượng so với ngưỡng AUX_PP1 `20 Hz`.
- **RMSE với L1:** độ lệch giữa kết quả demo và metadata của sản phẩm tham
  chiếu. Đây là kiểm tra khả năng tái hiện sản phẩm, không phải sai số so
  với Doppler thật ngoài thực địa.

Chế độ `reference` dùng I/Q bias, replica, PG và EAP từ L1 theo yêu cầu
người dùng. Vì thế, kết quả của chế độ này **không phải kiểm chứng độc lập
L0/AUX**. Tần số Fine DC và hệ số DC polynomial của L1 chỉ dùng sau khi
ước lượng để đối chiếu; không đưa vào tính tín hiệu hoặc hình học.

Layout 6.000 dòng × 20 range block được suy từ annotation, không phải
cấu hình IPF công bố đầy đủ. Khi thay đổi range layout, chỉ so Fine DC trên
giao support và nội suy tuyến tính L1; không ngoại suy. Khi đổi cửa sổ
azimuth, hai bên quan sát nội dung khác nhau theo thời gian, nên RMSE chỉ
là kiểm tra độ nhạy, không được dùng xếp hạng khả năng tái hiện IPF.

## 2. Luồng đang được kiểm tra

1. Đọc packet header; chọn hai echo chunk 13 và 14, đúng S4/RX V.
2. Giải mã FDBAQ thành complex I/Q bằng `sentinel1decoder 2.0.0`.
3. Trừ I/Q bias; trong chế độ tham chiếu tiếp tục chia RX gain và nhân
   PG correction complex tương đối.
4. Tạo nominal matched filter, bù internal delay nếu dùng tham chiếu;
   zero pad → FFT → nhân RRF → IFFT → lấy phần `slc` hợp lệ.
5. Sửa trục range bằng SWST bias, xử lý phần lẻ của thay đổi SWST trong
   miền tần số, ghép các segment trên lưới range chung bằng dịch nguyên mẫu.
6. Nhân EAP/RSL correction ở chế độ tham chiếu; bỏ mép xa thiếu chirp
   support khi advance do internal delay.
7. Chọn phần giao range hợp lệ và hai cửa sổ azimuth. Block 2 đi qua ranh
   giới chunk; không tính lag-1 qua khoảng thiếu dòng nếu có gap.
8. Tính ACCC lag-1, Fine DC baseband, coherence → unwrap theo range.
9. Tính geometry DC từ orbit/attitude L0 tại tâm block, giải ambiguity PRF.
10. Fit delta giữa data và geometry, cộng lại thành quadratic data DC;
    kiểm RMS và ghi nguồn data/geometry được chọn. Không thay Fine DC bằng
    annotation để giảm residual.

Công thức và checklist đầy đủ nằm trong
[workflow theo DAD](sentinel1_l0_to_l1_workflow.md), đặc biệt §5–§8 của
workflow và DAD §5, §6.1–6.2, §9.2–9.6.

## 3. Kiểm kê tham số và nguồn

| Nhóm / tham số | Giá trị thực tế hoặc quy tắc | Nguồn, đánh giá |
|---|---|---|
| Vệ tinh / mode / kênh | S1A / Stripmap S4 / VV, header swath number `3`, RX channel `0` | L0 và L1 khớp; không lấy preset S6 |
| Echo / kích thước | Chunk 13: 11.550 dòng; chunk 14: 4.950 dòng; mỗi dòng 21.154 complex samples | Đọc toàn bộ header của hai chunk; decode kiểm shape và finite theo batch |
| Packet / PRI counters | Bước tăng bằng 1, kể cả qua ranh giới chunk; error flag đều false | Không thấy mất packet/PRI trong phần echo |
| SWAP flag | Chunk 13 false, chunk 14 true | Cờ cập nhật tham số động; phải đọc lại SWST, không phải yêu cầu đổi I/Q |
| BAQ | FDBAQ MODE 0 | Decoder hiện có; chưa đối chiếu từng reconstruction LUT của thư viện với AUX_INS |
| PRI / PRF | `0.0006059988896297211 s` / `1650.168040094962 Hz` | Hằng trên toàn bộ echo headers |
| Timeline | Max chênh giữa bước timestamp và PRI `19.611462 µs`, nhỏ hơn guard `0.05 PRI` | Có lượng tử hóa timestamp; counter check bổ sung tránh suy luận gap chỉ từ timestamp |
| Range decimation / Fr | RGDEC 5; `56,302,083.36 Hz` | PDU S4 có L/M=3/8; Fr=4×F_REF×3/8 |
| SWL / số quads | `0.0003771174836326696 s`; SWL code 14.155; 10.577 quads | Công thức Stage-3 PDU cho đúng 21.154 complex samples |
| Suppressed time | `1.0656799254897057e-6 s` | `320/(8 F_REF)` khớp AUX_INS `deltaTSuppr` trong sai số số học |
| Rank | `9` | Header; trục tau có Rank×PRI, không chỉ dùng SWST |
| SWST | `90.28973168711532 µs` / `91.7816835828009 µs` | Thay đổi tương đương 84 range samples giữa hai chunk |
| SWST bias | `-8.2256909e-9 s` | AUX_INS S4/VV; dùng hàm `correct`, tránh đảo dấu thủ công |
| Tx ramp rate | `982835489094.9583 Hz/s` | L0 header, hằng trong hai chunk |
| Tx start frequency | `-25308314.710424766 Hz` | L0 header; phân biệt với hệ số linear của chirp dùng thời gian đã center |
| Tx pulse length / M | `5.1498982399290024e-5 s` / `ceil(Fr×TXPL)=2900` | L0; `slc` convention cho 18.254 mẫu/chunk trước delay guard |
| Range FFT | 32.768 điểm, đủ linear convolution | Suy từ N+M−1; không dùng FFT unwrap hoặc FFT azimuth thay thế |
| I/Q bias cơ bản | Chunk 13 `0.2883251961 + 0.3642456462j`; chunk 14 `0.2925056041 + 0.3656420610j` | Ước lượng từ tối đa 256 echo đầu mỗi chunk; chưa tái hiện raw analysis toàn cảnh IPF |
| I/Q bias tham chiếu | `0.2893930078 + 0.3621337116j` | L1 `support/*BiasUsedForCorrection`; AUX cho phép bias correction |
| I/Q gain / orthogonality | Đều tắt | Cờ AUX_PP1; không tự bật vì XML có số thống kê |
| RX gain header | `-6 dB`, hằng | Scale toàn cảnh không làm đổi lag-1 phase; không đồng nhất với RX variation |
| RX variation | Hệ số rational trend 8 phần tử, overshoot 5 phần tử, kênh V | Lượt cũ chia `10^(GV_dB/10)` là sai; DAD Eq. 9-16 phải chia căn của power GV |
| RX correction range | Gain 0,943959–1,026669 ở chunk 13; 0,944980–1,026922 ở chunk 14 | Không thấy mẫu số 0, gain không dương hoặc non-finite |
| Replica amplitude / phase | `[1,0,0,0]`; phase cycles `[0,-800.9333,4.914177e11,0]` | L1 reference replica; AUX nominal có thêm chữ số. Không so hai chirp khác gốc thời gian bằng trực tiếp coefficient |
| Internal delay | `4.333092e-7 s`, khoảng 24,40 mẫu | Advance bằng phase ramp dương; bỏ thêm `ceil(delay×Fr)=25` mẫu mép xa |
| PG | Hai node valid; nội/ngoại suy correction complex | Lượt cũ dùng p0/p(t) là sai chiều; annotation đã inverse PG, nay dùng value(t)/value(first) |
| PG phase drift | Equivalent endpoint phase rate `-0.000683759 Hz` | Nhỏ hơn residual Fine DC cỡ Hz; không đủ để giải thích residual chỉ bằng drift tuyến tính này |
| PG validity / coverage | Hai node hợp lệ; scene start không cần ngoại suy | Loại node có flag invalid; chưa phục hồi absolute radiometry |
| EAP | 7 node thời gian, mỗi node 574 mẫu complex | Đọc cặp re/im; nội suy complex LUT theo range rồi dùng `sqrt(abs(z))/z` |
| EAP time axis | Acquisition time + `zeroDopMinusAcqTime` | Chặn echo ngoài bank; không dùng thời gian acquisition trực tiếp với bank zero-Doppler |
| RSL | `(R / 800000 m)^1.5`, R=c×tau/2 | Amplitude correction; không dùng bình phương hoặc số mũ của power |
| ROI không delay | Phần giao 18.170 samples, origin `0.00554684559586702 s` | Suy từ L0/AUX; không dùng toàn union có thiếu support |
| ROI có delay | 18.145 samples | Do guard 25 mẫu; range centers cuối hơi khác L1, phải đối chiếu trên cùng tau |
| Azimuth layout | 6.000 dòng; starts `[0,10500]`; hai block có khoảng cách không phải 3.000 dòng | Suy start/stop L1 rồi kiểm trên timeline L0 |
| Range layout | 20 block, width 1.000 mẫu, trải đều trong ROI; center=start+width/2 | Layout có bằng chứng từ annotation; không coi là cấu hình IPF duy nhất có thể sinh các center đó |
| Time offset | `zeroDopMinusAcqTime=0.335899 s` | Dùng để ánh xạ annotation và corrections; không thêm vào PRI trong ACCC |
| DC method / input | `Data Analysis` / `Range Compressed` | AUX_PP1 `SM_SL1__1`; không chạy estimator trên raw chưa range-compress |
| ACCC weighting | Power; thử thêm phase để kiểm tra độ nhạy | DAD Eq. 5-18; không áp preset phase của S6 sang S4 |
| Unwrap weighting / FFT | Coherence / 4.096 | Lựa chọn demo; thử uniform và FFT 128/1.024/8.192 |
| Fit | Quadratic, uniform | Fit lại 20 Fine DC L1 khôi phục polynomial L1 trong 2,1 µHz ở các center, ủng hộ uniform LS |
| Outlier / iterations | 3,5σ / 5 vòng | Giả định nghiên cứu; thử σ=2/2,5/5 và 1/10 vòng; không suy được duy nhất từ L1 |
| RMS threshold | `20 Hz` | AUX_PP1; thử 1/100 Hz để xác minh nhánh chọn nguồn |
| Radar frequency / lambda | `5405000454.33435 Hz`; lambda=c/f | AUX_INS, không dùng hằng số băng C làm tròn |
| Orbit / attitude | ECEF position/velocity; quaternion Q0 scalar, reorder khi dùng scipy | L0; Hermite orbit, Slerp attitude, đổi GCRS→ITRS cho vector plane normal |
| Geometry interpolation | 9 control points, quadratic theo range | Thử 3/5/20: khác 9 điểm tối đa 0,000189 Hz tại hai tâm chuẩn |
| Geometry time coverage | Hai tâm chuẩn nằm trong orbit và attitude; mọi tâm của sweep đều được kiểm tra | Cuối scene vượt orbit 0,626854 s và attitude 0,876854 s; không mở rộng kết luận coverage sang toàn bộ focuser |
| Batch / dtype | 256 dòng, memmap complex64 | Không giữ toàn raw trong RAM; kiểm finite sau decode/corrections |

Theo [PDU §3.2.5.11–12 và Annex 5.1](https://sentinels.copernicus.eu/documents/247904/2142675/Sentinel-1-SAR-Space-Packet-Protocol-Data-Unit.pdf),
RGDEC 5 dùng L=3, M=8, FilterOutputOffset=92,
D=(0,1,1,1,2,2,3,3). Kiểm tra lại bằng helper `stage3_rx_samples` có sẵn.

### Tham số được phân loại khỏi đường DCE này

Hamming coefficient 0,75, range processing bandwidth 50,6 MHz, azimuth
processing bandwidth 1.429 Hz, azimuth focus FFT 4.096, overlap 50 và AAP
thuộc cấu hình focus/post-processing; chúng không mặc nhiên là window,
block size hoặc FFT của Fine DCE. Thermal noise correction trong L1 tắt;
không trừ noise power khỏi complex raw. RFI mitigation L1 là `Never`.
TOPSAR deramp không áp cho Stripmap S4. Các hằng absolute calibration và
scale complex toàn cảnh không đổi lag-1 phase, nên demo không phục hồi
absolute radiometry từ PG annotation.

### AUX và khả năng tái lập L1

Theo [Auxiliary Product Specification §3.2.1](https://sentiwiki.copernicus.eu/__attachments/1673968/S1-RS-MDA-52-7443%20-%20Sentinel-1%20IPF%20Auxiliary%20Product%20Specification%202024%20-3-16.pdf),
chọn V gần nhất không sau job start, cùng V thì chọn G mới nhất. AUX_INS
cục bộ `V20231201_G20251015` không bị loại vì G sau ngày thu cảnh.
Manifest L1 ghi `V20231201_G20250116`: đây là khác provenance. Chưa có
payload bản cũ để so từng hệ số, chưa kiểm toàn catalog AUX để chứng minh
lựa chọn tối ưu trên toàn kho. PP1/CAL cục bộ trùng tên bản manifest ghi.

File AUX_RESORB mà L1 dùng không có. Geometry DC từ L0 khác polynomial
geometry L1 khoảng 1,304/3,161 Hz RMSE ở hai block chuẩn, dù cả hai vẫn
cho ambiguity 0. Thay số control point chỉ ảnh hưởng dưới 0,0002 Hz tại
các tâm đã thử. Chưa đủ bằng chứng quy residual hình học cho một nguyên
nhân riêng: orbit source, attitude/frame, timing hoặc mô hình plane/terrain.

## 4. Thiết kế thử nghiệm

Mỗi case chỉ đổi nhóm tham số ghi trong tên. `basic` dùng bias subset L0
và chirp header. Các case bỏ một correction đều xuất phát từ `reference`.
`no_delay` còn bỏ delay guard nên support rộng hơn và bị chặn vì vượt
range bank EAP ở chunk 14. Không có DCE/RMSE cho case này; nó xác nhận
guard support, chưa định lượng riêng ảnh hưởng phase ramp trên cùng ROI.

Sweep tham số estimator dùng cùng input `reference`, không decode lại
giữa mỗi cấu hình. Mỗi preprocessing case decode/compress lại để tránh
biến đổi in-place làm nhiễm tín hiệu case sau. File memmap tạm được xóa
sau từng preprocessing case. Có assert chặn cửa sổ azimuth không hợp lệ
trước khi builder có thể clamp start âm thầm.

`audit.json` giữ cấu hình thực tế, starts/stops, số điểm đối chiếu, số điểm
fit còn lại, ambiguity, coherence, polynomial và thời gian chạy. `metrics.csv`
giữ bảng từng block. SHA-256 của raw, annotation, INS/PP1 và source code
được ghi để nhận biết thay đổi dữ liệu/implementation ở lần chạy sau.
Geometry được memoize theo tâm azimuth và range grid giữa các case;
thời gian chạy ghi trong JSON không phải benchmark công bằng giữa các cấu hình.

<!-- RESULTS_START -->
30 trường hợp trên dữ liệu thật: **29 case tạo DCE, 1 case bị guard chặn**.
Mỗi ô số ghi **block 1 / block 2**, đơn vị Hz; `điểm so` là số center trong giao support.

| Case | Fine RMSE/L1 | Polynomial RMSE/L1 | Fit RMS | Điểm fit | Nguồn chọn | Điểm so |
|---|---|---|---|---|---|---|
| `basic` | 1.362 / 1.451 | 1.331 / 1.410 | 1.425 / 1.214 | 20 / 20 | data / data | 20 / 20 |
| `reference` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `no_bias` | 1.229 / 1.444 | 1.194 / 1.400 | 1.373 / 1.158 | 20 / 20 | data / data | 20 / 20 |
| `no_rx` | 1.371 / 1.494 | 1.341 / 1.451 | 1.418 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `no_pg` | 1.377 / 1.492 | 1.348 / 1.448 | 1.419 / 1.207 | 20 / 20 | data / data | 20 / 20 |
| `no_eap` | 1.370 / 1.492 | 1.340 / 1.448 | 1.417 / 1.205 | 20 / 20 | data / data | 20 / 20 |
| `no_rsl` | 1.371 / 1.495 | 1.341 / 1.451 | 1.418 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `phase_weight` | 1.380 / 1.497 | 1.351 / 1.453 | 1.417 / 1.207 | 20 / 20 | data / data | 20 / 20 |
| `coherence_fit` | 1.372 / 1.495 | 1.347 / 1.450 | 1.418 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `uniform_unwrap` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `fft_128` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `fft_1024` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `fft_8192` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `outlier_2.0` | 1.372 / 1.495 | 1.525 / 1.912 | 1.226 / 0.799 | 19 / 17 | data / data | 20 / 20 |
| `outlier_2.5` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `outlier_5.0` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `iterations_1` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `iterations_10` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `width_500` | 1.509 / 1.660 | 1.364 / 1.583 | 1.590 / 1.377 | 20 / 20 | data / data | 18 / 18 |
| `width_1500` | 1.393 / 1.481 | 1.397 / 1.415 | 1.213 / 1.089 | 20 / 20 | data / data | 20 / 20 |
| `ranges_10` | 1.268 / 1.467 | 1.082 / 1.687 | 1.437 / 1.021 | 10 / 10 | data / data | 10 / 10 |
| `ranges_40` | 1.420 / 1.521 | 1.395 / 1.447 | 1.390 / 1.197 | 40 / 40 | data / data | 40 / 40 |
| `degree_1` | 1.372 / 1.495 | 1.463 / 1.453 | 1.550 / 1.209 | 20 / 20 | data / data | 20 / 20 |
| `degree_3` | 1.372 / 1.495 | 1.344 / 1.560 | 1.415 / 1.062 | 20 / 20 | data / data | 20 / 20 |
| `threshold_1.0` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | geometry / geometry | 20 / 20 |
| `threshold_100.0` | 1.372 / 1.495 | 1.342 / 1.452 | 1.417 / 1.206 | 20 / 20 | data / data | 20 / 20 |
| `azimuth_4000` | 1.890 / 1.278 | 0.650 / 0.683 | 1.934 / 1.710 | 20 / 20 | data / data | 20 / 20 |
| `azimuth_8000` | 3.592 / 2.045 | 3.393 / 1.559 | 1.281 / 1.523 | 20 / 20 | data / data | 20 / 20 |
| `shift_inward_554` | 3.157 / 1.731 | 2.961 / 1.265 | 1.731 / 1.824 | 20 / 20 | data / data | 20 / 20 |
| `no_delay` | — | — | — | — | Từ chối: vượt EAP range support | — |
<!-- RESULTS_END -->

Các case `degree_1`/`degree_3` chỉ là đối chứng mô hình, không thay thế
quadratic của DAD. Các case `azimuth_4000`, `azimuth_8000` và
`shift_inward_554` thay nội dung quan sát; không dùng RMSE của chúng để
chốt layout chính thức. Số điểm giao range có thể nhỏ hơn tổng số block
ở các case width/count khác chuẩn; JSON ghi rõ số điểm đã so.

## 5. Kiểm tra tín hiệu tổng hợp và lỗi đầu vào

12 test trong `test_sentinel1_dce_demo.py` đã qua, bao gồm:

- Tone 7 Hz và annotation giả −999 Hz: chứng minh DCE lấy tần số từ tín
  hiệu, không sao chép giá trị L1.
- Bias + RX gain + reciprocal PG đã biết: phục hồi tone 7 Hz.
- RX time tính bằng µs, bao gồm thời gian suppression.
- EAP giữ phase khi căn magnitude, RSL đúng amplitude exponent.
- Echo có positive hardware delay: matched filter advance đưa peak về
  đúng mẫu; delay guard giữ DCE trong range support hợp lệ.
- Layout range L1 sai hoặc timeline có gap: từ chối.
- PG invalid không được dùng; thiếu hai node hợp lệ thì từ chối.
- Tone −49/−7/7/49/107 Hz với PRF 100 Hz: kiểm dấu, alias và phase với
  amplitude 1 hoặc 1e−12.
- EAP ngoài range/time support và RX denominator bằng 0: từ chối.
- Đối chiếu chỉ dùng giao support; dưới ba điểm đối chiếu thì từ chối.

Full unittest: **48 test, 42 qua, 4 skip, 2 lỗi thiếu fixture S6**.
Hai lỗi là `test_s6_annotation_fit_reproduces_ipf_valid_points` và
`test_s6_range_blocks_match_annotation_layout`, do thiếu file
`references/s1a-s6-slc-vv-20251226t214357-20251226t214426-062491-07d496-002.xml`.
Không đổi thành skip hoặc tạo fixture giả để báo full suite xanh.

## 6. Cách chạy lại và đọc artifact

Chạy từ root repo, không dùng Python hệ thống thay interpreter dự án:

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/audit_doppler_centroid.py
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  -m unittest discover -s test -p 'test_sentinel1_dce_demo.py'
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  -m unittest discover -s test -p 'test_*.py'
```

CLI nhận `--product`, `--output` và `--input-case`; mặc định chạy mọi case tại
`output/sentinel-1/dce_audit/` (gitignored). Cần khoảng 2,3 GiB tạm cho
range-compressed memmap; bộ sweep chạy lâu hơn demo một cấu hình.
Artifact bổ sung của lần khảo sát này: `parameter_checks.json`,
`header_checks.json`, `geometry_checks.json`, bản PDU trong `sources/`,
và log test. Đây là kiểm tra bổ sung tại thời điểm khảo sát; CLI chính
tái tạo `audit.json`/`metrics.csv`. Bảng trong tài liệu giữ lại kết quả
đọc được ngay cả khi thư mục output bị xóa.

Lần chạy đầu hoàn tất 29 case DCE rồi dừng ở guard của `no_delay`.
CLI đã được sửa để ghi rejection dự kiến này và kết thúc bình thường;
chạy lại riêng bằng `--input-case no_delay --output
output/sentinel-1/dce_audit/no_delay_check`. Kết quả này được tổng hợp vào
artifact chính; bản source đã chạy 29 case được giữ ở `executed_audit.py`
với SHA-256 tương ứng. Không chạy lại các phép tính số không thay đổi.

## 7. Kết luận và giới hạn cần giữ cho bước tiếp theo

<!-- CONCLUSION_START -->
**Cấu hình giữ lại cho demo:** 6.000 azimuth lines, starts `[0,10500]`,
20 range block rộng 1.000 mẫu, quadratic uniform fit, power ACCC,
coherence unwrap với FFT 4.096, outlier 3,5σ tối đa 5 vòng và RMS limit
20 Hz. ROI là 18.170 mẫu ở chế độ cơ bản, 18.145 mẫu khi bù internal
delay tham chiếu. Trong số này FFT unwrap, outlier và iteration vẫn là
giả định nghiên cứu, không được ghi thành cấu hình IPF đã xác nhận.

- Input corrections tham chiếu không làm giảm residual với L1: Fine RMSE
  cơ bản 1,362/1,451 Hz, tham chiếu 1,372/1,495 Hz. Không bỏ các correction
  do AUX/L1 yêu cầu chỉ vì một ablation cho RMSE thấp hơn.
- FFT 128/1.024/4.096/8.192 và uniform/coherence unwrap không đổi Fine DC
  đáng kể trên cảnh này. Cảnh có DC gần zero, ambiguity 0; kết quả này
  không chứng minh độ bền unwrap ở các cảnh có wrap hoặc ambiguity khác.
- Outlier 2σ làm fit RMS giảm từ 1,417/1,206 xuống 1,226/0,799 Hz nhưng
  polynomial RMSE với L1 tăng từ 1,342/1,452 lên 1,525/1,912 Hz; số điểm
  giữ lại giảm từ 20/20 xuống 19/17. RMS thấp hơn không đồng nghĩa tái hiện
  L1 tốt hơn. Với 2,5/3,5/5σ và 1/5/10 vòng trong các sweep riêng đã thử,
  cấu hình tương ứng không thay đổi fit chuẩn.
- Width/count của range block ảnh hưởng Fine DC; phải giữ tọa độ và support
  trong báo cáo, không so hai vector theo index khi center khác nhau.
- Đổi cửa sổ azimuth cho Fine RMSE tới khoảng 3,592 Hz trong các case đã
  thử. Vì support thời gian khác L1, không dùng kết quả đó để thay layout
  có bằng chứng annotation.
- Ngưỡng RMS 1 Hz chuyển cả hai block sang geometry; 20/100 Hz giữ data.
  Fine DC không đổi khi chỉ đổi ngưỡng. CSV ghi polynomial RMSE của **data
  polynomial**, không đổi nghĩa cột đó thành RMSE của nguồn đã chọn.

Ưu tiên bước tiếp theo là tái hiện RRF Extended Tapered đúng đặc tả và
kiểm tra lại chirp/normalization/support trên cùng tọa độ; sau đó thay các
correction lấy từ L1 bằng kết quả calibration/raw analysis/EAP dựng từ
L0/AUX. Để giải thích residual geometry, cần đối chiếu orbit/attitude,
frame và timing với nguồn orbit của lần xử lý L1. Payload AUX_INS cũ giúp
kiểm tra provenance; chỉ xem khác generation là giả thuyết cho tới khi
so sánh được những trường thực sự đi vào phép tính.
<!-- CONCLUSION_END -->

Chưa tái hiện chính xác **Extended Tapered RRF spectrum** của IPF; hiện
dùng nominal time-domain matched filter. Chưa tái dựng raw analysis,
replica/PG và EAP hoàn toàn từ L0/AUX. Chưa so payload INS cũ, chưa có
AUX_RESORB của L1 và chưa có measurement TIFF để kiểm tra pixel/focus.
XML được parse và kiểm các trường cần dùng, chưa validate toàn bộ tài liệu
bằng XSD. Kiểm tra decoder ở đây gồm mode, kích thước, finite và tín hiệu
sau xử lý; chưa chứng minh giải mã bit-exact với IPF.
Không suy ra rằng chênh lệch AUX generation gây residual, cũng không
cam kết các tham số nghiên cứu chính là cấu hình IPF chỉ vì fit RMS thấp.
