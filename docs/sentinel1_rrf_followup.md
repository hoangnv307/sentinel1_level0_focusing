# RRF: kiểm chứng và đề xuất phổ mở rộng — 21/10/2026

Đây là bước ưu tiên tiếp theo của [audit DCE](sentinel1_dce_audit.md):
kiểm tra chirp, chuẩn hóa và support của Range Reference Function, rồi
thử hai cách mở rộng phổ do dự án đề xuất. Không dùng RMSE với L1 để
nhận một giả thuyết là thuật toán IPF.

**Đính chính receiver:** các lượt so sánh input tham chiếu bên dưới chạy
trước khi sửa mẫu số RX từ GV sang **sqrt(GV)** theo Eq. 9-16. Giữ số
liệu lịch sử, xem [báo cáo RX](sentinel1_receiver_gain_compensation.md) cho
luồng hiện tại. Các test RRF về energy/support vẫn độc lập với lỗi scale RX.

## 1. Những gì đã xác định từ tài liệu

DAD §6.1.1, trang in 6-3–6-4 (PDF trang 70–71), mô tả replica whitening,
FFT/conjugate, phase ramp, chuẩn hóa energy rồi nhân complex gain.
Với SLC, `Nifft=Nfft`, `E=mean(abs(H)^2)` và chia H cho **sqrt(E)**.
Đã kiểm tra trực tiếp ảnh PDF vì text extraction làm mất dấu căn ở công
thức. DAD đặt Hamming range weighting ở post-processing, không tự thêm
window Hamming tại range compression.

[Auxiliary Product Specification](https://sentiwiki.copernicus.eu/__attachments/1673968/S1-RS-MDA-52-7443%20-%20Sentinel-1%20IPF%20Auxiliary%20Product%20Specification%202024%20-3-16.pdf),
[Product Specification](https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Specification-18052021.pdf)
và XSD AUX cục bộ định nghĩa ba tên `Unextended`, `Extended Flat`,
`Extended Tapered`. Trong các mục đã đối chiếu, chưa tìm thấy hàm taper,
điểm nối, quy tắc extension phase hoặc điều kiện biên để triển khai
bit-exact chế độ `Extended Tapered` của IPF 003.92.

Tài liệu [EOPF S1 L1 Core](https://s1.pages.eopf.copernicus.eu/s1-l12-rp/0.0.0/svvp/process.html)
có chỉ repository `S1/s1-l1-core`. Lần kiểm tra này không đọc được mã nguồn
qua web/raw URL; GitLab API repository tree trả HTTP 404. Không diễn giải
404 thành bằng chứng rằng mã nguồn không tồn tại.

**Điều chỉnh cách hiểu từ audit trước:** FFT của chirp hữu hạn tự có phổ
kéo dài với roll-off/ripple ở biên. Vì vậy, chưa có cơ sở gọi bộ lọc cũ
là “Unextended” chỉ vì code không chứa một window riêng. Cũng chưa đủ
bằng chứng xác nhận nó chính là `Extended Tapered` của IPF.

## 2. Thay đổi có cơ sở, đưa vào đường xử lý chung

- Thêm `range_reference_function.from_coefficients`, dùng chung cho RRF
  từ header và từ reference-input. Có kiểm tra hệ số/timing/gain hợp lệ.
- Dùng thời gian centered `t=(n-(M-1)/2)/Fr`,
  `M=ceil(TXPL×Fr)`, phase polynomial tính bằng **cycles**.
- Whitening đảo amplitude trước tạo matched filter; chuẩn hóa energy
  trong miền tần số sau phase ramp, trước complex gain.
- Giữ time reversal để peak zero-delay nằm tại `M-1`. Crop hiện có loại
  implementation delay này và giữ nguyên gốc tau vật lý.
- Internal delay dương được loại bằng ramp `exp(+2jπf×delay)`; đây là
  advance, không đổi dấu thành delay thêm.
- Khi L1 ghi chirp source Nominal, lấy đủ hệ số nominal S4/VV từ AUX_INS.
  Hệ số L1 đã làm tròn được giữ để chẩn đoán, không làm input RRF mặc định.
  Bias/PG/EAP/delay tham chiếu vẫn dùng L1 trong chế độ nghiên cứu đã cho phép.

Các caller cũ của `create_freq_domain` giữ nguyên API; chúng chuyển tham số
header sang coefficient rồi gọi cùng builder. Không thêm dependency.
Hai candidate dưới đây chỉ có trong CLI thực nghiệm, không tự đưa vào
notebook focus hay demo mặc định.

## 3. Hai phương án nghiên cứu được đề xuất

Giới hạn cho nominal LFM amplitude phẳng, phase bậc hai, bandwidth nằm
trong Nyquist. Nếu chirp có cubic phase hoặc amplitude không phẳng thì
candidate từ chối, không bỏ qua các hệ số đó.

Đặt `K=2φ2`, `fc=φ1`, `B=abs(K)×TXPL`, `tc=(M-1)/(2Fr)`.
Từ xấp xỉ stationary phase của Fourier transform LFM, pha matched filter
được đề xuất là:

```text
θ(f) = π(f-fc)²/K - 2π f tc - 2πφ0 - sign(K)π/4 + 2π f delay
H(f) = A(f) exp(jθ(f)) / sqrt(mean(A(f)²)) × complex_gain
```

Hệ số `-2π f tc` giữ cùng vị trí peak/crop với builder hữu hạn. Không
thay độ dài TX pulse hoặc nhãn tau để làm peak khớp.

| Candidate | Envelope A(f) | Ý nghĩa và giới hạn |
|---|---|---|
| `flat` | 1 trên các bin FFT từ −Fr/2 đến Fr/2 | Xấp xỉ pha LFM và mở rộng phổ phẳng toàn Nyquist; có thể gián đoạn pha tại biên FFT và sinh kernel dài. |
| `cosine` | 1 trong [fc−B/2, fc+B/2]; ngoài band nối về 0 tại hai mép Nyquist bằng `0.5×(1+cos(πx))`, x chạy 0→1 riêng mỗi bên | Giảm biên và đạo hàm về 0 ở Nyquist; không dùng hệ số Hamming 0,75 hay fit taper theo L1. Đây là công thức do dự án đề xuất. |

Đây **không phải định nghĩa IPF đã xác nhận** cho hai tên enum tương ứng.
Envelope và pha đều được ghi cụ thể để có thể thay khi có nguồn chuẩn.

## 4. Kiểm tra support và điều kiện so sánh

FFT range cơ bản 32.768; M=2.900. Đối chiếu cùng input nhiễu complex với
FFT 65.536, rồi crop cùng tọa độ range. Do fractional advance và các
candidate có kernel không hữu hạn chặt, không coi `M` là support chính
xác của mọi bộ lọc chỉ vì TX pulse dài M mẫu.

Guard nghiên cứu bổ sung **512 mẫu mỗi mép ROI**, dùng như nhau cho cả
ba phương án khi đối chiếu DCE. Phần giao ROI còn 17.121 mẫu, origin
polynomial vẫn giữ t0 gốc; chỉ dời các range block vào phía trong.
Guard này lớn hơn phần tăng nửa độ dài chirp theo mô hình bandwidth
toàn Fr (khoảng 163 mẫu), nhưng không phải cấu hình IPF và không bảo đảm
đuôi kernel bằng 0. Thử tăng FFT là phép kiểm hội tụ trên input tổng hợp,
không phải kiểm chứng thiếu support ở mọi cảnh.

Cả `annotation_fft` và `aux_fft` chạy thêm với ROI gốc để kiểm tra riêng
việc đổi hệ số làm tròn. Không so candidate dùng guard 512 với baseline
guard 0 rồi diễn giải khác biệt đó là hiệu quả của taper.

## 5. Kết quả thực nghiệm

<!-- RRF_RESULTS_START -->
Energy trước gain của ba filter sai khác 1 dưới 3×10⁻¹⁰ ở miền tần số; IFFT complex64 cho sai số energy dưới 6×10⁻⁸.

| Phổ | Energy ngoài nominal band | Relative RMS khi tăng FFT, vùng bên trong |
|---|---|---|
| `aux_fft` | 0.4282% | 1.573e-05 |
| `flat` | 10.1013% | 5.640e-04 |
| `cosine` | 4.0435% | 1.481e-08 |

Nominal band được tính bằng K×TXPL (~50,6150 MHz), không dùng bandwidth post-processing làm ngưỡng cắt.

| RRF | Guard bổ sung mỗi mép | Fine RMSE/L1 B1 / B2 [Hz] | Polynomial RMSE/L1 B1 / B2 [Hz] | Fit RMS B1 / B2 [Hz] |
|---|---|---|---|---|
| `annotation_fft` | 0 | 1.371543 / 1.495289 | 1.341903 / 1.451801 | 1.417340 / 1.205822 |
| `annotation_fft` | 512 | 1.416999 / 1.498902 | 1.442106 / 1.383553 | 1.346576 / 1.195966 |
| `aux_fft` | 0 | 1.371543 / 1.495289 | 1.341902 / 1.451801 | 1.417340 / 1.205822 |
| `aux_fft` | 512 | 1.416998 / 1.498902 | 1.442105 / 1.383553 | 1.346576 / 1.195966 |
| `flat` | 512 | 1.418323 / 1.506308 | 1.444491 / 1.392306 | 1.342746 / 1.193414 |
| `cosine` | 512 | 1.419985 / 1.506928 | 1.446668 / 1.392747 | 1.345537 / 1.193551 |

Cả sáu case giữ 20/20 điểm fit, ambiguity 0 và nguồn data ở ngưỡng RMS 20 Hz.
`annotation_fft` với guard 0 tái tạo các metric audit trước trong độ chính xác đã ghi.
Đổi coefficient annotation → AUX trên cùng ROI làm Fine DC thay đổi tối đa khoảng **2,2 µHz**, không giải thích residual cỡ Hz.
Chuyển header sang phase centered cho φ1=−800,9332753382623 và φ2=491417744547,4791, trùng nominal AUX trong số học hiện tại.
<!-- RRF_RESULTS_END -->

Biểu đồ phổ và impulse response: `output/sentinel-1/rrf_check/spectra.png`.
`rrf_check.json` giữ energy, tỷ lệ energy ngoài nominal band, phép thử
FFT doubling, tọa độ Fine DC, cấu hình từng case và SHA-256 input/source.

## 6. Kiểm tra tự động và tái chạy

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/check_range_reference.py
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  -m unittest discover -s test -p 'test_range_reference.py'
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  -m unittest discover -s test -p 'test_sentinel1_dce_demo.py'
```

5 test RRF và 15 test DCE đều qua. Các test mới kiểm Parseval, tính độc
lập energy với độ dài FFT, inverse-amplitude whitening, complex gain
sau normalization, dấu delay, gốc range với chirp chẵn/lẻ, candidate
về 0 ở mép Nyquist, từ chối chirp ngoài phạm vi và không dùng coefficient
L1 khi đã có nominal AUX.

Full suite: 56 test, 50 qua, 4 skip, 2 lỗi thiếu annotation S6 đã ghi
trong audit trước. Chưa có measurement TIFF để kiểm impulse/ảnh L1 thật.

## 7. Quyết định cấu hình

<!-- RRF_DECISION_START -->
**Giữ FFT của chirp hữu hạn với hệ số AUX đầy đủ làm mặc định.** Hai
candidate không làm giảm RMSE với L1 trên cùng ROI có guard 512: native
1,416998/1,498902 Hz, flat 1,418323/1,506308 Hz, cosine
1,419985/1,506928 Hz. Không đưa chúng vào pipeline focus chính.

Nếu cần một phương án extension để nghiên cứu tiếp, ưu tiên **cosine**
vì hội tụ theo FFT tốt hơn trong phép thử nhiễu tổng hợp và tránh mép
Nyquist không trơn. Đây là tiêu chí tính ổn định của mô hình, không phải
bằng chứng nó khớp IPF. Phần energy ngoài band của cosine ~4,04%, khác
FFT chirp hữu hạn ~0,43%; hai filter thay đổi power weighting của dữ
liệu, không chỉ đổi một scale hằng.

Không chọn guard 0 thay cho 512 để làm candidate có vẻ khớp hơn. ROI
khác nhau làm nội dung range block đổi; các control riêng đã thể hiện
ảnh hưởng này. Chưa cần thay FFT unwrap/outlier của DCE khi chỉ đổi RRF.

Bước RRF hiện đã có builder chung, nguồn nominal AUX độc lập L1,
kiểm chứng normalization/delay/support và các candidate chạy thật.
Phần xác nhận **công thức Extended Tapered chính xác của IPF** vẫn mở:
cần định nghĩa edge/extension hoặc implementation/reference vectors
đủ để phân biệt taper tự nhiên của chirp hữu hạn với taper thiết kế.
<!-- RRF_DECISION_END -->

Những bước thay raw analysis, PG, EAP tham chiếu bằng phép tính L0/AUX và
đối chiếu orbit/attitude vẫn thuộc các bước sau. Lần này chỉ xử lý RRF
và thử nghiệm phổ mở rộng theo ưu tiên đã nêu; chưa xác nhận IPF bit-exact.
