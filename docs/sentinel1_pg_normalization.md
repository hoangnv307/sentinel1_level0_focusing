# Kiểm định reference và chuẩn hóa PG — S1A/S4/VV

Sau báo cáo này đã [nối internal delay L0 vào RRF](sentinel1_internal_delay.md).
Các lệnh tái chạy bên dưới ghim nguồn delay cũ để tái lập đúng baseline;
CLI hiện mặc định delay L0.
Đã [tách delay và PG availability](sentinel1_calibration_availability.md):
`validate_pg` nay trả QC/source kể cả khi không đủ node; gate >=2 thuộc
interpolation và caller yêu cầu PG drift.

Thực hiện ngày 21/10/2026, tiếp nối snapshot
[kiểm kê calibration](sentinel1_pg_calibration.md). Tài liệu này mô tả phiên bản
hiện hành; các con số của snapshot cũ là baseline trước sửa reference.

## 1. Kết luận và quyết định triển khai

Đã xác định nguyên nhân PG cũ lệch thang: nén replica PG bằng **nguyên replica
đo được đầu tiên** thay vì nominal PG chirp có biên độ **1/N**. Reference cũ
mang cả gain, phase và năng lượng của lần đo đầu vào bộ lọc; kết quả tương
quan có đơn vị DN² × samples. `PGref` và model AUX dành cho đỉnh nén bằng
reference chuẩn hóa, có đơn vị DN. Không có căn cứ nhân/chia thêm FFT length
để sửa khác biệt này.

Nhánh hiện hành tạo nominal PG chirp từ **header L0**, tính complex `PGcal`
bằng correlation, rồi `C(t)=PGref/PGcal(t)` **trước** QC với AUX model.
`C` là correction factor; không gọi nhầm nó là đáp ứng thiết bị `PGcal`.
Không fit scale/phase theo L1. DCE tiếp tục chỉ áp `C(t)/C(first)` qua
`--l0-pg-drift`; chưa bật PG mặc định hoặc áp absolute correction vào SLC.
`pgSource=Extracted` vẫn có nghĩa đo PG từ calibration L0. Dùng nominal
**reference chirp** không đồng nghĩa chuyển sang `pgSource=Model`.

## 2. Đối chiếu tài liệu và mâu thuẫn cần giữ trong context

| Nguồn | Điều đã kiểm tra | Cách áp dụng |
|---|---|---|
| DAD Issue 2/5, §4.2.1.1.2, Eq. 4-25 | Nominal PG có biên độ 1/nsamples, chirp từ downlink | Hàm `nominal_pg_replica` |
| DAD §4.2.1.3, Eq. 4-32 | Tái dựng replica từ 5 đường calibration | Giữ PCC/Average và tỷ số FFT |
| DAD §4.2.1.4 | Định vị bằng correlation, FFT zoom/parabola; kiểm định IRW/PSLR/ISLR và cắt TXPL được mô tả riêng cho **image** replica | PG kiểm tra peak hữu hạn, khác zero, nằm trong cửa sổ chứa đủ chirp; không tự áp ngưỡng image cho PG |
| DAD §4.2.1.6.1, Eq. 4-36 | Vẫn mô tả reference là replica hợp lệ đầu tiên đã trích xuất | Không coi nguyên cửa sổ replica đầu là reference đã chuẩn hóa |
| DAD §4.2.2, bước 1 | Nominal PG được dùng cả để tính PG product | Phù hợp nhánh nominal, nhưng chưa đủ tự giải quyết mâu thuẫn Eq. 4-36 |
| DAD §9.20, Eq. 9-57–58 | Forward FFT không scale; IFFT có 1/N | SciPy đã thực hiện đúng, không chia thêm N |
| Calibration Plan 8.1, §4.6.1, §9.2.4–6 | Cách dùng measured replica là thiết kế trước; cách được mô tả dùng chirp lý tưởng chuẩn hóa | Chọn nominal 1/N có căn cứ tài liệu, không chọn vì RMSE thấp hơn |
| Auxiliary Product Specification 3/16, Table 6-15–16 | `PGref/PGcal ≈ PGmodel`, cùng thang; model lý tưởng gần 1+0j | QC dùng `C`, không dùng `1/PGcal` hoặc raw autocorrelation |

Tài liệu nguồn ngoài repo:

- [Calibration and Characterisation Plan, Issue 8.1, 25/02/2016](https://sentiwiki.copernicus.eu/__attachments/a_50b520d3188da65898086d651f828368627281b555bdfe2be7575e78fbea067d/S1-PL-ASD-PL-0001-Sentinel-1A-1B-SAR-Instrument-Calibration-and-Characterisation-Plan-8.1.pdf).
- [Auxiliary Product Specification 3/16](https://sentiwiki.copernicus.eu/__attachments/a_ac101207bbd6069e56e3922c261317dac89cb2f1bc1b44e0696bad24def0df0f/S1-RS-MDA-52-7443%20-%20Sentinel-1%20IPF%20Auxiliary%20Product%20Specification%202024%20-3-16.pdf).
  File URL có chữ 2024 nhưng header bản tải về ghi **08/09/2025**.
- [Product Specification 3/16.1](https://sentiwiki.copernicus.eu/__attachments/a_19b24d9a39965738b9030586e95d8cae0fe3a26d0c571e5761848cef8ac2f9b4/S1-RS-MDA-52-7441%20-%20Sentinel-1%20Product%20Specification%202025%20-%203.16.1.pdf),
  `pgProductAmplitude/Phase`: annotation là inverse response, không chứa
  `PGref`. L1 chỉ được đọc sau khi hoàn thành nhánh tính L0/AUX.

DAD [R-13] là **S1-DD-DLR-SY-002, Issue 2.0, 26/04/2010**. Chưa tìm được
bản công khai để kiểm tra; không khẳng định code bit-exact với IPF từ các
tài liệu hiện có. Đặc biệt, sửa nominal PG không đồng nghĩa đã hoàn thiện
extraction/validation của **image replica** dùng cho RRF.

## 3. Luồng và bảng units

```text
L0 BYPASS → chọn pulse theo AUX_INS → PCC2 / Average
         → IFFT(TX × RX × TA / (EPDN × APDN)) → replica r_m [DN]

Header PG L0 → q[n] = exp(j*phase[n])/N
r_m + q → correlation → FFT zoom/parabola → p_m = PGcal_m [DN]
p_m + AUX_INS PGref → C_m = PGref/p_m [dimensionless]
C_m + model/threshold AUX → QC / chọn extracted hoặc model
→ C_m/C_first → nội suy complex theo acquisition time → provider DCE

L1 annotation → đối chiếu 1/p_m và drift, không tham gia tính/chọn C_m
```

| Bước | Công thức / chuẩn hóa | Units và điều không được làm |
|---|---|---|
| Decode calibration | BYPASS complex I/Q từ decoder | DN; không áp bias/RX correction dành cho echo |
| PCC2 | `(x-y)/2` theo order AUX | DN; không quên hệ số 1/2 |
| Average APDN | Trung bình đúng các pulse AUX chọn | DN; không thêm scale theo số pulse sau khi đã lấy mean |
| Replica | `ifft(fft(tx)*fft(rx)*fft(ta)/(fft(epdn)*fft(apdn)))` | DN; tỷ số 3 đường / 2 đường, không normalize năng lượng replica |
| Nominal PG | `q[n]=exp(2πj*(f0*t+0.5*K*t²))/N` | Reference 1/N; không dùng unit-energy 1/sqrt(N) |
| Correlation | `ifft(fft(r)*conj(fft(q)))` | DN; sum với reference 1/N loại compression gain |
| FFT zoom | `scipy.signal.resample` giữ biên độ, refine parabola | Không chia thêm oversampling factor |
| Peak | Complex sample tại vị trí cực đại magnitude | Giữ phase; không thay bằng `max(abs(...))` |
| Model normalization | `C=PGref/peak` | Dimensionless, đúng thang AUX model |
| DCE drift | `C(t)/C(first)` | Dimensionless; loại static complex scale, vẫn giữ chiều correction |

`N=ceil(TXPL*Fs)`. Với header thực: `TXPL*Fs≈5798,999999999999`, nên
**N=5799**. Dùng thời gian `t=n/Fs`, không dùng linspace bao cả endpoint
TXPL. Chirp parameters phải giống nhau giữa các sequence dùng chung reference;
adapter báo lỗi nếu khác grid/chirp, không ghép âm thầm.

Guard spectral floor vẫn là tham số caller; mặc định nghiên cứu 0 chỉ chặn
singularity chính xác. Không regularize mẫu số để ép ra kết quả.

## 4. Gain ratio của Stripmap và đơn vị QC

Plan §9.2.6 xác định Stripmap là **một swath**: RX settings của PG và image
tương ứng cùng hoạt động thu. Ratio bổ sung cho IW2/IW3, EW2…EW5 hoặc WV2
không áp cho S4. Vì vậy ratio hiện tại **1**, có nguồn tài liệu; không cần
file S1/S2/S3 và không dùng `RxCal cuối/RxCal đầu` làm inter-swath ratio.
Image RxCal power vẫn được kiểm kê để chẩn đoán.

AUX_PP1 payload hiện tại schema **3.15**, trong khi XSD đi kèm ghi 3.14
và còn mô tả `maxPgAmpError` theo dB, `maxPgPhaseError` theo degrees.
Aux Spec 3/16 Table 4-5 và lịch sử sửa đơn vị xác nhận amplitude **tuyến tính**,
phase **radian**. Reader hiện hỗ trợ rõ schema 3.15/3.16; version khác bị từ
chối để tránh đoán units. Không tự đổi `6,28` sang độ hoặc dùng log amplitude.

Giữ mọi threshold AUX cũ: amp std fraction=1e6, phase std fraction=1e6,
max amp error=1e6, max phase error=6,28 rad, max invalid fraction=1.
Chúng rất rộng. QC pass không chứng minh model khớp tuyệt đối: phase của C
so model vẫn lệch **1,68258 / 1,72596 rad**. Không xoay phase để ép model.

## 5. API và thay đổi tương thích

Core `pre_processing/internal_calibration.py`:

```python
q = nominal_pg_replica(
    sample_rate_hz=..., pulse_length_s=...,
    start_frequency_hz=..., ramp_rate_hz_per_s=...,
)
corrections, peaks, lags = pg_products(
    replicas, reference=q, pg_reference=...,
    oversampling_factor=...,
)
```

`reference` là keyword bắt buộc mới: caller phải quyết định rõ reference,
không còn default ngầm `replicas[0]`. Hàm vẫn hỗ trợ reference complex do
caller cung cấp để kiểm thử/đối chứng; adapter hiện hành luôn dùng nominal.
Các caller trong adapter và unittest đã được cập nhật. Shape, finite values,
reference dài hơn data, reference zero và oversampling factor không hợp lệ
đều bị chặn. QC thresholds, PCC orders, FFT zoom/floor tiếp tục đi qua API.

Adapter `analyze_calibration` giữ nguyên signature và return
`times, relative_corrections, report`. Report thay các field raw-first-replica
cũ bằng `pgcal_peak_dn`, `inverse_pgcal`, `normalized_pg_correction`;
giữ kết quả cách cũ dưới `normalization_audit` chỉ để so sánh.

- `pg_reference_function`: nguồn, header chirp, N, amplitude, energy và FFT convention.
- `quality.absolute_scale_verified=true`: **chỉ** xác nhận normalization và
  thang `PGref/PGcal` để so model; đọc cùng `absolute_scale_verification_scope`.
- `absolute_radiometric_scale_recovered=false`: chưa xác nhận radiometry SLC/IPF.
- `quality.model_*_residual`: báo riêng lệch amplitude/phase, không che bằng QC pass.
- `normalization_audit.stripmap_receive_gain_ratio=1`: có nguồn single-swath rõ ràng.
- `l1_diagnostic_only`: chỉ CLI đối chứng thêm vào; adapter không đọc L1.

Không còn phụ thuộc “reference đo được đầu tiên” nên **không cắt/normalize
một measured replica để thay nominal**. Nhánh image extraction, metrics
IRW/PSLR/ISLR và coefficient fitting vẫn là công việc riêng trước khi dùng
extracted image RRF; AUX của cảnh này chọn nominal RRF.

## 6. Số liệu S4/VV thực tế

Fs=112.604.166,72 Hz; nominal amplitude=energy=`1/5799 ≈ 0,000172443525`.
PCC selection, timestamp và 28 chunk giữ như snapshot cũ.

| Đại lượng | Node đầu | Node cuối |
|---|---:|---:|
| Complex PGcal [DN] | −91,4424803 − j814,5874970 | −125,2619729 − j800,8255151 |
| C=PGref/PGcal | −0,096346063 + j0,858269571 | −0,134972689 + j0,862908119 |
| abs(C) | 0,863660361 | 0,873400280 |
| abs(C) − abs(model) | −0,007329639 | +0,002410280 |
| Legacy raw-first peak / nominal peak, magnitude | 4.781.740,765 | 4.778.672,982 |
| abs(1/PGcal) / amplitude L1 − 1 | −0,0154338% | −0,0142082% |
| arg((1/PGcal) / complex L1) [rad] | −0,000147490 | −0,000150832 |

Drift node cuối: **1,010326522452 + j0,043846021970**, amplitude
**1,011277486950**, phase **0,043370658779 rad**. So drift L1: amplitude
lệch **0,00122574%**, phase lệch **−3,34122e−6 rad**. So sánh L1 không dùng
fit gain/phase, không dùng L1 để chọn N, PCC order hoặc FFT zoom.

Hai lượt DCE toàn bộ 16.500 echo, cùng I/Q/RX, layout và estimator như baseline:

| Input / block | Fine DC/L1 RMSE [Hz] | Polynomial/L1 RMSE [Hz] | Fit RMS [Hz] |
|---|---:|---:|---:|
| Default + PG nominal / 1 | 1,367668564 | 1,336653847 | 1,426073346 |
| Default + PG nominal / 2 | 1,447891816 | 1,406500543 | 1,215218835 |
| Reference, thay PG L1 bằng PG nominal L0 / 1 | 1,381625485 | 1,352814681 | 1,419865713 |
| Reference, thay PG L1 bằng PG nominal L0 / 2 | 1,487421013 | 1,443421129 | 1,208617096 |

So default + PG measured cũ: Fine DC thay đổi max **0,000777314 /
0,000534828 Hz**. Với mọi input correction khác giữ nguyên ở reference
mode, PG nominal L0 so PG L1 làm Fine DC lệch max **1,53037e−5 /
1,02714e−5 Hz**, RMS **9,15782e−6 / 5,16065e−6 Hz**. ROI tương ứng
18.170 và 18.145, 20 range block ×1.000, azimuth [0,6000), [10500,16500).
Cả hai lượt đều chọn data, ambiguity 0, RMS dưới ngưỡng AUX 20 Hz.
Đây là regression correction PG, chưa giải thích residual DCE cỡ Hz.

## 7. Kiểm chứng và tái chạy

`test/test_internal_calibration.py` có **24 test**: các test cũ và thêm
known-gain xuyên PCC→reconstruction→PG; nominal chirp/header lỗi; pulse length
31/64/129, hai chiều dài zero padding, FFT zoom 8/16/32 không đổi gain;
reference lỗi; từ chối schema cũ chưa có hợp đồng units. Test drift 7 Hz sinh
instrument gain từ mô hình thuận độc lập, không sinh raw bằng correction
đang được kiểm thử. Oracle L0 thật tự chọn pulse/NumPy FFT và tìm đỉnh bằng
Fourier evaluation dày tại fractional lag cho **cả hai** sequence.

Full suite: **125 test, 119 đạt, 4 skip, 2 lỗi thiếu XML S6 cũ**. Không coi
đây là toàn suite xanh; không đổi lỗi fixture thành skip để che lỗi.

```bash
MPLBACKEND=Agg /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/analyze_calibration.py \
  --output output/sentinel-1/pg_normalization

MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py --l0-pg-drift \
  --internal-delay-source none \
  --output output/sentinel-1/pg_normalization/dce_l0_pg

MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 /home/xiaoxin/python_envs/sentinel1/bin/python \
  workflows/sentinel-1/estimate_doppler_centroid.py \
  --l1-input-corrections --l0-raw-analysis --l0-pg-drift \
  --internal-delay-source l1 \
  --output output/sentinel-1/pg_normalization/dce_reference_l0_pg
```

JSON/PNG nằm dưới output trên và không đưa vào Git. Lượt DCE giữ lại report
đúng thời điểm chạy; report kiểm kê mới bổ sung cả tên/đơn vị các chirp header.
Không sửa các mảng số của JSON sau khi chạy. Baseline measured-reference cũ
vẫn ở `output/sentinel-1/pg_calibration/` để kiểm tra hồi quy.
`output/sentinel-1/pg_normalization/verification.json` ghi kết quả kiểm tra
cùng config, segment correction metadata, ROI và các gate DCE của hai lượt.

Giới hạn còn lại: chưa có bit-exact IPF, model LUT không hằng cần epoch L0;
chưa repair/discard sequence lỗi để tiếp tục tự động; peak validity hiện
là finite/nonzero và support hợp lệ, chưa có tiêu chí SNR/leakage tách riêng;
chưa triển khai image-replica QC/extraction cho extracted RRF; chưa đưa PG
vào notebook S6 hoặc chứng minh radiometry toàn chuỗi. Mọi thay đổi default
hoặc absolute SLC correction phải được kiểm chứng riêng ở phạm vi đó.
