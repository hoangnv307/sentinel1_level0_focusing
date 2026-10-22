# Tìm và tải annotation Sentinel-1 qua Copernicus Data Space

CLI `workflows/sentinel-1/copernicus_metadata.py` phục vụ **nghiên cứu DCE**:
tìm sản phẩm Stripmap SLC S1–S6, tải riêng annotation chính cùng manifest,
và kiểm chứng lịch azimuth block bằng script đã có. Các XML tải về không
được nối vào pipeline xử lý L0/AUX.

Không cần tài khoản để tìm kiếm hoặc chạy `download --dry-run`. Tải nội
dung file cần token CDSE. Mọi ví dụ dưới đây chạy từ thư mục gốc repo;
dùng interpreter `/home/xiaoxin/python_envs/sentinel1/bin/python`.

## 1. Tìm các sản phẩm cùng datatake

```bash
python workflows/sentinel-1/copernicus_metadata.py search \
  --seed S1C_S6_SLC__1SDH_20250330T180412_20250330T180440_001674_002C4C_A098.SAFE \
  --output output/sentinel-1/copernicus/catalogue.json
```

Seed là **tên sản phẩm SAFE**, không phải tên annotation XML. Search giới hạn
cùng mission, swath, absolute orbit và datatake, trong ±2 giờ quanh start của
seed. Có thể chỉ định lại `--start`/`--stop`; không mặc định rằng kết quả
đã bao trùm toàn datatake. `verify` báo các slice còn thiếu so với context.

Để tìm nhiều datatake theo khoảng ngày:

```bash
python workflows/sentinel-1/copernicus_metadata.py search \
  --start 2025-10-17T00:00:00Z --stop 2025-10-18T00:00:00Z \
  --satellite S1A --swath S2 --polarisation HH --limit 30 \
  --output output/sentinel-1/copernicus/catalogue.json
```

Thời gian không có timezone được hiểu là UTC. Start inclusive, stop exclusive.
Giới hạn mặc định 30 sản phẩm, tối đa 1.000; query được phân trang. JSON ghi
filter, thời gian truy vấn, UUID, tên, trạng thái Online, S3Path, dung lượng
toàn sản phẩm và các nhóm datatake. `truncated=true` nghĩa là còn kết quả
ngoài limit; tool mặc định chặn download danh sách này. Thu hẹp query/tăng
limit, hoặc dùng `--allow-truncated` khi chủ động chấp nhận mẫu thiếu.
Polarisation HH chấp nhận cả SH và DH; HV chỉ DH, VV chấp nhận SV/DV, VH chỉ DV.

## 2. Xem kế hoạch tải mà chưa đăng nhập

```bash
python workflows/sentinel-1/copernicus_metadata.py download \
  --catalogue output/sentinel-1/copernicus/catalogue.json \
  --polarisation HH --dry-run
```

Tool duyệt Nodes API, chỉ chọn `manifest.safe` và
`annotation/s1*-s*-slc-<pol>-*.xml`. Không tải measurement, noise/calibration
LUT hoặc ZIP. Không truyền `--polarisation` thì lấy mọi kênh annotation chính
có trong sản phẩm. Tên/số UUID, swath, datatake và đường dẫn được kiểm tra.

Mặc định lưu vào `~/sar_data/focusing_data/copernicus_metadata/`:

```text
copernicus_metadata/
├── download_plan.json              # danh sách URL/file/byte khi dry-run
├── download_index.json             # trạng thái từng sản phẩm khi tải thật
└── <product>.SAFE/                  # bản trích metadata, không phải SAFE đầy đủ
    ├── manifest.safe
    ├── annotation/s1*-s*-slc-*.xml
    └── metadata.json               # UUID, URL nguồn, SHA-256, số byte, thời gian tải
```

`--directory` thay đổi thư mục này. File plan/catalogue chỉ chứa metadata công
khai, không có username, password, access token hoặc refresh token.

## 3. Xác thực và tải

Cách thuận tiện khi chạy tại terminal:

```bash
python workflows/sentinel-1/copernicus_metadata.py download \
  --catalogue output/sentinel-1/copernicus/catalogue.json \
  --polarisation HH --login
```

CLI hỏi username/email, password **ẩn ký tự**, và OTP nếu bật 2FA (Enter nếu
không). Không cần gửi mật khẩu vào hội thoại hoặc đặt trên command line.

Để tự động hóa, nạp `CDSE_USERNAME`, `CDSE_PASSWORD` và nếu cần `CDSE_TOTP`
vào môi trường của process từ nơi lưu credential local ngoài Git. Có thể
thay bằng `CDSE_ACCESS_TOKEN` có sẵn; token này được ưu tiên. Với token bên
ngoài đã hết hạn, cần nạp lại token; tool không có refresh token của phiên đó.
Tool không tự đọc `.env`, không lưu credential hoặc token xuống đĩa.

Khi đăng nhập bằng username/password, dùng `client_id=cdse-public`; token
được giữ trong RAM và refresh gần lúc hết hạn hoặc sau một HTTP 401. Chỉ
thử refresh một lần cho request 401. Không retry POST đăng nhập tự động.

GET có timeout 30 giây và retry tối đa 3 lần cho lỗi kết nối/429/5xx được
cấu hình. Chỉ gửi Bearer tới các host download CDSE được cho phép; từ chối
redirect sang host khác. Không in response body của endpoint xác thực.

Mỗi file giới hạn 16 MiB, kiểm số byte từ Nodes và XML trước khi đổi tên
file tạm thành file hoàn tất. File cache chỉ được dùng lại nếu SHA-256,
byte count, URL và product identity khớp provenance; cache sai/không có
provenance báo lỗi để người dùng kiểm tra, không ghi đè âm thầm. Các file
hoàn tất trước lỗi được giữ lại và có thể dùng khi chạy lại. Lỗi từng sản
phẩm được ghi vào index; sản phẩm tiếp theo vẫn được kiểm.

Sản phẩm `Online=false` được báo rõ; bản này không tự yêu cầu khôi phục
archive, không tự chuyển sang S3 hoặc tải cả SAFE để thay thế.

## 4. Kiểm chứng DCE sau tải

```bash
python workflows/sentinel-1/copernicus_metadata.py verify \
  ~/sar_data/focusing_data/copernicus_metadata \
  --output output/sentinel-1/copernicus/verification.json
```

Cũng nhận nhiều file/thư mục XML cục bộ. Duyệt thư mục đệ quy; tham số
`--block-size-lines` (6.000), `--nominal-spacing-s` (10), `--tolerance-s`
(5 µs) được truyền vào checker. Không fit spacing hoặc offset để làm khớp.

Mỗi annotation có kết quả **cả `last_line` và `after_last_line`**, bao gồm
control model cũ, residual và lỗi nếu cấu trúc không hỗ trợ. Mẫu lỗi được
giữ trong báo cáo, không bỏ qua để cải thiện tỷ lệ khớp. Manifest nếu có
được đọc để ghi software/IPF và các resource AUX tham chiếu. Báo cáo phân
nhóm theo mission/orbit/datatake/swath/polarisation, ghi slice thiếu và nhận
diện cửa sổ acquisition chung giữa hai sản phẩm. Số record quan sát vẫn
bao gồm các record chung: không coi đó là các mẫu độc lập.

Exit code 0 khi mọi annotation khớp ít nhất một trong hai convention và
việc so context không lỗi. Exit code 1 khi có phản ví dụ/lỗi; JSON vẫn được
lưu. Việc khớp chỉ là kiểm chứng các giả thuyết đã nêu, không chứng minh
thuật toán IPF hoặc độ chính xác Doppler. Xem [báo cáo layout](sentinel1_dce_azimuth_layout.md).

## Kiểm chứng ngày 22/10/2026

- Live search seed `002C4C`: trả đúng 3 sản phẩm S1C/S6/HH-HV, không truncated.
- Live dry-run chọn HH: 6 file, tổng **3.548.208 byte** (annotation + manifest).
- Verify 9 XML cục bộ: 7 datatake; `last_line` khớp 3 annotation,
  `after_last_line` khớp 8 (hai slice chưa cuối khớp cả hai); không có mẫu
  cả hai đều sai; 2 cặp cửa sổ acquisition chung.
- Unit tests mock kiểm pagination/limit, file selection, tải/cached hash,
  file thiếu byte/XML lỗi, token refresh/OTP, redirect, path traversal,
  dry-run không login và báo cáo giữ phản ví dụ.

- Đã kiểm chứng sau lần tải có xác thực: 3 sản phẩm, 6 file, tổng
  **3.548.208 byte**; số byte, SHA-256, XML và cache đều hợp lệ, không còn
  file `.part`. Ba annotation giống byte-for-byte với bộ ba tải thủ công.
- Manifest của cả ba ghi **Sentinel-1 IPF 003.91**. Verify trên chính các
  file vừa tải nhận đủ slice 1/2/3, 8 estimate, không thiếu slice, 2 cặp
  cửa sổ chung. `last_line` khớp 3/3 annotation; `after_last_line` khớp 2/3
  vì hai slice chưa cuối không phụ thuộc tail convention. Slice cuối sai
  2,276 µs với `last_line`, 600,868 µs với `after_last_line`.
- Báo cáo tại `output/sentinel-1/copernicus/downloaded_verification.json`.
  Refresh token vẫn mới được kiểm bằng mock, chưa kiểm trong phiên thật.

## Nguồn tham khảo

- [CDSE OData: catalogue và Nodes](https://documentation.dataspace.copernicus.eu/APIs/OData.html).
- [CDSE token](https://documentation.dataspace.copernicus.eu/APIs/Token.html).
- [CDSETool download.py](https://github.com/CDSETool/CDSETool/blob/main/src/cdsetool/download.py):
  tham khảo cách tải file riêng qua Nodes. Tool dùng `requests` đã có trong
  lockfile, không thêm CDSETool làm dependency hoặc sao chép framework tải.
