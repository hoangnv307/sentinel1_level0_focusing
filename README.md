# Sentinel-1 Level-0 Decoding Demo

Pipeline xử lý Sentinel-1 Level-0 thành ảnh SLC, phát triển từ dự án của
[Rich Hall](https://github.com/Rich-Hall/sentinel1decoder).

## Cài đặt

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Mở notebook

Notebook dùng **marimo** và được lưu dưới dạng file Python:

```bash
marimo edit workflows/sentinel-1/focus_chunk_13.py
marimo edit workflows/sentinel-1/focus_chunks_13_14.py
```

Marimo tự chạy cell theo quan hệ phụ thuộc và lưu các kết quả tốn thời gian
vào `.cache/sentinel1/`. Không cần cấu hình phiên bản hoặc policy cache thủ
công.

Các kết quả được lưu gồm:

- Decode dữ liệu I/Q của chunk 13 và 14.
- I/Q bias của từng chunk.
- Range-compressed data của từng chunk.
- Doppler centroid estimation.
- Focused SLC.

## Các trường hợp sử dụng

| Thao tác | Điều xảy ra |
|---|---|
| Mở lại notebook, không đổi gì | Các bước nặng được đọc từ ổ đĩa; bảng và biểu đồ vẫn có thể được dựng lại. |
| Sửa cell DCE | DCE được tính và lưu lại. Focus chỉ tính lại nếu dữ liệu DCE thay đổi. Decode và Range Compression dùng dữ liệu đã lưu. |
| Sửa source DCE trong `src/sentinel1_processing/doppler_centroid_estimation.py` | Tương tự sửa cell DCE. Marimo tự reload module và chạy lại phần phụ thuộc. |
| Sửa Range Compression | Range Compression được tính lại. DCE và Focus chỉ tính lại nếu dữ liệu đầu vào của chúng thay đổi. Decode vẫn dùng dữ liệu đã lưu. |
| Sửa Focus | Chỉ Focus chạy lại. |
| Đổi tham số hoặc file đầu vào | Cell dùng giá trị đó và các cell phía sau chạy lại; các cell phía trước không liên quan được giữ nguyên. |
| Chỉ sửa cell `print`, bảng hoặc biểu đồ | Chỉ phần hiển thị liên quan chạy lại; dữ liệu xử lý không được tính lại. |

Khi một cache mới thay thế cache cũ của cùng công đoạn, notebook tự xóa bản
cũ sau khi Focus hoàn tất để thư mục cache không tăng mãi.

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
- `references/`: tài liệu và metadata tham chiếu.
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

Nếu đặt `print()` hoặc vẽ biểu đồ bên trong block `mo.persistent_cache`, phần
hiển thị sẽ bị bỏ qua khi cache được dùng. Vì vậy các block cache chỉ chứa
tính toán tạo dữ liệu.

Cache dùng pickle; chỉ mở cache do chính dự án tạo ra. Muốn tính lại toàn bộ,
xóa thư mục `.cache/sentinel1/` rồi mở lại notebook.
