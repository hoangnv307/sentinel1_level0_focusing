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
marimo edit sentinel1_level_1_decoder.py
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
