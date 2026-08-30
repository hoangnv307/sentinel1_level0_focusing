# Sentinel 1 Level 0 Decoding Demo

Phát triển từ repo của tác giả [Rich-Hall](https://github.com/Rich-Hall/sentinel1decoder/tree/main/sentinel1decoder) 

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Bộ nhớ đệm của notebook

Notebook dùng `joblib.Memory` để lưu kết quả các bước tốn thời gian vào `.cache/sentinel1/`.

Các bước được lưu:

- Giải mã dữ liệu I/Q của chunk 13 và chunk 14.
- Ước lượng độ lệch I/Q của hai chunk.
- Nén cự ly của chunk 13 và chunk 14.
- Ước lượng Doppler centroid.
- Tạo ảnh SLC hội tụ.

Effective velocity, block layout, bảng thống kê và biểu đồ vẫn được tính hoặc vẽ lại vì nhẹ hơn các bước trên.

### Cấu hình

```python
CACHE_POLICY = "reuse"
CACHE_TAG = "default"
PIPELINE_VERSION = 1
```

### Các chế độ sử dụng

| Trường hợp | Cấu hình | Kết quả |
|---|---|---|
| Làm việc hằng ngày | `CACHE_POLICY = "reuse"` | Dùng cache nếu có; nếu thiếu thì tính và lưu. |
| Muốn tính lại tất cả checkpoint | `CACHE_POLICY = "refresh"` | Luôn tính lại và cập nhật cache của tag hiện tại. |
| Chỉ cho phép dùng dữ liệu đã có | `CACHE_POLICY = "readonly"` | Đọc cache; dừng ngay nếu checkpoint bị thiếu. |
| Chạy thử mà không dùng cache | `CACHE_POLICY = "off"` | Không đọc và không ghi cache; dữ liệu cũ được giữ nguyên. |

### Khi chạy toàn bộ notebook

Lần chạy đầu tiên với `reuse`, các checkpoint chưa tồn tại sẽ được tính và lưu. Những lần sau với cùng file đầu vào, tham số và phiên bản pipeline:

- Decode chunk 13/14: đọc từ cache.
- I/Q bias chunk 13/14: đọc từ cache.
- Range compression chunk 13/14: đọc từ cache.
- Doppler Estimation: đọc từ cache.
- Focus SLC: đọc từ cache.
- Effective velocity, block layout và biểu đồ: vẫn chạy lại.

Các ma trận NumPy lớn được đọc bằng memory-map ở chế độ chỉ đọc, giúp tránh nạp thêm một bản sao đầy đủ vào RAM.

### Khi thay đổi tham số hoặc code

| Thay đổi | Việc cần làm |
|---|---|
| Đổi file đầu vào, chunk hoặc tham số xử lý | Không cần thao tác; Joblib tạo cache key mới. |
| Sửa code có thể ảnh hưởng kết quả | Tăng `PIPELINE_VERSION` để invalid cả stage đó và các stage phía sau. |
| Đổi hằng số toàn cục, AUX data hoặc dependency gián tiếp | Tăng `PIPELINE_VERSION`. |
| Muốn thử nghiệm mà không ảnh hưởng cache mặc định | Đổi `CACHE_TAG`, ví dụ `"rcmc_test"`. |
| Sửa trực tiếp nội dung một mảng trung gian | Đổi `CACHE_TAG` và dùng `refresh`. |

Joblib tự phát hiện thay đổi trực tiếp trong hàm được cache, nhưng `PIPELINE_VERSION` vẫn cần thiết để invalid các stage phía sau và những dependency nằm ngoài thân hàm.

### Dọn cache

Xóa cache của `CACHE_TAG` hiện tại trong notebook:

```python
checkpoints.memory.clear(warn=False)
```

Cache dùng pickle cho một số object Python. Chỉ sử dụng cache do chính dự án tạo ra.

Cache được tạo bởi cơ chế cũ không được Joblib sử dụng lại. Sau khi chạy thành công một lần với Joblib, có thể xóa các thư mục cache cũ `raw_decoding`, `range_compression` và `focus_slc` để giải phóng dung lượng.
