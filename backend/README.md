# Tông Màu — Backend AI (v4, chạy local trên GPU của bạn)

Photorealistic color/style transfer dùng model **WCT2** (clovaai, MIT license).
Chạy hoàn toàn trên máy bạn — không có chi phí server, không gửi ảnh lên cloud.

## 1. Cài đặt

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Cài torch bản CUDA đúng với driver GPU của bạn — LẤY LỆNH CHÍNH XÁC tại
# https://pytorch.org/get-started/locally/ (chọn Stable, hệ điều hành, Pip, CUDA)
# Ví dụ với CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài các thư viện còn lại
pip install -r requirements.txt
```

Kiểm tra CUDA nhận GPU chưa:
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Nếu in ra `False`, torch bạn cài không khớp bản CUDA/driver — quay lại bước cài torch, chọn đúng phiên bản.

## 2. Chạy server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Mở `http://localhost:8000/health` để kiểm tra — sẽ thấy `"cuda_available": true` và tên GPU.
Swagger UI để test thử API trực tiếp: `http://localhost:8000/docs`

## 3. Dùng từ điện thoại (cùng wifi với máy tính)

1. Tìm IP nội bộ của máy tính:
   - Windows: `ipconfig` (tìm IPv4, dạng `192.168.x.x`)
   - Mac/Linux: `ifconfig` hoặc `ip addr`
2. Trên điện thoại (cùng wifi), gọi API tới `http://192.168.x.x:8000` thay vì `localhost`.

## 4. API

`POST /style-transfer` — multipart/form-data:
| field | kiểu | mô tả |
|---|---|---|
| `target` | file | ảnh cần chỉnh |
| `references` | file (nhiều) | 1 hoặc nhiều ảnh tham chiếu |
| `alpha` | float, 0–1 | cường độ áp style (mặc định 1.0) |
| `image_size` | int | giới hạn cạnh dài nhất khi xử lý (mặc định 1024) |

Trả về: ảnh PNG kết quả.

Test nhanh bằng curl:
```bash
curl -X POST http://localhost:8000/style-transfer \
  -F "target=@/duong/dan/anh_can_chinh.jpg" \
  -F "references=@/duong/dan/anh_tham_chieu.jpg" \
  -F "alpha=0.9" \
  -o ket_qua.png
```

## Xử lý theo tile (tiling) — vì sao VRAM không còn phụ thuộc image_size

Bản gốc WCT2 xử lý toàn ảnh trong một lần — ở `image_size=1024` cần khoảng **~5.5GB VRAM**, vượt quá các GPU phổ thông 4GB (vd. GTX 1650, 3935MB VRAM thực dùng được).

Từ bản này, ảnh nội dung (`target`) được cắt thành các ô (tile) ~384px chồng mép, xử lý riêng từng ô rồi ghép lại. Điểm quan trọng: WCT chuẩn hóa màu (whitening) dựa trên thống kê pixel — nếu tính riêng theo từng tile sẽ tạo **đường nối rõ rệt** giữa các ô. Để tránh việc này, thống kê whitening được tính **một lần từ toàn bộ ảnh** (qua bản thu nhỏ ~384px, xem `tiling.py`) rồi áp dụng thống nhất cho mọi tile — kết quả tương đương xử lý toàn ảnh một lần, chỉ riêng phần convolution mã hoá/giải mã chạy theo từng ô để tiết kiệm VRAM.

Nhờ vậy, nhu cầu VRAM gần như không đổi bất kể `image_size` (đã đo: ~2.2GB ở 1024px, ~2.5GB ở 2048px — mức trần) — kể cả GPU 4GB (GTX 1650) cũng chạy tốt ở độ phân giải tối đa 2048px, đổi lại thời gian xử lý tăng theo số tile.

## Giới hạn cần biết

- **Tốc độ**: đo thực tế trên GTX 1660 — ~4s/ảnh ở `image_size=1024`, ~14s ở `image_size=2048` (GPU yếu hơn như GTX 1650 sẽ chậm hơn khoảng 1.3–1.5 lần). Tăng `image_size` chủ yếu làm chậm hơn (nhiều tile hơn), không còn làm tăng rủi ro hết VRAM như trước.
- Có thể chỉnh `TILE_SIZE`/`TILE_OVERLAP`/`STYLE_MAX_DIM` ở đầu `main.py` — tăng `TILE_SIZE` nếu có GPU nhiều VRAM hơn (xử lý nhanh hơn, ít tile hơn), giảm nếu VRAM còn ít hơn 4GB.
- **Nhiều ảnh tham chiếu**: hiện gộp bằng cách chạy transfer riêng với từng ảnh tham chiếu rồi lấy trung bình pixel của các kết quả — đơn giản, ổn định, nhưng chưa phải cách tối ưu nhất về mặt lý thuyết.
- **Không có segmentation/mask theo vùng** (mặt, tóc, nền...) trong bản này — đó là phần việc riêng của v3 (face-aware masking), có thể kết hợp thêm sau.
- Server phải đang chạy thì frontend mới dùng được chế độ "AI nâng cao" — khác với bản MVP chạy thẳng trong trình duyệt.

## Giấy phép

Kiến trúc model trong thư mục `wct2/` lấy từ [clovaai/WCT2](https://github.com/clovaai/WCT2), giấy phép MIT (xem `wct2/LICENSE_WCT2.md`) — được phép dùng cho mục đích thương mại, chỉ cần giữ lại thông báo bản quyền.
