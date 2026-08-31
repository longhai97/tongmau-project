# Tông Màu — Color Style Transfer Tool

Công cụ phân tích tông màu/ánh sáng/tương phản từ ảnh tham chiếu và áp dụng lên ảnh khác.

## Cấu trúc

```
frontend/
  index.html      — MVP: chạy hoàn toàn trong trình duyệt (khớp thống kê màu LAB),
                     không cần cài đặt, mở file là dùng được ngay.

backend/
  main.py, wct2/   — v4: AI photorealistic style transfer (WCT2), chạy local trên
                     GPU NVIDIA của bạn. Cần cài Python + torch trước.
                     Xem hướng dẫn chi tiết trong backend/README.md.
```

## Bắt đầu nhanh

**Chỉ muốn dùng ngay, không cài gì:** mở `frontend/index.html` bằng trình duyệt.

**Muốn dùng bản AI (chất lượng cao hơn, cần GPU):**
1. Làm theo `backend/README.md` để cài và chạy server local
2. (Bước tiếp theo, chưa làm) nối `frontend/index.html` gọi sang backend — hiện 2 phần đang chạy độc lập, frontend chưa có nút gọi API AI
