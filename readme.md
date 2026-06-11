# 🛡️ Hệ Thống Dự Báo & Phát Hiện Giao Dịch Gian Lận (Streamlit App)

Ứng dụng web tương tác thông minh được phát triển dựa trên nền tảng **Streamlit**, chuyển đổi trực tiếp từ quy trình phân tích học máy trong Notebook mục tiêu sang giao dịch sản xuất. Hệ thống sử dụng thuật toán mạnh mẽ **RandomForestClassifier** để nhận diện và khoanh vùng các mẫu giao dịch có xác suất gian lận hoặc rủi ro tín dụng.

## ✨ Tính Năng Cốt Lõi của Hệ Thống

- **Cấu hình tham số AI động:** Tự do tùy chỉnh số lượng cây (`n_estimators`), độ sâu (`max_depth`), `random_state` trực tiếp trên thanh Sidebar điều phối.
- **Tổng quan trực quan hóa cấu trúc dữ liệu:** Thống kê phân phối tần suất của 14 trường chỉ số (`X_1` tới `X_14`) bằng biểu đồ động **Plotly**.
- **Đánh giá hiệu năng chi tiết:** Cung cấp tức thời hệ thống chỉ số đo lường hiệu quả phân loại: Accuracy, Precision, Recall, F1-Score cùng Ma trận nhầm lẫn (Confusion Matrix).
- **Triển khai ứng dụng linh hoạt theo 2 chế độ:**
  - *Chế độ Đơn lẻ:* Nhập trực tiếp các chỉ số qua biểu mẫu giao diện (Form), trả về xác suất rủi ro thời gian thực.
  - *Chế độ Hàng loạt:* Tải lên tệp khách hàng quy mô lớn (.csv/.xlsx) và kết xuất tệp kết quả phân loại tự động.

## 📁 Cấu Trúc File Dữ Liệu Đầu Vào Kì Vọng

Tệp dữ liệu mẫu để đưa vào huấn luyện mô hình cần bao gồm các cột cấu trúc như sau:
- **Biến độc lập ($X$):** Gồm 14 cột định lượng có tên chính xác từ `X_1`, `X_2`, `X_3`, ..., `X_14`.
- **Biến mục tiêu ($y$):** Cột tên `default` nhận giá trị số nhị phân:
  - `0`: Giao dịch an toàn / Bình thường.
  - `1`: Giao dịch phát sinh rủi ro / Gian lận.

## 🛠️ Hướng Dẫn Cài Đặt Và Vận Hành

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo máy tính của bạn đã cài đặt phiên bản Python 3.9 đến Python 3.12. Mở terminal/command prompt tại thư mục chứa mã nguồn ứng dụng.

### Bước 2: Cài đặt các thư viện cần thiết
Chạy lệnh sau để tự động tải về các gói package tối ưu quy định trong file cấu hình hệ thống:
```bash
pip install -r requirements.txt
