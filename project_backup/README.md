# MentalHealth_AI_Project

## Giới thiệu
Hệ thống phân tích sức khỏe tâm thần từ văn bản tiếng Việt, bao gồm hai module chính:

- **Module 1 – Phân tích cảm xúc & Hate Speech**  
  Nhận diện 7 cảm xúc (Anger, Disgust, Enjoyment, Fear, Other, Sadness, Surprise) và 3 mức độ hate speech (Clean, Offensive, Hate) từ văn bản tự do.

- **Module 2 – Dự đoán bệnh tâm lý (Multi‑label, top‑k)**  
  Dự đoán danh sách các bệnh tâm lý có khả năng mắc phải từ văn bản, trả về top‑k bệnh kèm xác suất (mặc định k=5).

Cả hai module đều sử dụng mô hình **PhoBERT‑base** (VinAI) và có thể chạy offline.

---

## Kết quả nổi bật

| Module | Task | Chỉ số chính | Giá trị |
|--------|------|--------------|---------|
| **Module 1** | Emotion | Weighted F1 | **64%** |
| | Hate Speech | Weighted F1 | **79%** |
| | Trung bình | Accuracy | **71.5%** |
| **Module 2** | Multi-label (68 nhãn) | Top‑5 Accuracy | **94.4%** |
| | | Test F1 micro | **78.96%** |

> Chi tiết kết quả xem trong các báo cáo tương ứng.

---

## Cấu trúc thư mục
MentalHealth_AI_Project/
├── checkpoints/ # Trọng số mô hình đã huấn luyện
│ ├── best_multitask_model.pth
│ └── best_mental_health_multilabel.pth
├── data/
│ ├── external/ # xử lý teencode.xlsx
│ ├── processed/ # Dữ liệu train/val/test (CSV)
│ └── raw/ # Dữ liệu thô (Excel)
├── notebooks/ # Jupyter notebooks phân tích
├── scripts/ # Scripts huấn luyện và đánh giá
│ ├── module1/ # train_multitask.py, test_module1.py
│ └── module2/ # train_module2.py, test_module2.py
├── src/ # Mã nguồn chính
│ ├── module1/ # dataset, model, class_weights
│ └── module2/ # dataset, model (multi‑label)
├── requirements.txt
├── README.md
├── run_train_module#1.bat
├── run_test_module#1.bat
├── run_train_module#2.bat
├── run_test_module#2.bat
└── setup_env.bat # Script cài đặt môi trường

---

## Yêu cầu hệ thống

- **Python 3.12** (các phiên bản khác có thể gây lỗi thư viện)
- GPU NVIDIA (khuyến nghị) với CUDA ≥ 11.8, hoặc CPU (chậm hơn)
- Dung lượng RAM ≥ 16 GB (khuyến nghị), VRAM ≥ 4 GB cho training

---

## Cài đặt nhanh

1. **Clone repository**  
   ```bash
   git clone <repo-url>
   Sau đó click vào setup_env.bat để cài đặt thư viện

2. **Tải trọng số mô hình**
    Liên hệ nhóm để để lấy file bat

## Lưu ý khi triển khai
Teencode: Module 1 sử dụng từ điển Xử lý teencode.xlsx (chỉ thay thế từ viết tắt, giữ nguyên từ lóng).

Tokenizer: Cả hai module đều tokenize trước toàn bộ dataset trong __init__ để tăng tốc.

Device: Mặc định dùng GPU (cuda). Nếu không có GPU, sửa DEVICE = torch.device("cpu") trong các script.

Threshold cho module 2: Khi dự đoán thực tế, nên giữ ngưỡng mặc định (0.1) hoặc điều chỉnh theo nhu cầu recall/precision.


## Tác giả
**Name: Phan Trường Giang**
**Contact me:**
Gmail: giangkirito11@gmail.com
Phone number: 0353960753

## About dataset
├── Module#1
│ ├── UIT_VSEMC
│ └── UIT_ViHSD
├── Module#2
│ ├── VMHQA: Đã được cấp phép sử dụng, được chỉnh sửa nhằm phù hợp với yêu cầu của mô hình.






