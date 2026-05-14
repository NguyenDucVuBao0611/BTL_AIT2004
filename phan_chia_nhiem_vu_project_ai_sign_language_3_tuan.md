# PHÂN CHIA NHIỆM VỤ PROJECT AI NHẬN DIỆN NGÔN NGỮ KÝ HIỆU REALTIME

## Công nghệ sử dụng
- Python
- OpenCV
- MediaPipe Holistic
- TensorFlow / Keras
- NumPy
- Matplotlib
- Webcam realtime inference

---

# THÀNH VIÊN NHÓM

| Người | Vai trò chính |
|---|---|
| Người 1 | Data & MediaPipe Engineer |
| Người 2 | AI Model Engineer |
| Người 3 | Inference & UI Engineer |

---

# TUẦN 1 — DATASET & NỀN TẢNG

## Người 1 — Data & MediaPipe Engineer
### Nhiệm vụ
- Setup môi trường Python
- Cài thư viện:
  - OpenCV
  - MediaPipe
  - TensorFlow
- Kết nối webcam realtime
- Tích hợp MediaPipe Holistic
- Hiển thị landmark realtime:
  - Pose
  - Left Hand
  - Right Hand
- Viết hàm extract keypoint
- Test shape vector `(126,)`
- Viết script thu thập dữ liệu
- Lưu dữ liệu `.npy`
- Visualize landmark để kiểm tra chất lượng

### Output
```bash
project/
├── dataset/
├── core/
│   └── mediapipe_detector.py
└── collect_data.py
```

---

## Người 2 — AI Model Engineer
### Nhiệm vụ
- Tìm hiểu dữ liệu keypoint
- Chuẩn hóa format sequence
- Thiết kế pipeline train
- Chuẩn bị notebook train
- Viết script build:
  - X.npy
  - y.npy
- Chia train/validation

### Output
```bash
training/
└── prepare_dataset.py
```

---

## Người 3 — Inference & UI Engineer
### Nhiệm vụ
- Tìm hiểu pipeline inference realtime
- Thiết kế rolling window 30 frames
- Viết module sequence buffer
- Test realtime FPS
- Thiết kế UI overlay:
  - text
  - confidence
  - FPS

### Output
```bash
core/
└── buffer.py
```

---

# MILESTONE TUẦN 1
- MediaPipe chạy ổn định
- Webcam detect landmark realtime
- Thu được dataset gesture
- Sequence shape đúng `(30,126)`

---

# TUẦN 2 — TRAINING & INFERENCE

## Người 1 — Data & MediaPipe Engineer
### Nhiệm vụ
- Thu thêm dữ liệu
- Làm sạch dataset
- Kiểm tra missing landmark
- Augment dữ liệu nhẹ
- Debug dữ liệu lỗi

### Output
```bash
project/
└── dataset/
```

---

## Người 2 — AI Model Engineer
### Nhiệm vụ
- Train LSTM baseline
- Log:
  - accuracy
  - loss
- Thử GRU
- So sánh model
- Vẽ confusion matrix
- Export model `.h5`
- Đặt target:
  - val_acc > 90%

### Output
```bash
project/
├── models/
│   └── sign_lstm.h5
└── training/
    ├── train.py
    └── evaluate.py
```

---

## Người 3 — Inference & UI Engineer
### Nhiệm vụ
- Load model `.h5`
- Kết nối inference realtime
- Predict realtime
- Smoothing prediction:
  - majority vote
  - threshold
- Overlay text lên webcam
- Hiển thị:
  - confidence bar
  - FPS
- Test offline inference

### Output
```bash
project/
├── core/
│   └── inference.py
└── main.py
```

---

# MILESTONE TUẦN 2
- Model train xong
- Accuracy > 90%
- Inference realtime hoạt động
- FPS > 15

---

# TUẦN 3 — POLISH & DEMO

## Người 1 — Data & MediaPipe Engineer
### Nhiệm vụ
- Thu thêm dữ liệu khó
- Giảm false positive
- Fine-tune detection
- Test nhiều điều kiện ánh sáng
- Viết document dataset

### Output
```bash
project/
└── docs/
    └── dataset.md
```

---

## Người 2 — AI Model Engineer
### Nhiệm vụ
- Fine-tune model
- So sánh:
  - LSTM
  - GRU
  - MLP
- Chọn model tốt nhất
- Tối ưu inference speed
- Chuẩn bị biểu đồ:
  - accuracy
  - loss
  - confusion matrix

### Output
```bash
project/
├── reports/
└── final_model.h5
```

---

## Người 3 — Inference & UI Engineer
### Nhiệm vụ
- Hoàn thiện UI demo
- Thêm:
  - confidence bar
  - sentence history
- Test end-to-end
- Fix bug realtime
- Chuẩn bị video demo
- Chuẩn bị slide demo

### Output
```bash
project/
├── demo/
└── presentation/
```

---

# MILESTONE TUẦN 3
- Demo realtime ổn định
- FPS > 20
- Nhận diện realtime chính xác
- Có video + slide demo

---

# CẤU TRÚC PROJECT

```bash
project/
│
├── dataset/
├── models/
├── training/
├── core/
├── demo/
├── docs/
│
├── collect_data.py
├── main.py
├── requirements.txt
└── README.md
```

---

# QUY TẮC LÀM VIỆC NHÓM

- Push code mỗi ngày
- Commit rõ ràng
- Review code chéo
- Mỗi người phải hiểu toàn bộ pipeline
- Test integration cuối mỗi tuần
- Demo thử trước ngày báo cáo

---

# KPI CUỐI PROJECT

| Tiêu chí | Mục tiêu |
|---|---|
| Gesture | 5–10 |
| Accuracy | >90% |
| FPS realtime | >20 |
| Delay | <1s |
| Model size | Nhẹ |
| Demo | Realtime ổn định |

