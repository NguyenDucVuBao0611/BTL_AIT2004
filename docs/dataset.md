# 🤟 HƯỚNG DẪN TẢI VÀ SỬ DỤNG DATASET TỪ HUGGING FACE

Tài liệu này hướng dẫn cách lấy và tích hợp bộ dữ liệu cử chỉ (landmark keypoints dưới dạng file `.npy`) được lưu trữ trên Hugging Face của dự án.

* **Dataset Repo URL**: [https://huggingface.co/datasets/khongcoten111/datasetHF](https://huggingface.co/datasets/khongcoten111/datasetHF)

---

## 📂 Cấu trúc Dataset trên Hugging Face
Dữ liệu được tổ chức dưới dạng cấu trúc cây thư mục phù hợp trực tiếp cho việc huấn luyện (train split):
```text
datasetHF/
├── .gitattributes
├── train/
│   ├── hello/        # 30 folders sequence (0 - 29) chứa các file landmark npy
│   ├── idle/         # 30 folders sequence
│   ├── iloveyou/     # 30 folders sequence
│   ├── no/           # ...
│   ├── thanks/
│   └── yes/
```

Mỗi sequence chứa 30 file `.npy` (được đánh số từ `0.npy` đến `29.npy`), tương đương với 30 khung hình của một hành động. Kích thước vector mỗi file là `(1662,)`.

---

## ⚡ Cách tải Dataset về máy

Có hai cách để tải dataset về chạy trên máy cục bộ của bạn:

### Cách 1: Clone trực tiếp bằng Git (Khuyên dùng)
Mở cửa sổ terminal tại thư mục gốc của dự án (`BTL_AIT2004`) và chạy lệnh sau:

```bash
# 1. Cài đặt Git LFS nếu máy chưa có
git lfs install

# 2. Clone repository dataset về thư mục con datasetHF
git clone https://huggingface.co/datasets/khongcoten111/datasetHF
```

---

### Cách 2: Tải tự động bằng Python script
Nếu bạn không muốn cài đặt Git LFS thủ công, bạn có thể tải thông qua thư viện `huggingface_hub` bằng cách tạo một file script Python nhỏ (ví dụ: `download_dataset.py`) và chạy nó:

```python
# Cài đặt thư viện trước nếu chưa có: pip install huggingface_hub
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="khongcoten111/datasetHF",
    repo_type="dataset",
    local_dir="datasetHF"
)
```

---

## 🛠️ Tích hợp và Cấu hình Dự án

Sau khi tải thành công thư mục `datasetHF` về dự án chính, hãy đảm bảo các cài đặt sau được định hình chính xác:

### 1. File Cấu hình trung tâm (`core/config.py`)
Đường dẫn của bộ dữ liệu cần trỏ tới thư mục `datasetHF/train` để các script thu thập dữ liệu mới hoặc huấn luyện tìm đúng vị trí:
```python
# core/config.py
import os

ACTIONS = ["hello", "idle", "thanks", "iloveyou", "yes", "no"]
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30

# Đường dẫn thư mục gốc lưu dataset
DATA_PATH = os.path.join("datasetHF", "train")
```

### 2. Kiểm tra chất lượng dữ liệu vừa tải
Chạy script kiểm tra chất lượng dữ liệu tự động có sẵn để chắc chắn toàn bộ file `.npy` đã được tải xuống đầy đủ và không bị lỗi định dạng:
```bash
python validate_dataset.py
```

Báo cáo thống kê sẽ hiển thị ngay trên màn hình Terminal của bạn.
