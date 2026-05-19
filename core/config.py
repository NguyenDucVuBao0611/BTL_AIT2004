import os

# ================================================================
# BẢNG ĐIỀU KHIỂN TRUNG TÂM — Chỉ cần sửa ở đây, toàn project tự cập nhật
# ================================================================

# Danh sách các từ/ký hiệu cần thu thập và nhận diện
# Thêm hoặc xóa gesture tại đây, collect_data.py và train.py sẽ tự nhận
ACTIONS = ["hello", "thanks", "iloveyou", "yes", "no"]

# Số lượng video mẫu thu thập cho mỗi gesture
NO_SEQUENCES = 30

# Số frame trong mỗi video mẫu (rolling window)
SEQUENCE_LENGTH = 30

# Đường dẫn thư mục gốc lưu dataset
DATA_PATH = os.path.join("dataset")
