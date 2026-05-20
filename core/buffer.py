from collections import deque
import numpy as np

class SequenceBuffer:
    """
    Bộ đệm dạng cửa sổ trượt (rolling window) lưu trữ các vector keypoint gần nhất.
    Được sử dụng để gom đủ số lượng khung hình (ví dụ: 30 frames) trước khi 
    đưa vào mô hình AI (LSTM/GRU) để dự đoán cử chỉ động.
    """
    def __init__(self, max_length=30):
        # Sử dụng deque với kích thước cố định để tự động đẩy frame cũ ra khi thêm frame mới
        self.buffer = deque(maxlen=max_length)
        self.max_length = max_length

    def append(self, keypoints):
        """Thêm một vector keypoint mới của frame hiện tại vào bộ đệm."""
        self.buffer.append(keypoints)

    def is_ready(self):
        """Kiểm tra bộ đệm đã tích lũy đủ số lượng frame chưa."""
        return len(self.buffer) == self.max_length

    def get_sequence(self):
        """
        Trả về chuỗi sequence dạng mảng numpy để đưa vào mô hình AI.
        Output shape: (30, số lượng features)
        """
        return np.array(self.buffer)

    def clear(self):
        """Xóa sạch bộ đệm."""
        self.buffer.clear()
