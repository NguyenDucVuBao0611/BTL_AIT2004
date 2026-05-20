import numpy as np

class HeuristicClassifier:
    """
    Bộ phân loại cử chỉ thủ ngữ bằng luật hình học (Rule-based).
    Phân tích trạng thái của các ngón tay (duỗi/gập) và khoảng cách giữa các khớp
    để đưa ra dự đoán tạm thời khi chưa có mô hình AI deep learning.
    """
    def __init__(self):
        # Danh sách cử chỉ được hỗ trợ
        self.actions = ["idle", "hello", "thanks", "iloveyou", "yes", "no"]

    def _get_distance(self, p1, p2):
        """Tính khoảng cách Euclid giữa 2 điểm landmark (x, y, z)."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _analyze_hand(self, hand_landmarks, face_landmarks=None, pose_landmarks=None):
        """
        Phân tích trạng thái bàn tay để nhận diện cử chỉ.
        Trả về: tên cử chỉ đoán được và điểm tin cậy (confidence).
        """
        if not hand_landmarks:
            return "idle", 0.0

        lm = hand_landmarks.landmark

        # Cổ tay
        wrist = lm[0]

        # Kiểm tra trạng thái Gập/Duỗi của 4 ngón chính (trỏ, giữa, áp út, út)
        # So sánh tọa độ Y của đầu ngón tay với khớp PIP (khớp thứ 2 từ đầu ngón)
        # Hệ tọa độ MediaPipe: Y giảm dần từ dưới lên trên (đầu ngón duỗi thẳng sẽ có Y nhỏ hơn khớp bên dưới)
        index_open = lm[8].y < lm[6].y
        middle_open = lm[12].y < lm[10].y
        ring_open = lm[16].y < lm[14].y
        pinky_open = lm[20].y < lm[18].y

        # Ngón cái: kiểm tra khoảng cách ngang (trục X) hoặc khoảng cách đến khớp ngón út
        # Nếu khoảng cách từ đầu ngón cái (4) đến gốc ngón trỏ (5) lớn hơn khoảng cách từ khớp khủy ngón cái (2) đến gốc ngón trỏ (5)
        thumb_open = self._get_distance(lm[4], lm[5]) > self._get_distance(lm[2], lm[5])

        # --- LUẬT BẮT CỬ CHỈ ---

        # 1. YES (Nắm đấm tay - fist)
        # Tất cả các ngón tay đều gập
        if not index_open and not middle_open and not ring_open and not pinky_open:
            return "yes", 0.95

        # 2. I LOVE YOU (Ngón cái, ngón trỏ, ngón út duỗi - ngón giữa, áp út gập)
        if index_open and pinky_open and not middle_open and not ring_open:
            return "iloveyou", 0.95

        # 3. NO (Ngón trỏ và ngón giữa duỗi - ngón cái, áp út, út gập)
        if index_open and middle_open and not ring_open and not pinky_open:
            return "no", 0.95

        # 4. THANKS (Tay mở rộng đặt gần cằm/miệng)
        # Phối hợp với landmark khuôn mặt hoặc dùng điểm Pose Mũi (index 0) làm dự phòng nếu mất diện khuôn mặt
        if index_open and middle_open and ring_open and pinky_open:
            ref_landmark = None
            if face_landmarks:
                ref_landmark = face_landmarks.landmark[152] # Cằm (chin)
                max_dist = 0.15
            elif pose_landmarks:
                ref_landmark = pose_landmarks.landmark[0]   # Mũi (nose) làm dự phòng
                max_dist = 0.18 # Cho phép khoảng cách xa hơn một chút vì mũi cao hơn cằm
                
            if ref_landmark:
                dist_to_ref = self._get_distance(lm[12], ref_landmark)
                # Nếu bàn tay mở và nằm rất gần mặt mốc
                if dist_to_ref < max_dist:
                    return "thanks", 0.98

        # 5. HELLO (Xòe cả bàn tay - mở hoàn toàn và không ở gần cằm/mặt)
        if index_open and middle_open and ring_open and pinky_open:
            return "hello", 0.95

        return "idle", 0.5

    def predict(self, results):
        """
        Nhận diện cử chỉ dựa trên kết quả MediaPipe Holistic.
        Ưu tiên nhận diện bàn tay phải, sau đó đến bàn tay trái.
        """
        # Trích xuất landmarks của tay phải, tay trái, mặt và tư thế (pose)
        right_hand = results.right_hand_landmarks
        left_hand = results.left_hand_landmarks
        face = results.face_landmarks
        pose = results.pose_landmarks

        # Thử nhận diện bằng tay phải trước
        gesture, conf = self._analyze_hand(right_hand, face, pose)
        
        # Nếu tay phải không hoạt động, thử nhận diện bằng tay trái
        if gesture == "idle" and left_hand:
            gesture, conf = self._analyze_hand(left_hand, face, pose)

        # Tạo vector xác suất mô phỏng để vẽ biểu đồ
        probabilities = [0.01] * len(self.actions)
        try:
            idx = self.actions.index(gesture)
            probabilities[idx] = conf
            # Phân phối phần còn lại cho các class khác để biểu đồ trông sống động
            remaining = 1.0 - conf
            for i in range(len(probabilities)):
                if i != idx:
                    probabilities[i] = remaining / (len(self.actions) - 1)
        except ValueError:
            probabilities[0] = 0.9  # Mặc định idle

        return gesture, conf, probabilities
