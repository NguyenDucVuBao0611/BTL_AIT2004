import os
import time
import cv2
import numpy as np
import mediapipe as mp

# Import các thông số cấu hình và module xử lý MediaPipe
from core.config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, DATA_PATH
from core.mediapipe_detector import (
    mp_holistic, 
    mediapipe_detection, 
    draw_styled_landmarks, 
    extract_keypoints
)

def create_dataset_folders():
    """Tạo sẵn cấu trúc thư mục lưu trữ dataset nếu chưa có."""
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
    print(f"[*] Cấu trúc thư mục dataset đã sẵn sàng tại: {DATA_PATH}")

def run_data_collection():
    """Vòng lặp chính quay webcam và thu thập dữ liệu."""
    create_dataset_folders()
    
    cap = cv2.VideoCapture(0)
    print("[*] Đang khởi động camera thu thập dữ liệu...")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            for sequence in range(NO_SEQUENCES):
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        print("[!] Lỗi không đọc được khung hình từ camera.")
                        break
                    
                    # 1. AI nhận diện trên frame gốc
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # 2. Vẽ khung xương trang trí
                    draw_styled_landmarks(image, results)
                    
                    # 3. Lật ảnh gương để hiển thị UI cho người dùng
                    image_mirrored = cv2.flip(image, 1)
                    
                    # ========================================================
                    # GIAO DIỆN HƯỚNG DẪN THU THẬP TRÊN MÀN HÌNH
                    # ========================================================
                    if frame_num == 0:
                        # Màn hình chờ bắt đầu sequence mới
                        cv2.putText(
                            image_mirrored, f'STARTING COLLECTION FOR "{action.upper()}"', (40, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA
                        )
                        cv2.putText(
                            image_mirrored, f'Get ready for video #{sequence + 1}/{NO_SEQUENCES}...', (40, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
                        )
                        cv2.imshow('Data Collection Feed', image_mirrored)
                        cv2.waitKey(2000)  # Dừng 2 giây để người dùng chuẩn bị tư thế
                    else:
                        # Màn hình hiển thị tiến độ trong lúc đang quay
                        cv2.putText(
                            image_mirrored, f'Action: {action} | Video #{sequence + 1} | Frame: {frame_num + 1}/{SEQUENCE_LENGTH}', (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
                        )
                        cv2.imshow('Data Collection Feed', image_mirrored)
                    
                    # ========================================================
                    # LƯU DỮ LIỆU KEYPOINT VECTOR VÀO FILE .npy
                    # ========================================================
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    np.save(npy_path, keypoints)

                    # Bấm 'q' để dừng khẩn cấp
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("\n[!] Đã dừng thu thập dữ liệu khẩn cấp.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                        
        print("\n[*] THU THẬP DỮ LIỆU HOÀN TẤT THÀNH CÔNG!")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_data_collection()
