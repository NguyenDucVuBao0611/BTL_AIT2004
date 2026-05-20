import os
import time
import argparse
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

def run_data_collection(target_action=None, target_sequence=None):
    """
    Vòng lặp chính quay webcam và thu thập dữ liệu.
    Hỗ trợ chế độ nhắm chọn (Targeted Mode) để ghi đè sequence cụ thể.
    """
    create_dataset_folders()
    
    cap = cv2.VideoCapture(0)
    print("[*] Đang khởi động camera thu thập dữ liệu...")
    
    # Xác định danh sách các hành động sẽ chạy
    actions_to_run = [target_action] if target_action else ACTIONS
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions_to_run:
            
            # Xác định các sequence sẽ chạy cho hành động này
            if target_sequence is not None:
                sequences_to_run = [target_sequence]
            else:
                sequences_to_run = list(range(NO_SEQUENCES))
                
            for sequence in sequences_to_run:
                # Vòng lặp while cho phép quay lại (Redo) dễ dàng
                while True:
                    frames_cache = []      # Lưu trữ ảnh gốc để phát lại (Replay)
                    keypoints_cache = []   # Lưu trữ keypoints tương ứng
                    
                    # --- Giai đoạn 1: Chuẩn bị quay (Đếm ngược mượt mà 3 giây) ---
                    for countdown in range(3, 0, -1):
                        # Chạy vòng lặp hiển thị liên tục trong 1 giây (25 frames x 40ms = 1000ms)
                        for _ in range(25):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            image_mirrored = cv2.flip(frame, 1)
                            
                            header_text = f'PREPARING: "{action.upper()}" (Seq #{sequence + 1})'
                            if target_action:
                                header_text += " [TARGETED OVERWRITE]"
                                
                            cv2.putText(
                                image_mirrored, header_text, (20, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if target_action else (0, 255, 0), 2, cv2.LINE_AA
                            )
                            cv2.putText(
                                image_mirrored, f'Recording starts in: {countdown}...', (20, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA
                            )
                            cv2.imshow('Data Collection Feed', image_mirrored)
                            cv2.waitKey(40)

                    # --- Giai đoạn 2: Quay dữ liệu (30 frames) ---
                    print(f"[*] Đang quay Sequence #{sequence + 1} cho từ: {action}")
                    for frame_num in range(SEQUENCE_LENGTH):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Chạy MediaPipe trên frame gốc
                        image, results = mediapipe_detection(frame, holistic)
                        # Trích xuất keypoints
                        keypoints = extract_keypoints(results)
                        
                        # Vẽ landmark trực quan để người dùng theo dõi
                        draw_styled_landmarks(image, results)
                        image_mirrored = cv2.flip(image, 1)
                        
                        cv2.putText(
                            image_mirrored, f'RECORDING: "{action}" | Seq: {sequence + 1} | Frame: {frame_num + 1}/{SEQUENCE_LENGTH}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
                        )
                        cv2.imshow('Data Collection Feed', image_mirrored)
                        cv2.waitKey(40) # Tốc độ ghi hình khoảng ~25 FPS (40ms) giúp kéo dài thời gian quay tự nhiên hơn

                        # Cache lại để replay
                        frames_cache.append(image_mirrored)
                        keypoints_cache.append(keypoints)

                    # --- Giai đoạn 3: Replay và Hỏi ý kiến người dùng ---
                    print("[*] Quay xong. Đang phát lại (Replay)...")
                    decision = None
                    while decision is None:
                        for cached_frame in frames_cache:
                            frame_to_show = cached_frame.copy()
                            cv2.putText(
                                frame_to_show, '--- REPLAYING (PREVIEW) ---', (150, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA
                            )
                            cv2.putText(
                                frame_to_show, 'Press: [Y] to Save/Overwrite | [R] to Redo | [Q] to Quit', (20, 450), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                            )
                            cv2.imshow('Data Collection Feed', frame_to_show)
                            
                            key = cv2.waitKey(50) & 0xFF
                            if key == ord('y'):
                                decision = 'save'
                                break
                            elif key == ord('r'):
                                decision = 'redo'
                                break
                            elif key == ord('q'):
                                decision = 'quit'
                                break
                                
                    # --- Giai đoạn 4: Thực thi quyết định ---
                    if decision == 'save':
                        # Ghi đè vào folder sequence chỉ định
                        for idx, kp in enumerate(keypoints_cache):
                            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{idx}.npy")
                            np.save(npy_path, kp)
                        print(f"[+] Đã ghi đè/lưu thành công Sequence #{sequence + 1}")
                        break # Thoát khỏi vòng lặp True của sequence hiện tại để sang sequence tiếp theo
                    elif decision == 'redo':
                        print(f"[-] HỦY bỏ lượt quay vừa rồi. Chuẩn bị quay lại Sequence #{sequence + 1}...")
                        # Vòng lặp while True tiếp tục quay lại sequence này
                    elif decision == 'quit':
                        print("[!] Đã dừng thu thập dữ liệu khẩn cấp.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        print("\n[*] THU THẬP DỮ LIỆU HOÀN TẤT THÀNH CÔNG!")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chương trình thu thập dữ liệu ngôn ngữ ký hiệu.")
    parser.add_argument("--action", type=str, default=None, help="Tên cử chỉ muốn quay đè (ví dụ: hello)")
    parser.add_argument("--sequence", type=int, default=None, help="Số thứ tự sequence muốn quay đè (0-indexed, ví dụ: 5)")
    
    args = parser.parse_args()
    
    # Kiểm tra tính hợp lệ của tham số truyền vào
    if args.action and args.action not in ACTIONS:
        print(f"[!] Lỗi: Hành động '{args.action}' không nằm trong cấu hình ACTIONS của config.py")
    elif (args.action is not None and args.sequence is None) or (args.action is None and args.sequence is not None):
        print("[!] Lỗi: Bạn cần truyền đồng thời cả --action và --sequence để ghi đè.")
    else:
        # Nếu có truyền sequence, trừ đi 1 để chuyển từ 1-indexed (người dùng nhập) sang 0-indexed (lập trình)
        seq_idx = args.sequence - 1 if args.sequence is not None else None
        if seq_idx is not None and (seq_idx < 0 or seq_idx >= NO_SEQUENCES):
            print(f"[!] Lỗi: Sequence chỉ được nằm trong khoảng từ 1 đến {NO_SEQUENCES}")
        else:
            run_data_collection(target_action=args.action, target_sequence=seq_idx)
