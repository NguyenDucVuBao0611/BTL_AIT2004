import os
# Tắt cảnh báo log không cần thiết của TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

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

# ---------------------------------------------------------
# HÀM THƯ VIỆN: DỰNG LẠI KHUNG XƯƠNG TỪ VECTOR CỌA ĐỘ (.NPY)
# ---------------------------------------------------------
def draw_skeleton_on_canvas(kp_vector, width=640, height=480):
    """
    Dựng lại toàn bộ khung xương mặt, thân và 2 tay từ vector tọa độ 1662 phần tử.
    Vẽ trực tiếp lên canvas nền đen.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if kp_vector.shape[0] != 1662:
        return canvas
        
    pose = kp_vector[0:132].reshape((33, 4))
    face = kp_vector[132:1536].reshape((468, 3))
    lh = kp_vector[1536:1599].reshape((21, 3))
    rh = kp_vector[1599:1662].reshape((21, 3))
    
    # 1. Vẽ mặt (Face mesh dots - Màu xám mờ)
    if not np.all(face == 0):
        for pt in face:
            x, y = int(pt[0] * width), int(pt[1] * height)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), 1, (80, 80, 80), -1)
                
    # 2. Vẽ thân (Pose - Đường màu xanh lá)
    if not np.all(pose == 0):
        pose_conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
        for p_idx1, p_idx2 in pose_conns:
            pt1, pt2 = pose[p_idx1], pose[p_idx2]
            x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
            x2, y2 = int(pt2[0] * width), int(pt2[1] * height)
            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    # Các kết nối xương tay
    hand_conns = [(0, 1), (1, 2), (2, 3), (3, 4), 
                  (5, 6), (6, 7), (7, 8), 
                  (9, 10), (10, 11), (11, 12), 
                  (13, 14), (14, 15), (15, 16), 
                  (17, 18), (18, 19), (19, 20), 
                  (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
                  
    # 3. Vẽ tay trái (Left hand - Đường màu tím hồng)
    if not np.all(lh == 0):
        for h_idx1, h_idx2 in hand_conns:
            pt1, pt2 = lh[h_idx1], lh[h_idx2]
            x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
            x2, y2 = int(pt2[0] * width), int(pt2[1] * height)
            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                cv2.line(canvas, (x1, y1), (x2, y2), (180, 105, 255), 2)
        for pt in lh:
            x, y = int(pt[0] * width), int(pt[1] * height)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
                
    # 4. Vẽ tay phải (Right hand - Đường màu hồng cánh sen)
    if not np.all(rh == 0):
        for h_idx1, h_idx2 in hand_conns:
            pt1, pt2 = rh[h_idx1], rh[h_idx2]
            x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
            x2, y2 = int(pt2[0] * width), int(pt2[1] * height)
            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                cv2.line(canvas, (x1, y1), (x2, y2), (203, 192, 255), 2)
        for pt in rh:
            x, y = int(pt[0] * width), int(pt[1] * height)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
                
    return canvas

# ---------------------------------------------------------
# HÀM VẼ GIAO DIỆN HÌNH ẢNH (OPENCV HUD LAYOUTS)
# ---------------------------------------------------------
def draw_hud(image, title, status_text="", status_dot_color=None, progress=None, outline_color=None, footer_text=None):
    """Vẽ giao diện HUD hiện đại đè lên frame hình ảnh OpenCV."""
    h, w, _ = image.shape
    
    # 1. Vẽ viền cảnh báo quanh màn hình
    if outline_color:
        cv2.rectangle(image, (0, 0), (w, h), outline_color, 4)
        
    # 2. Vẽ dải Header phía trên (màu đen bán trong suốt)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    # Đường line neon phía dưới Header
    cv2.line(image, (0, 55), (w, 55), (0, 242, 254), 1)
    
    # Tiêu đề chính bên trái
    cv2.putText(image, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Đèn trạng thái bên phải
    if status_dot_color:
        cv2.circle(image, (w - 130, 27), 6, status_dot_color, -1)
        cv2.putText(image, status_text, (w - 115, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    # 3. Vẽ dải Footer phía dưới nếu có hướng dẫn phím bấm
    if footer_text:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h - 55), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # Đường line neon phía trên Footer
        cv2.line(image, (0, h - 55), (w, h - 55), (0, 242, 254), 1)
        cv2.putText(image, footer_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    # 4. Vẽ thanh tiến trình thu thập (Progress Bar) dưới đáy màn hình
    if progress is not None:
        bar_width = int(progress * w)
        cv2.rectangle(image, (0, h - 6), (bar_width, h), (0, 0, 255), -1)

def draw_countdown_overlay(image, number):
    """Vẽ vòng tròn và đếm ngược số giữa màn hình."""
    h, w, _ = image.shape
    center_x, center_y = w // 2, h // 2
    
    overlay = image.copy()
    cv2.circle(overlay, (center_x, center_y), 70, (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    cv2.circle(image, (center_x, center_y), 70, (0, 255, 255), 3, cv2.LINE_AA)
    
    text = str(number)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 6, cv2.LINE_AA)

# ---------------------------------------------------------
# HÀM KHỞI TẠO THƯ MỤC DATASET
# ---------------------------------------------------------
def create_dataset_folders():
    """Tạo sẵn cấu trúc thư mục lưu trữ dataset nếu chưa có."""
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
    print(f"[*] Cấu trúc thư mục dataset đã sẵn sàng tại: {DATA_PATH}")

# ---------------------------------------------------------
# HÀM PHÁT LẠI DỮ LIỆU ĐÃ LƯU
# ---------------------------------------------------------
def play_sequence(action, sequence_idx):
    """Đọc dữ liệu keypoint (.npy) đã lưu và phát lại dưới dạng khung xương skeleton trên OpenCV."""
    seq_path = os.path.join(DATA_PATH, action, str(sequence_idx))
    if not os.path.exists(seq_path):
        print(f"[❌] LỖI: Không tìm thấy thư mục sequence: {seq_path}")
        return
        
    files = sorted([f for f in os.listdir(seq_path) if f.endswith(".npy")], key=lambda x: int(os.path.splitext(x)[0]))
    if len(files) == 0:
        print(f"[❌] LỖI: Không tìm thấy dữ liệu .npy hoàn chỉnh trong thư mục sequence này!")
        return
        
    print(f"[*] Bắt đầu phát lại sequence #{sequence_idx + 1} của cử chỉ '{action.upper()}'...")
    print("[*] Nhấn phím 'Q' trên màn hình hiển thị để thoát xem lại.")
    
    keypoints_cache = []
    for f in files:
        kp = np.load(os.path.join(seq_path, f))
        keypoints_cache.append(kp)
        
    play_active = True
    while play_active:
        for f_idx, kp in enumerate(keypoints_cache):
            canvas = draw_skeleton_on_canvas(kp)
            
            title_text = f'PLAYBACK SKELETON: "{action.upper()}" (Seq #{sequence_idx + 1})'
            footer_text = f"Frame: {f_idx + 1}/{len(files)}  |  Press 'Q' to close playback window"
            
            draw_hud(
                canvas,
                title=title_text,
                status_text="PLAYBACK",
                status_dot_color=(0, 255, 0),
                footer_text=footer_text
            )
            
            cv2.imshow('Data Collection Feed', canvas)
            key = cv2.waitKey(60) & 0xFF
            if key == ord('q') or key == ord('Q'):
                play_active = False
                break
                
    cv2.destroyAllWindows()
    print("[*] Đã đóng cửa sổ xem lại.")

# ---------------------------------------------------------
# VÒNG LẶP CHÍNH THU THẬP DỮ LIỆU
# ---------------------------------------------------------
def run_data_collection(target_action=None, target_sequence=None):
    """Vòng lặp chính quay webcam và thu thập dữ liệu với HUD overlays."""
    create_dataset_folders()
    
    # Khởi động webcam (Tối ưu hóa DirectShow trên Windows)
    print("[*] Đang khởi động camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("[❌] LỖI: Không thể mở Webcam!")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Tối ưu hóa Buffer: Chỉ lưu duy nhất 1 frame mới nhất để triệt tiêu lag tích lũy
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    actions_to_run = [target_action] if target_action else ACTIONS
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions_to_run:
            
            if target_sequence is not None:
                sequences_to_run = [target_sequence]
            else:
                sequences_to_run = list(range(NO_SEQUENCES))
                
            for sequence in sequences_to_run:
                while True:
                    raw_frames_cache = []  # Lưu frame gốc quay siêu mượt
                    frames_cache = []      # Lưu frame kèm landmark để hiển thị replay
                    keypoints_cache = []   # Lưu keypoints trích xuất
                    
                    # =========================================================
                    # GIAI ĐOẠN 1: CHUẨN BỊ (ĐẾM NGƯỢC 3 GIÂY - KHÔNG CHẠY MEDIAPIPE)
                    # =========================================================
                    for countdown in range(3, 0, -1):
                        for _ in range(25): # ~1 giây
                            start_time = time.time()
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            image_mirrored = cv2.flip(frame, 1)
                            title_text = f'PREPARING GESTURE: "{action.upper()}" (Seq #{sequence + 1})'
                            
                            draw_hud(
                                image_mirrored, 
                                title=title_text, 
                                status_text="READY", 
                                status_dot_color=(0, 255, 255)
                            )
                            draw_countdown_overlay(image_mirrored, countdown)
                            
                            cv2.imshow('Data Collection Feed', image_mirrored)
                            
                            # Đo thời gian trễ thực tế để ngủ bù chính xác
                            elapsed = time.time() - start_time
                            sleep_time = max(1, int((0.04 - elapsed) * 1000))
                            cv2.waitKey(sleep_time)
                            
                    # =========================================================
                    # GIAI ĐOẠN 2: THU THẬP DỮ LIỆU (RECORDING - KHÔNG CHẠY MEDIAPIPE)
                    # =========================================================
                    print(f"[*] Đang quay Sequence #{sequence + 1} cho từ: {action}")
                    
                    for frame_num in range(SEQUENCE_LENGTH):
                        start_time = time.time()
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        raw_frames_cache.append(frame)
                        image_mirrored = cv2.flip(frame, 1)
                        
                        # Vẽ HUD màu đỏ REC siêu mượt (ko có xương tay)
                        title_text = f'RECORDING: "{action.upper()}" | Sequence #{sequence + 1}'
                        progress_val = (frame_num + 1) / SEQUENCE_LENGTH
                        
                        draw_hud(
                            image_mirrored,
                            title=title_text,
                            status_text=f"REC {frame_num+1}/{SEQUENCE_LENGTH}",
                            status_dot_color=(0, 0, 255),
                            progress=progress_val,
                            outline_color=(0, 0, 255)
                        )
                        
                        cv2.imshow('Data Collection Feed', image_mirrored)
                        
                        # Ngủ bù chính xác để giữ đúng 25 FPS mượt mà
                        elapsed = time.time() - start_time
                        sleep_time = max(1, int((0.04 - elapsed) * 1000))
                        cv2.waitKey(sleep_time)
                        
                    # =========================================================
                    # GIAI ĐOẠN TRUNG GIAN: CHẠY MEDIAPIPE TRÊN TOÀN BỘ CACHE (BATCH)
                    # =========================================================
                    print("[*] Đang phân tích cử chỉ bằng MediaPipe...")
                    # Tạo hiệu ứng thông báo trên màn hình OpenCV
                    loading_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        loading_frame, "PROCESSING LANDMARKS...", (140, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        loading_frame, "Please wait, extracting features...", (150, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
                    )
                    cv2.imshow('Data Collection Feed', loading_frame)
                    cv2.waitKey(1)
                    
                    for frame in raw_frames_cache:
                        # Chạy nhận diện xương
                        image, results = mediapipe_detection(frame, holistic)
                        kp = extract_keypoints(results)
                        keypoints_cache.append(kp)
                        
                        # Vẽ khung xương phục vụ Replay
                        draw_styled_landmarks(image, results)
                        image_mirrored = cv2.flip(image, 1)
                        frames_cache.append(image_mirrored)
                        
                    # =========================================================
                    # GIAI ĐOẠN 3: PHÁT LẠI VÀ CHỜ QUYẾT ĐỊNH (REPLAY & REVIEW)
                    # =========================================================
                    print("[*] Phân tích xong. Đang phát lại (Replay)...")
                    decision = None
                    
                    while decision is None:
                        for cached_frame in frames_cache:
                            frame_to_show = cached_frame.copy()
                            
                            title_text = f'REVIEW PLAYBACK: "{action.upper()}" (Seq #{sequence + 1})'
                            footer_guide = "Press:  [Y] to SAVE / OVERWRITE  |  [R] to REDO / CANCEL  |  [Q] to QUIT"
                            
                            draw_hud(
                                frame_to_show,
                                title=title_text,
                                status_text="REPLAY",
                                status_dot_color=(254, 242, 0),
                                outline_color=(254, 242, 0),
                                footer_text=footer_guide
                            )
                            
                            cv2.imshow('Data Collection Feed', frame_to_show)
                            
                            key = cv2.waitKey(50) & 0xFF
                            if key == ord('y') or key == ord('Y'):
                                decision = 'save'
                                break
                            elif key == ord('r') or key == ord('R'):
                                decision = 'redo'
                                break
                            elif key == ord('q') or key == ord('Q'):
                                decision = 'quit'
                                break
                                
                    # =========================================================
                    # GIAI ĐOẠN 4: THU THỰC QUYẾT ĐỊNH
                    # =========================================================
                    if decision == 'save':
                        save_path = os.path.join(DATA_PATH, action, str(sequence))
                        os.makedirs(save_path, exist_ok=True)
                        for idx, kp in enumerate(keypoints_cache):
                            npy_path = os.path.join(save_path, f"{idx}.npy")
                            np.save(npy_path, kp)
                        print(f"[+] Đã ghi đè/lưu thành công Sequence #{sequence + 1}")
                        break
                        
                    elif decision == 'redo':
                        print(f"[-] HỦY bỏ lượt quay vừa rồi. Chuẩn bị quay lại Sequence #{sequence + 1}...")
                        
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
    parser.add_argument("--action", type=str, default=None, help="Tên cử chỉ muốn quay/xem lại (ví dụ: hello)")
    parser.add_argument("--sequence", type=int, default=None, help="Số thứ tự sequence muốn quay/xem lại (1-indexed, ví dụ: 5)")
    parser.add_argument("--play", action="store_true", help="Chạy chế độ phát lại (xem thử) cử chỉ đã lưu của sequence này.")
    
    args = parser.parse_args()
    
    if args.action and args.action not in ACTIONS:
        print(f"[!] Lỗi: Hành động '{args.action}' không nằm trong cấu hình ACTIONS của config.py")
    elif (args.action is not None and args.sequence is None) or (args.action is None and args.sequence is not None):
        print("[!] Lỗi: Bạn cần truyền đồng thời cả --action và --sequence để chạy chỉ định.")
    else:
        seq_idx = args.sequence - 1 if args.sequence is not None else None
        if seq_idx is not None and (seq_idx < 0 or seq_idx >= NO_SEQUENCES):
            print(f"[!] Lỗi: Sequence chỉ được nằm trong khoảng từ 1 đến {NO_SEQUENCES}")
        else:
            if args.play:
                play_sequence(args.action, seq_idx)
            else:
                run_data_collection(target_action=args.action, target_sequence=seq_idx)
