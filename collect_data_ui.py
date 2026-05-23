import os
# Tắt cảnh báo log không cần thiết của TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import pandas as pd

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="AI Sign Language Data Collector",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import các thông số cấu hình và module xử lý MediaPipe
from core.config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, DATA_PATH
from core.mediapipe_detector import (
    mp_holistic, 
    mediapipe_detection, 
    draw_styled_landmarks, 
    extract_keypoints
)

# Import các hàm đồng bộ từ sync_dataset
from sync_dataset import export_dataset, smart_merge_dataset

# ---------------------------------------------------------
# THIẾT KẾ PHONG CÁCH GIAO DIỆN (CUSTOM CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Tổng quan nền tối hiện đại */
    .stApp {
        background-color: #0e1117;
        color: #f8f9fa;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Trang trí tiêu đề chính */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ff007f, #7f00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

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
# KHỞI TẠO SESSION STATE
# ---------------------------------------------------------
if 'collector_replay_active' not in st.session_state:
    st.session_state.collector_replay_active = False
if 'collector_frames' not in st.session_state:
    st.session_state.collector_frames = []
if 'collector_keypoints' not in st.session_state:
    st.session_state.collector_keypoints = []

# Khởi tạo index sequence mặc định cho từng action
for action in ACTIONS:
    state_key = f"selected_seq_{action}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 1

# Helper check trạng thái sequence trên bộ nhớ
def check_sequence_status(action, seq_idx):
    seq_path = os.path.join(DATA_PATH, action, str(seq_idx))
    if not os.path.exists(seq_path):
        return "missing"
    files = [f for f in os.listdir(seq_path) if f.endswith(".npy")]
    if len(files) == 0:
        return "missing"
    elif len(files) < SEQUENCE_LENGTH:
        return "incomplete"
    else:
        return "completed"

# Tự động tạo thư mục rỗng nếu chưa có
for action in ACTIONS:
    for s in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(s)), exist_ok=True)

# ---------------------------------------------------------
# SIDEBAR - ĐIỀU KHIỂN & CẤU HÌNH THU THẬP
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Cấu hình ghi hình")
camera_index = st.sidebar.number_input("Chọn cổng Camera (Webcam)", min_value=0, max_value=5, value=0, step=1)
countdown_sec = st.sidebar.slider("Thời gian chuẩn bị (giây)", min_value=1, max_value=5, value=3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎬 Lựa chọn cử chỉ")
selected_action = st.sidebar.selectbox("Chọn cử chỉ cần thu thập", ACTIONS)

# Lấy sequence đang chọn từ Session State
selected_seq = st.session_state[f"selected_seq_{selected_action}"]
st.sidebar.info(f"📍 Sequence đang chọn: **#{selected_seq}**")
st.sidebar.caption("👉 Bạn có thể nhấp chọn trực tiếp Sequence khác trên lưới sơ đồ bên phải.")

# ---------------------------------------------------------
# GIAO DIỆN CHÍNH (MAIN COLLECTOR APP)
# ---------------------------------------------------------
st.markdown("<div class='main-title'>📹 AI Sign Language Data Collector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Giao diện Web UI chuyên dụng để ghi hình và đóng gói Dataset cử chỉ động</div>", unsafe_allow_html=True)

# Thống kê tiến trình
st.markdown("### 📊 Tiến độ Dataset hiện tại:")
stats_cols = st.columns(len(ACTIONS))
for idx, act in enumerate(ACTIONS):
    with stats_cols[idx]:
        comp_count = 0
        for s in range(NO_SEQUENCES):
            if check_sequence_status(act, s) == "completed":
                comp_count += 1
        percent = (comp_count / NO_SEQUENCES) * 100
        st.metric(label=f"🎬 {act.upper()}", value=f"{comp_count}/{NO_SEQUENCES}", delta=f"{percent:.0f}%")
        st.progress(percent / 100.0)

st.markdown("---")

col_cam, col_grid = st.columns([1.2, 1])

# Cột hiển thị hình ảnh Camera
with col_cam:
    st.markdown(f"### 🎥 Giao diện Camera: **{selected_action.upper()}** (Seq #{selected_seq})")
    cam_placeholder = st.empty()

    # PHẦN 1: CHẾ ĐỘ XEM LẠI CHUỒI VỪA QUAY (REPLAY MODE)
    if st.session_state.collector_replay_active:
        st.info("💡 Đang phát lại (Replay) sequence vừa quay để bạn kiểm tra:")
        
        col_save, col_redo = st.columns(2)
        with col_save:
            if st.button("✅ LƯU DỮ LIỆU", use_container_width=True, type="primary"):
                # Ghi dữ liệu thực tế xuống thư mục dataset
                save_path = os.path.join(DATA_PATH, selected_action, str(selected_seq - 1))
                os.makedirs(save_path, exist_ok=True)
                for f_idx, kp in enumerate(st.session_state.collector_keypoints):
                    np.save(os.path.join(save_path, f"{f_idx}.npy"), kp)
                
                st.toast(f"🎉 Đã lưu thành công Sequence #{selected_seq}!")
                
                # Tự động nhảy sang sequence tiếp theo nếu chưa quay hết
                if selected_seq < NO_SEQUENCES:
                    st.session_state[f"selected_seq_{selected_action}"] = selected_seq + 1
                
                # Reset trạng thái cache
                st.session_state.collector_replay_active = False
                st.session_state.collector_frames = []
                st.session_state.collector_keypoints = []
                st.rerun()
                
        with col_redo:
            if st.button("🔄 HỦY & QUAY LẠI", use_container_width=True):
                st.session_state.collector_replay_active = False
                st.session_state.collector_frames = []
                st.session_state.collector_keypoints = []
                st.toast("Đã hủy lượt quay vừa rồi.")
                st.rerun()

        # Vòng lặp phát lại
        if st.session_state.collector_frames:
            frame_count = len(st.session_state.collector_frames)
            play_idx = 0
            while st.session_state.collector_replay_active:
                f = st.session_state.collector_frames[play_idx]
                cam_placeholder.image(f, channels="BGR", width="stretch")
                play_idx = (play_idx + 1) % frame_count
                time.sleep(0.04) # Giả lập ~25 FPS

    # PHẦN 2: CHƯA QUAY, HIỂN THỊ MÀN HÌNH CHỜ & NÚT GHI HÌNH
    else:
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            dummy_img, "Camera san sang. Nhan nut ben duoi de bat dau ghi hinh.", (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        cam_placeholder.image(dummy_img, channels="BGR", width="stretch")
        
        if st.button("🚀 BẮT ĐẦU GHI HÌNH SEQUENCE NÀY", use_container_width=True, type="primary"):
            # Mở camera với DirectShow trên Windows
            with st.spinner("🔌 Đang kết nối Webcam..."):
                cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(camera_index))
            
            if not cap.isOpened():
                st.error("🚨 Không thể kết nối với Webcam! Hãy thử đổi cổng Camera ở sidebar hoặc tắt ứng dụng camera khác.")
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Chống lag đọng buffer
                
                # --- A. Đếm ngược (Countdown - Siêu mượt) ---
                for count in range(countdown_sec, 0, -1):
                    for _ in range(25): # ~1 giây
                        start_time = time.time()
                        ret, frame = cap.read()
                        if not ret:
                            break
                        image_mirrored = cv2.flip(frame, 1)
                        
                        # Vẽ đếm ngược
                        cv2.putText(
                            image_mirrored, f"CHUAN BI: {count}", (150, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5, cv2.LINE_AA
                        )
                        cv2.putText(
                            image_mirrored, f'Cu chi: "{selected_action.upper()}" | Seq #{selected_seq}', (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                        )
                        cam_placeholder.image(image_mirrored, channels="BGR", width="stretch")
                        
                        elapsed = time.time() - start_time
                        sleep_time = max(0.001, 0.04 - elapsed)
                        time.sleep(sleep_time)
                
                # --- B. Ghi hình dữ liệu (30 frames - Siêu mượt) ---
                rec_raw_frames = []
                for f_num in range(SEQUENCE_LENGTH):
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    rec_raw_frames.append(frame)
                    image_mirrored = cv2.flip(frame, 1)
                    
                    # Vẽ HUD ghi hình
                    cv2.putText(
                        image_mirrored, f"GHI HINH: {f_num + 1}/{SEQUENCE_LENGTH}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        image_mirrored, f'Cu chi: "{selected_action.upper()}" | Seq #{selected_seq}', (15, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    
                    progress_w = int((f_num + 1) / SEQUENCE_LENGTH * 640)
                    cv2.rectangle(image_mirrored, (0, 470), (progress_w, 480), (0, 0, 255), -1)
                    
                    cam_placeholder.image(image_mirrored, channels="BGR", width="stretch")
                    
                    elapsed = time.time() - start_time
                    sleep_time = max(0.001, 0.04 - elapsed)
                    time.sleep(sleep_time)
                    
                cap.release()
                
                # --- C. Phân tích MediaPipe ---
                rec_frames = []
                rec_keypoints = []
                
                loading_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    loading_img, "DANG PHAN TICH DAC TRUNG...", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
                )
                cv2.putText(
                    loading_img, "Trich xuat landmarks bang MediaPipe, vui long cho...", (90, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
                )
                cam_placeholder.image(loading_img, channels="BGR", width="stretch")
                
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    for r_frame in rec_raw_frames:
                        image, results = mediapipe_detection(r_frame, holistic)
                        kp = extract_keypoints(results)
                        rec_keypoints.append(kp)
                        
                        draw_styled_landmarks(image, results)
                        image_mirrored = cv2.flip(image, 1)
                        rec_frames.append(image_mirrored)
                        
                st.session_state.collector_frames = rec_frames
                st.session_state.collector_keypoints = rec_keypoints
                st.session_state.collector_replay_active = True
                st.rerun()

# Cột hiển thị lưới Grid 30 sequences của cử chỉ hiện tại
with col_grid:
    st.markdown(f"### 📅 Sơ đồ 30 Sequences của: **{selected_action.upper()}**")
    st.caption("👉 Nhấp trực tiếp vào ô số dưới đây để chọn Sequence muốn ghi hình / xem lại:")
    
    # Tạo lưới hiển thị 10 cột x 3 dòng dưới dạng các Button
    grid_cols = st.columns(10)
    for seq in range(NO_SEQUENCES):
        status = check_sequence_status(selected_action, seq)
        col_idx = seq % 10
        
        # Biểu tượng trạng thái
        if status == "completed":
            label = f"✅\n#{seq+1}"
        elif status == "incomplete":
            label = f"⚠️\n#{seq+1}"
        else:
            label = f"📁\n#{seq+1}"
            
        # Highlight nút đang chọn bằng kiểu "primary" (màu chính)
        is_current = (selected_seq == seq + 1)
        btn_type = "primary" if is_current else "secondary"
        
        with grid_cols[col_idx]:
            if st.button(label, key=f"btn_seq_{seq}", use_container_width=True, type=btn_type):
                st.session_state[f"selected_seq_{selected_action}"] = seq + 1
                st.rerun()
                
    st.markdown("""
    **Chú thích trạng thái:**
    * **✅ Xanh lá (Có Check)**: Đã có đủ 30 frames (Hợp lệ)
    * **⚠️ Cam (Có Cảnh báo)**: Thiếu frame / Đang quay dở
    * **📁 Xám (Có Folder)**: Chưa có dữ liệu
    * **Nút được tô màu nổi bật**: Sequence hiện đang được chọn để ghi đè hoặc xem lại.
    """, unsafe_allow_html=True)
    
    # --- PHẦN XEM LẠI CỬ CHỈ ĐÃ LƯU ---
    st.markdown("---")
    st.markdown("### 👁️ Xem lại cử chỉ đã lưu (Skeleton Replay)")
    if check_sequence_status(selected_action, selected_seq - 1) == "completed":
        if st.button("👁️ PHÁT LẠI KHUNG XƯƠNG CỦA SEQUENCE NÀY", use_container_width=True):
            seq_path = os.path.join(DATA_PATH, selected_action, str(selected_seq - 1))
            files = sorted([f for f in os.listdir(seq_path) if f.endswith(".npy")], key=lambda x: int(os.path.splitext(x)[0]))
            
            kps = [np.load(os.path.join(seq_path, f)) for f in files]
            
            st.info("📺 Đang chiếu lại chuyển động khung xương (3 vòng lặp):")
            play_placeholder = st.empty()
            
            for loop in range(3):
                for f_idx, kp in enumerate(kps):
                    canvas = draw_skeleton_on_canvas(kp)
                    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    play_placeholder.image(
                        canvas_rgb, 
                        width="stretch", 
                        caption=f"Vòng lặp {loop+1}/3 - Frame {f_idx+1}/{len(files)}"
                    )
                    time.sleep(0.04)
            st.success("🎉 Đã phát xong!")
    else:
        st.warning("⚠️ Sequence đang chọn chưa được lưu hoặc chưa đủ 30 frames để xem lại.")

    st.markdown("---")
    
    # Nút chạy quét kiểm tra lỗi chi tiết dataset
    if st.button("🔍 Chạy validate dataset chi tiết", use_container_width=True):
        st.toast("Đang kiểm tra toàn bộ file trong dataset...")
        os.system("python validate_dataset.py")
        st.success("Đã hoàn tất kiểm tra! Hãy xem chi tiết kết quả thống kê lỗi ở màn hình Terminal/Command Prompt.")

# --- BẢNG ĐIỀU KHIỂN ĐỒNG BỘ & CHIA SẺ DATASET GIỮA CÁC THÀNH VIÊN ---
st.markdown("---")
with st.expander("📦 ĐỒNG BỘ & CHIA SẺ DATASET GIỮA C CÁC THÀNH VIÊN"):
    col_exp, col_imp = st.columns(2)
    
    with col_exp:
        st.markdown("#### 1. Đóng gói gửi cho Teammate")
        st.write("Nén toàn bộ thư mục `dataset/` hiện tại của bạn thành một file ZIP sạch để gửi cho bạn bè qua Zalo/Discord.")
        if st.button("📦 TẠO FILE ZIP DATASET XUẤT KHẨU", use_container_width=True):
            with st.spinner("Đang đóng gói..."):
                zip_path = export_dataset()
            if zip_path and os.path.exists(zip_path):
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ TẢI XUỐNG FILE DATASET.ZIP",
                        data=f,
                        file_name=zip_path,
                        mime="application/zip",
                        use_container_width=True
                    )
                st.success("Đã nén thành công! Nhấn nút Tải xuống để lưu về máy tính của bạn.")
                
    with col_imp:
        st.markdown("#### 2. Gộp Dataset từ Teammate gửi đến")
        st.write("Tải lên file ZIP dataset của teammate để tự động gộp. Số thứ tự sequence bị trùng sẽ được tăng tiến tự động.")
        uploaded_zip = st.file_uploader("Kéo thả file ZIP dataset vào đây", type=["zip"])
        if uploaded_zip is not None:
            if st.button("🧬 TIẾN HÀNH GỘP THÔNG MINH (SMART MERGE)", use_container_width=True, type="primary"):
                # Lưu file zip tạm thời để xử lý
                temp_zip_name = "temp_uploaded_dataset.zip"
                with open(temp_zip_name, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                
                with st.spinner("Đang phân tích và gộp dữ liệu..."):
                    success = smart_merge_dataset(temp_zip_name)
                
                # Dọn dẹp file tạm
                if os.path.exists(temp_zip_name):
                    os.remove(temp_zip_name)
                    
                if success:
                    st.toast("🎉 Đã gộp và đồng bộ thành công dữ liệu từ Teammate!")
                    st.success("Đã hoàn tất gộp dữ liệu! Màn hình sẽ tự động cập nhật lại các chỉ số tiến trình sau 2 giây.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Gộp dữ liệu thất bại. Hãy chắc chắn file tải lên đúng định dạng ZIP dataset của dự án.")
