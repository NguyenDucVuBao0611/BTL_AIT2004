import os
import time
import threading
import cv2
import numpy as np
import streamlit as st
import pyttsx3
import mediapipe as mp

# Thiết lập cấu hình trang Streamlit đầu tiên (bắt buộc phải đặt ở đầu file)
st.set_page_config(
    page_title="AI Sign Language Translation Dashboard",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import các module cốt lõi của dự án
from core.config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH
from core.mediapipe_detector import (
    mp_holistic, 
    mediapipe_detection, 
    draw_styled_landmarks, 
    extract_keypoints
)
from core.buffer import SequenceBuffer
from core.heuristic_classifier import HeuristicClassifier

# ---------------------------------------------------------
# HÀM TRANSLATION & TEXT-TO-SPEECH (TTS) KHÔNG BLOCK STREAM
# ---------------------------------------------------------
def speak_out_loud(text):
    """Phát âm từ/câu bằng giọng nói trong luồng chạy ngầm (non-blocking thread)."""
    def _speak_thread():
        try:
            # Khởi tạo engine TTS cục bộ trong thread để tránh lỗi luồng
            engine = pyttsx3.init()
            # Cấu hình giọng nói nếu muốn
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id) # Giọng nam/nữ mặc định
            engine.setProperty('rate', 150) # Tốc độ nói vừa phải
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            # Bỏ qua nếu máy chủ không hỗ trợ audio driver
            pass

    threading.Thread(target=_speak_thread, daemon=True).start()

# Dịch nhãn sang tiếng Việt để hiển thị thân thiện
TRANSLATION_MAP = {
    "idle": "Chờ hành động...",
    "hello": "Xin chào 👋",
    "thanks": "Cảm ơn 💖",
    "iloveyou": "Tôi yêu bạn 🤟",
    "yes": "Đồng ý / Có ✅",
    "no": "Không / Từ chối ❌"
}

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
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    
    /* Thiết kế thẻ Card cho Cử chỉ hiện tại */
    .gesture-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .gesture-label-header {
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .gesture-name {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
        margin-top: 10px;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
    }
    
    /* Thiết kế hộp ghép câu */
    .sentence-box {
        background: rgba(79, 172, 254, 0.1);
        border: 1px dashed #4facfe;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    /* Layout Sidebar */
    .sidebar .sidebar-content {
        background-color: #1a1c23;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INITIALIZE SESSION STATE (LƯU TRỮ TRẠNG THÁI APP)
# ---------------------------------------------------------
if 'sentence_history' not in st.session_state:
    st.session_state.sentence_history = []
if 'last_predicted_action' not in st.session_state:
    st.session_state.last_predicted_action = "idle"
if 'last_action_time' not in st.session_state:
    st.session_state.last_action_time = time.time()
if 'detection_streak' not in st.session_state:
    st.session_state.detection_streak = 0

# Khởi tạo các classifier và buffer
heuristic_clf = HeuristicClassifier()
seq_buffer = SequenceBuffer(max_length=SEQUENCE_LENGTH)

# ---------------------------------------------------------
# SIDEBAR - BẢNG ĐIỀU KHIỂN & CẤU HÌNH
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Cấu hình hệ thống")

# 1. Trình lựa chọn camera
camera_index = st.sidebar.number_input("Chọn chỉ số Camera (Webcam)", min_value=0, max_value=5, value=0, step=1)

# 2. Ngưỡng tin cậy (Confidence Threshold)
conf_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy nhận diện",
    min_value=0.5,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Độ chính xác tối thiểu để chấp nhận cử chỉ và hiển thị kết quả."
)

# 3. Số frame duy trì để ghi nhận từ mới (Streak)
streak_required = st.sidebar.slider(
    "Số frame ổn định liên tục",
    min_value=3,
    max_value=15,
    value=6,
    step=1,
    help="Số lượng khung hình liên tục nhận diện cùng một cử chỉ để thêm từ đó vào câu (chống giật/nhảy nhãn)."
)

# 4. Kiểm tra sự tồn tại của mô hình AI (.h5)
model_path = os.path.join("models", "sign_lstm.h5")
model_exists = os.path.exists(model_path)

if model_exists:
    st.sidebar.success("🤖 Tìm thấy AI Model: `sign_lstm.h5`")
    running_mode = st.sidebar.selectbox("Chế độ xử lý", ["Quy tắc ngón tay (Heuristics)", "Mô hình học sâu (Keras/LSTM)"])
else:
    st.sidebar.warning("⚠️ Chưa có AI Model (.h5)")
    st.sidebar.info("Hệ thống sẽ chạy ở chế độ **Quy tắc ngón tay (Heuristics)** dựa trên hình học của tay.")
    running_mode = "Quy tắc ngón tay (Heuristics)"

# 5. Các nút tương tác điều khiển câu
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Điều khiển câu ghép")
col_reset, col_speak = st.sidebar.columns(2)
with col_reset:
    if st.button("🔄 Xóa câu", use_container_width=True):
        st.session_state.sentence_history = []
        st.toast("Đã xóa toàn bộ câu ký hiệu!")
with col_speak:
    if st.button("🔊 Đọc to câu", use_container_width=True):
        if st.session_state.sentence_history:
            full_sentence = " ".join(st.session_state.sentence_history)
            st.toast(f"Đang phát âm: '{full_sentence}'")
            speak_out_loud(full_sentence)
        else:
            st.toast("Câu đang rỗng, vui lòng thực hiện ký hiệu trước!")

# ---------------------------------------------------------
# GIAO DIỆN CHÍNH (MAIN DASHBOARD)
# ---------------------------------------------------------
st.markdown("<div class='main-title'>AI Sign Language Translation Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Giao diện kiểm tra cử chỉ thủ ngữ thời gian thực (Tuần 1 Prototype - Người 3)</div>", unsafe_allow_html=True)

# Khởi tạo nút kích hoạt Camera
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
run_cam = st.toggle("👉 Bật Webcam để bắt đầu kiểm tra 👈", value=False)
st.markdown("</div>", unsafe_allow_html=True)

# Thiết lập hai cột chính
col_feed, col_metrics = st.columns([1.3, 1])

# Cột 1: Webcam Feed
with col_feed:
    st.markdown("### 🎥 Luồng Camera & Bộ lọc xương (MediaPipe)")
    # Khung chứa hình ảnh webcam trực tiếp
    video_placeholder = st.empty()
    if not run_cam:
        # Hiển thị ảnh chờ khi camera tắt
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            dummy_img, "Webcam dang tat. Vui long bat toggle ben tren.", (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        video_placeholder.image(dummy_img, channels="BGR", use_column_width=True)

# Cột 2: Bảng kết quả & Đồ thị
with col_metrics:
    st.markdown("### 📊 Trạng thái nhận diện thực tế")
    
    # 1. Thẻ hiển thị Cử chỉ hiện tại
    gesture_placeholder = st.empty()
    
    # 2. Thanh biểu đồ thanh ngang cho độ tin cậy
    chart_placeholder = st.empty()
    
    # 3. Hộp hiển thị câu ghép được
    st.markdown("##### 💬 Câu đã ghép hoàn chỉnh:")
    sentence_placeholder = st.empty()

# ---------------------------------------------------------
# VÒNG LẶP XỬ LÝ FRAME CAMERA (CHỈ CHẠY KHI BẬT CAMERA)
# ---------------------------------------------------------
if run_cam:
    cap = cv2.VideoCapture(int(camera_index))
    
    # Cấu hình độ phân giải tối ưu để giữ FPS cao (> 20fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_time = time.time()
    
    # Mở mô hình MediaPipe Holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể kết nối với Webcam. Vui lòng kiểm tra lại thiết bị hoặc chỉ số camera ở Sidebar!")
                break
                
            start_proc_time = time.time()
            
            # 1. Nhận diện các điểm landmarks thông qua MediaPipe
            image, results = mediapipe_detection(frame, holistic)
            
            # 2. Vẽ khung xương trang trí với style cá nhân hóa đẹp mắt
            draw_styled_landmarks(image, results)
            
            # 3. Trích xuất vector keypoint
            keypoints = extract_keypoints(results)
            # Thêm keypoints vào bộ đệm trượt
            seq_buffer.append(keypoints)
            
            # 4. Thực hiện suy luận dự đoán cử chỉ
            predicted_action = "idle"
            confidence = 0.0
            probs = [0.0] * len(heuristic_clf.actions)
            
            if running_mode == "Quy tắc ngón tay (Heuristics)" or not model_exists:
                # Chế độ Rule-based (Quy tắc ngón tay)
                predicted_action, confidence, probs = heuristic_clf.predict(results)
            else:
                # Chế độ AI Model (Deep Learning) - Để dành khi Người 2 train xong model .h5
                # if seq_buffer.is_ready():
                #     sequence = seq_buffer.get_sequence()
                #     # Chạy model.predict(np.expand_dims(sequence, axis=0))
                #     pass
                predicted_action, confidence, probs = heuristic_clf.predict(results)
            
            # 5. Xử lý Logic ghép chữ thành câu (Sentence Builder & Smoothing)
            if confidence >= conf_threshold:
                if predicted_action != "idle":
                    # Nếu nhận diện liên tục cùng một hành động
                    if predicted_action == st.session_state.last_predicted_action:
                        st.session_state.detection_streak += 1
                    else:
                        st.session_state.detection_streak = 1
                        st.session_state.last_predicted_action = predicted_action
                    
                    # Nếu đạt đủ số lượng frame ổn định liên tục
                    if st.session_state.detection_streak == streak_required:
                        now = time.time()
                        time_since_last = now - st.session_state.last_action_time
                        
                        # A. THUẬT TOÁN GHI ĐÈ CHUYỂN TIẾP (HELLO -> THANKS)
                        # Nếu nhận diện được THANKS ngay sau HELLO trong vòng < 2 giây, ghi đè HELLO thành THANKS
                        if (predicted_action == "thanks" and 
                            st.session_state.sentence_history and 
                            st.session_state.sentence_history[-1] == "hello" and 
                            time_since_last < 2.0):
                            
                            st.session_state.sentence_history.pop() # Xóa HELLO
                            st.session_state.sentence_history.append("thanks")
                            st.session_state.last_action_time = now
                            speak_out_loud("thanks")
                            st.toast("👋➡️💖 Tự động chuyển đổi HELLO sang THANKS!")
                            
                        # B. THUẬT TOÁN COOLDOWN HẠ TAY (THANKS -> HELLO)
                        # Bỏ qua HELLO nếu người dùng vừa mới làm THANKS trong vòng < 2 giây (đang hạ tay xuống)
                        elif (predicted_action == "hello" and 
                              st.session_state.sentence_history and 
                              st.session_state.sentence_history[-1] == "thanks" and 
                              time_since_last < 2.0):
                            st.session_state.detection_streak = 0 # Reset streak để tránh tích lũy
                            
                        # C. THÊM TỪ BÌNH THƯỜNG
                        else:
                            # Tránh lặp lại cùng một từ vừa thêm
                            if not st.session_state.sentence_history or st.session_state.sentence_history[-1] != predicted_action:
                                st.session_state.sentence_history.append(predicted_action)
                                st.session_state.last_action_time = now
                                speak_out_loud(predicted_action)
                                st.toast(f"Đã thêm từ: {predicted_action.upper()}")
            else:
                # Reset streak nếu độ tin cậy tụt xuống dưới ngưỡng
                st.session_state.detection_streak = 0
            
            # 6. Tính toán FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            latency = (curr_time - start_proc_time) * 1000 # Độ trễ xử lý (ms)
            
            # 7. Cập nhật giao diện hình ảnh camera (Lật gương cho dễ nhìn)
            image_mirrored = cv2.flip(image, 1)
            # Vẽ thông số FPS và Latency lên góc màn hình camera
            cv2.putText(
                image_mirrored, f"FPS: {fps:.1f} | Latency: {latency:.0f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if fps > 18 else (0, 165, 255), 2, cv2.LINE_AA
            )
            video_placeholder.image(image_mirrored, channels="BGR", use_column_width=True)
            
            # 8. Cập nhật các bảng số liệu bên cột phải
            translated_action = TRANSLATION_MAP.get(predicted_action, predicted_action.upper())
            
            # Cập nhật Thẻ cử chỉ hiện tại
            gesture_placeholder.markdown(f"""
            <div class="gesture-box">
                <div class="gesture-label-header">Cử chỉ hiện tại</div>
                <div class="gesture-name">{translated_action}</div>
                <div style="color: #4facfe; margin-top: 5px; font-weight: 500;">
                    Độ tin cậy: {confidence*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Cập nhật biểu đồ phân phối độ tin cậy các hành động
            import pandas as pd
            chart_data = pd.DataFrame({
                "Cử chỉ": [TRANSLATION_MAP.get(a, a.upper()) for a in heuristic_clf.actions],
                "Độ tin cậy": probs
            })
            chart_placeholder.bar_chart(
                data=chart_data,
                x="Cử chỉ",
                y="Độ tin cậy",
                use_container_width=True,
                height=200
            )
            
            # Cập nhật Hộp hiển thị câu ghép được
            if st.session_state.sentence_history:
                full_sentence_vn = " ".join([TRANSLATION_MAP.get(word, word).split(" ")[0] for word in st.session_state.sentence_history])
                sentence_placeholder.markdown(f"<div class='sentence-box'>👉 {full_sentence_vn.upper()}</div>", unsafe_allow_html=True)
            else:
                sentence_placeholder.markdown("<div class='sentence-box' style='color:#8892b0;'>Chưa ghi nhận cử chỉ nào để ghép câu</div>", unsafe_allow_html=True)
                
            # Nghỉ một khoảng cực nhỏ (1ms) để nhường quyền xử lý cho các tiến trình khác của Streamlit
            time.sleep(0.001)
            
        cap.release()
        st.rerun()
