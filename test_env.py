import sys
import time

print("=" * 50)
print("🔍 BẮT ĐẦU KIỂM TRA MÔI TRƯỜNG DỰ ÁN AI SIGN LANGUAGE")
print("=" * 50)

# 1. Kiểm tra phiên bản Python
print(f"🐍 Phiên bản Python hiện tại: {sys.version}")
major, minor = sys.version_info[:2]
if (major, minor) != (3, 11) and (major, minor) != (3, 10):
    print(f"⚠️ Cảnh báo: Bạn đang dùng Python {major}.{minor}. Khuyên dùng Python 3.10 hoặc 3.11 để tránh lỗi.")
else:
    print(f"✅ Phiên bản Python đạt chuẩn: {major}.{minor}")

# 2. Kiểm tra các thư viện
packages = [
    ("streamlit", "Streamlit (Giao diện)"),
    ("cv2", "OpenCV (Xử lý ảnh/camera)"),
    ("mediapipe", "MediaPipe (Trích xuất xương tay)"),
    ("numpy", "NumPy (Tính toán ma trận)"),
    ("pyttsx3", "Pyttsx3 (Phát âm giọng nói)"),
    ("pandas", "Pandas (Quản lý dữ liệu)")
]

all_ok = True
for import_name, display_name in packages:
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "Không rõ phiên bản")
        print(f"✅ Nhập thư viện thành công: {display_name} (Phiên bản: {version})")
    except ImportError as e:
        print(f"❌ Lỗi: Không thể nhập thư viện '{display_name}' ({import_name}).")
        print(f"   Chi tiết lỗi: {e}")
        all_ok = False

if all_ok:
    print("-" * 50)
    print("🎉 KẾT QUẢ: MÔI TRƯỜNG ĐÃ ĐƯỢC THIẾT LẬP HOÀN HẢO! 🎉")
    print("👉 Bạn đã sẵn sàng chạy lệnh: streamlit run app.py")
    print("-" * 50)
else:
    print("-" * 50)
    print("❌ KẾT QUẢ: Môi trường chưa hoàn thiện. Vui lòng kiểm tra lại các bước cài đặt.")
    print("-" * 50)
