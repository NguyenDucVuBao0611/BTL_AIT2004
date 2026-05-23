import os
import numpy as np

# Cấu hình kiểm tra
EXPECTED_FRAME_COUNT = 30
EXPECTED_SHAPE = (1662,)
DATASET_DIR = os.path.join("datasetHF", "train")


def validate_dataset():
    print("=" * 60)
    print("🔍 BẮT ĐẦU KIỂM TRA CHẤT LƯỢNG DATASET")
    print("=" * 60)
    
    if not os.path.exists(DATASET_DIR):
        print(f"[❌] Không tìm thấy thư mục dữ liệu '{DATASET_DIR}'. Vui lòng tạo thư mục hoặc thu thập dữ liệu trước.")
        return

    actions = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    if not actions:
        print(f"[❌] Thư mục '{DATASET_DIR}' đang trống. Chưa có dữ liệu gesture nào.")
        return

    print(f"[+] Tìm thấy {len(actions)} cử chỉ (gestures): {', '.join(actions)}")
    print("-" * 60)

    total_sequences = 0
    total_valid_sequences = 0
    errors_found = []
    
    report_data = {}

    for action in actions:
        action_path = os.path.join(DATASET_DIR, action)
        sequences = [s for s in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, s))]
        
        valid_seq_count = 0
        total_seq_count = len(sequences)
        
        for seq in sequences:
            seq_path = os.path.join(action_path, seq)
            total_sequences += 1
            seq_is_valid = True
            
            # 1. Kiểm tra số lượng file npy trong sequence
            npy_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
            if len(npy_files) != EXPECTED_FRAME_COUNT:
                errors_found.append(
                    f"⚠️ [{action.upper()}] Seq #{seq}: Thiếu/thừa file. Có {len(npy_files)}/{EXPECTED_FRAME_COUNT} files."
                )
                seq_is_valid = False
            
            # 2. Kiểm tra chi tiết từng file npy
            zero_frames_count = 0
            for frame_idx in range(EXPECTED_FRAME_COUNT):
                file_name = f"{frame_idx}.npy"
                file_path = os.path.join(seq_path, file_name)
                
                if not os.path.exists(file_path):
                    continue
                
                try:
                    data = np.load(file_path)
                    
                    # Kiểm tra shape của dữ liệu
                    if data.shape != EXPECTED_SHAPE:
                        errors_found.append(
                            f"❌ [{action.upper()}] Seq #{seq} Frame {frame_idx}: Sai kích thước. Có {data.shape} thay vì {EXPECTED_SHAPE}."
                        )
                        seq_is_valid = False
                    
                    # Kiểm tra xem dữ liệu có bị zero hoàn toàn (không detect được landmark nào)
                    if np.all(data == 0):
                        zero_frames_count += 1
                        
                except Exception as e:
                    errors_found.append(
                        f"💥 [{action.upper()}] Seq #{seq} Frame {frame_idx}: Lỗi đọc file. Chi tiết: {e}"
                    )
                    seq_is_valid = False
            
            if zero_frames_count > 0:
                # Cảnh báo nếu có quá nhiều frame trống (nhận diện tay bị mất liên tục)
                if zero_frames_count >= (EXPECTED_FRAME_COUNT / 2):
                    errors_found.append(
                        f"⚠️ [{action.upper()}] Seq #{seq}: Quá nhiều frame trống. Có {zero_frames_count}/{EXPECTED_FRAME_COUNT} frame không nhận diện được landmark."
                    )
                    seq_is_valid = False
                    
            if seq_is_valid:
                valid_seq_count += 1
                total_valid_sequences += 1
                
        report_data[action] = {
            "total": total_seq_count,
            "valid": valid_seq_count,
            "failed": total_seq_count - valid_seq_count
        }

    # In báo cáo chi tiết
    print("\n📊 BÁO CÁO THỐNG KÊ CHI TIẾT:")
    print(f"{'Cử chỉ (Gesture)':<20} | {'Tổng Sequence':<15} | {'Hợp lệ':<10} | {'Bị lỗi':<10}")
    print("-" * 65)
    for action, stats in report_data.items():
        print(f"{action:<20} | {stats['total']:<15} | {stats['valid']:<10} | {stats['failed']:<10}")
    
    print("-" * 65)
    print(f"Tổng số Sequence đã quét: {total_sequences}")
    print(f"Tổng số Sequence hợp lệ : {total_valid_sequences} ({total_valid_sequences/total_sequences*100:.1f}%)")
    print(f"Tổng số Sequence lỗi    : {total_sequences - total_valid_sequences}")
    print("=" * 60)

    if errors_found:
        print("\n🚨 DANH SÁCH CHI TIẾT CÁC LỖI PHÁT HIỆN:")
        for err in errors_found[:30]:  # Giới hạn in tối đa 30 lỗi để tránh tràn màn hình
            print(err)
        if len(errors_found) > 30:
            print(f"... và {len(errors_found) - 30} lỗi khác tương tự.")
        print("\n👉 Lời khuyên: Hãy sử dụng chế độ ghi đè targeted trong collect_data.py để quay lại các sequence lỗi.")
    else:
        print("\n🎉 Tuyệt vời! Không phát hiện lỗi nào. Toàn bộ dataset đã sẵn sàng để train model!")
    print("=" * 60)

if __name__ == "__main__":
    validate_dataset()
