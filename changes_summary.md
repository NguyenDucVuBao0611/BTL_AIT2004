# Báo Cáo Thay Đổi (Changes Summary) - Liar's Dice AI

Tài liệu này tổng hợp toàn bộ các thay đổi, cải tiến và sửa lỗi đã được thực hiện trên mã nguồn trò chơi Liar's Dice AI so với phiên bản gốc (nhánh `main` lúc pull về).

---

## 1. Tổng quan các tệp thay đổi

Dưới đây là thống kê nhanh số lượng dòng code thay đổi (thêm/bớt) trên các file:
- **`agents/bayesian_agent.py`**: +263 dòng / -4 dòng (Nâng cấp Multi-Context Bayes & Lưu/Reset hồ sơ thói quen).
- **`agents/cfr_agent.py`**: +122 dòng / -2 dòng (Chuyển tiếp quan sát tới fallback & đồng bộ dashboard).
- **`ui/gui.py`**: +410 dòng / -80 dòng (Căn giữa giao diện 800x720, xoá Lịch sử đấu, xuất trạng thái & tự động học ngầm).
- **`ui/web_dashboard.py`** [NEW]: Thêm mới Web Dashboard giám sát thời gian thực.
- **`results/user_habit_profile.json`** [NEW]: Lưu trữ thói quen người chơi (tự động đồng bộ và reset về 0).
- **`tests/test_bayesian_agent.py`**: +168 dòng (Bổ sung unit test cho Multi-Context và file IO).
- **`tests/test_cfr_agent.py`**: +29 dòng (Bổ sung test chuyển tiếp quan sát).
- **`tests/test_engine.py`, `test_probabilistic_agent.py`, `test_probability.py`**: Cập nhật các assert phù hợp với cơ chế mới.

---

## 2. Chi tiết các thay đổi chính

### A. Web Dashboard Giám sát Real-time (`ui/web_dashboard.py`)
- **Tạo mới Dashboard Web**: Chạy song song trên cổng `8000`, tự động bật ngầm khi chạy game Pygame.
- **Thiết kế Glassmorphism hiện đại**: Sử dụng CSS HSL cao cấp, hiệu ứng chuyển động mượt mà (micro-animations), các bảng phân phối xác suất và thanh đo ngưỡng nghi ngờ trực quan.
- **Giám sát thời gian thực**:
  - Trạng thái vòng đấu, số xúc xắc, cược hiện tại và xúc xắc của AI.
  - Phân phối chiến thuật CFR (CFR Strategy Distribution) khi AI tra bảng.
  - Thanh đo xác suất nói thật của đối thủ ($P_{truth}$) so với ngưỡng nghi ngờ động khi AI ở chế độ Bayes.
  - Bảng thói quen người chơi (Bayes Multi-Context).

### B. Nâng cấp Giao diện Pygame (`ui/gui.py`)
- **Căn chỉnh bố cục đối xứng**:
  - Hạ độ phân giải màn hình từ `1024x720` xuống `800x720`.
  - Loại bỏ hoàn toàn bảng "Lịch sử đấu" dư thừa ở bên phải.
  - Căn giữa toàn bộ xúc xắc (AI và Bạn), bảng cược, các nút bấm steppers (`Số lượng`, `Mặt`) và các nút hành động (`CƯỢC`, `LIAR!`) quanh tâm dọc `x=400`.
- **Tự động học ngầm (Online Adaptive CFR)**:
  - Khi một ván đấu kết thúc, game sẽ tự động chạy 200 vòng self-play huấn luyện CFR bổ sung trong một tiến trình ngầm (threading) và lưu brain weights đè lên file `.gz`. Game không bị đơ/khựng, giúp AI thích ứng dần với cách chơi của người dùng.
- **Xuất trạng thái real-time**:
  - Ghi trạng thái tính toán hiện tại của AI ra file `results/ai_thinking_state.json` sau mỗi nước đi để Web Dashboard hiển thị.

### C. Nâng cấp Bayesian Agent (`agents/bayesian_agent.py`)
- **Suy luận đa ngữ cảnh (Bayes Multi-Context)**:
  - Thay vì chỉ tính xác suất bluff chung, Agent được bổ sung 6 ngữ cảnh cụ thể: *Mặc định (General)*, *Bạn ít xúc xắc (<= 2)*, *Thế trận bạn thắng*, *Thế trận bạn thua*, *Bạn cược số lượng lớn (>= 50% tổng số)*, *Bạn cược xúc xắc mặt [1] (Wild)*.
- **Hệ số bước nhảy cược (Modifier)**:
  - Phân tích lịch sử vòng chơi để tăng/giảm ngưỡng nghi ngờ. Ví dụ: đối thủ hô tăng cược đột biến ($\ge 2$ xúc xắc) sẽ tăng độ khả nghi lên $1.3\times$, đối thủ cược tối thiểu thì giảm độ khả nghi về $0.8\times$.
- **Lưu trữ & Reset hồ sơ thói quen**:
  - Thêm phương thức `load_profile()` và `save_profile()` tự động ghi/đọc hồ sơ thói quen của người chơi ra file `results/user_habit_profile.json`.
  - Đồng bộ việc reset về `0` cả trong bộ nhớ lẫn trên file JSON khi người chơi nhấn **R** để chơi ván mới.
  - Tách biệt kiểm thử bằng biến `_skip_file_io` tránh làm bẩn tệp tin thực tế của người dùng khi chạy `pytest`.

### D. Tương tác Vòng đời Agent (`agents/cfr_agent.py`)
- **Quan sát trung gian (Observation Forwarding)**:
  - Bổ sung hàm `observe()` cho CFR Agent để chuyển tiếp tất cả các hành động trong game sang cho Bayesian fallback agent. Nhờ đó, hồ sơ thói quen của đối thủ vẫn được cập nhật liên tục ngay cả khi CFR Agent đang ở chế độ tra bảng và chưa từng kích hoạt fallback.
- **Reset đồng bộ**:
  - Cài đặt phương thức `reset()` cho CFR Agent để dọn dẹp và reset trạng thái của cả CFR lẫn Bayesian fallback agent khi bắt đầu ván đấu mới.
- **Lưu file weights an toàn (Atomic Write)**:
  - Sử dụng tệp tạm thời (`tempfile`) và thực hiện `os.replace` để tránh trường hợp đứt gãy/hỏng file weights khi tắt game đột ngột.

### E. Mở rộng Hệ thống Kiểm thử (`tests/`)
- Bổ sung test case `test_cfr_agent_observe` kiểm tra việc chuyển tiếp quan sát thành công.
- Bổ sung test case `test_bayesian_agent_load_save_profile` kiểm tra việc đọc/ghi và reset file `user_habit_profile.json`.
- Tất cả 25 bài kiểm thử tự động của hệ thống đều vượt qua thành công:
  ```bash
  tests\test_bayesian_agent.py .........                                   [ 36%]
  tests\test_cfr_agent.py ....                                             [ 52%]
  tests\test_engine.py .......                                             [ 80%]
  tests\test_probabilistic_agent.py ...                                    [ 92%]
  tests\test_probability.py ..                                             [100%]
  ============================= 25 passed in 1.83s ==============================
  ```
