# Danh sách các Issues để phát triển thành NCKH

Dựa trên những góp ý cải thiện, đây là danh sách các Issue cần được xử lý, chia theo mức độ ưu tiên và phân công vai trò.

## 🔴 Mức độ Cao (Blocker / Cần làm ngay trong Tuần 1)

### Issue 1: Cập nhật Agent API (Thêm vòng đời) [Người 1]
- **Vấn đề:** Các Agent hiện tại chỉ có hàm `act()`, khiến `BayesianAgent` không cập nhật được belief sau mỗi nước đi của đối thủ hoặc dễ bị rò rỉ trạng thái giữa các ván đấu.
- **Task:** Thêm hàm `observe(self, action, acting_player)` và `reset(self)` vào interface của `Agent`.
- **Note:** Cần chốt gấp để Người 2 có thể viết code AI dựa trên API này.

### Issue 2: Chốt Log Format dạng JSON [Người 1 + Người 3]
- **Vấn đề:** Nếu log chỉ in ra terminal dưới dạng chuỗi (text), Người 3 sẽ không thể tự động parse dữ liệu để phân tích (thống kê bluff, tính win-rate...).
- **Task:** Định nghĩa format log chuẩn. Mỗi sự kiện (`bid`, `challenge`, `game_end`) phải là một dòng JSON.
- **Note:** Phải chốt sớm để Người 1 code vào `Referee` và Người 3 bắt đầu code phân tích.

### Issue 3: Chốt Câu hỏi Nghiên cứu (Hypotheses) [Cả Nhóm]
- **Vấn đề:** Báo cáo hiện tại đang thiếu mục tiêu khoa học. Cần có hypothesis rõ ràng để việc code có mục đích chứng minh.
- **Task:** Chọn 2-3 câu hỏi cốt lõi. Ví dụ: *"BayesianAgent có khai thác triệt để được ProbabilisticAgent không?"* hoặc *"CFR có hội tụ về Nash Equilibrium không?"*

### Issue 4: Rủi ro Information Leakage [Người 1]
- **Vấn đề:** Cực kỳ nguy hiểm nếu biến `observation` truyền vào `act()` chứa luôn mảng xúc xắc của đối thủ.
- **Task:** Rà soát lại hàm `get_observation()` trong `GameState`, đảm bảo tuyệt đối không rò rỉ dữ liệu (chỉ trả về số lượng xúc xắc của đối thủ, không trả về giá trị mặt).

---

## 🟡 Mức độ Trung bình (Cần làm trong Tuần 1 & Tuần 2)

### Issue 5: Unit test riêng biệt cho luật Wild (mặt 1) trong xác suất [Người 2]
- **Vấn đề:** Công thức tính xác suất nhị thức rất dễ tính sai số lượng mặt 1 (do nó là wild). Nếu module tính sai, toàn bộ AI sau này sẽ đưa ra quyết định sai một cách tĩnh lặng.
- **Task:** Viết test cứng (hard-coded) các trường hợp đếm xác suất có chứa xúc xắc mặt 1 để đảm bảo hàm `probability.py` trả về đúng.

### Issue 6: Viết stub cho main.py sớm [Người 3]
- **Vấn đề:** Nếu để `main.py` đến cuối mới viết thì sẽ dễ gặp lỗi tích hợp (integration) giữa phần AI và Engine.
- **Task:** Viết trước bộ khung CLI (gọi các mode `--play`, `--tournament`, `--demo`) với dữ liệu giả.

### Issue 7: Tối ưu hoá probability.py [Người 2]
- **Vấn đề:** Khi CFR chạy self-play hàng triệu ván, tính toán xác suất sẽ bị thắt cổ chai (bottleneck).
- **Task:** Biến các hàm trong `probability.py` thành *pure functions* và dùng decorator `@functools.lru_cache` để tái sử dụng kết quả.

### Issue 8: Tài liệu hoá Invariants của GameState [Người 1]
- **Vấn đề:** Nếu không làm rõ GameState đảm bảo các tính chất bất biến nào (VD: tổng xúc xắc giảm, bid phải luôn tăng), Unit Test sẽ thiếu các case hiểm.
- **Task:** Thêm comment/docstring giải thích rõ các luật bất biến trong `state.py`.

---

## 🟢 Mức độ Thấp (Tập trung cho Báo cáo & Evaluation - Tuần 2 & 3)

### Issue 9: Thiết kế Tournament chống chia sẻ trạng thái (Shared State) [Người 3]
- **Vấn đề:** Khi chạy đa luồng (multiprocessing), các Agent nếu dùng chung instance sẽ bị lẫn lộn belief (trạng thái).
- **Task:** Tournament luôn phải khởi tạo instance Agent mới hoàn toàn cho mỗi trận. Ghi chú rõ điều này vào code.

### Issue 10: Nâng cấp Evaluation Metrics [Người 3]
- **Vấn đề:** Chỉ so sánh win-rate là chưa đủ thuyết phục cho một NCKH.
- **Task:** 
  - Tính khoảng tin cậy (Confidence Interval) hoặc p-value cho win-rate.
  - Phân tích tần suất Bluff (Bluff frequency).
  - Thêm metric đo độ đo Exploitability so với Nash Equilibrium.

### Issue 11: Bổ sung Cơ sở Lý thuyết vào Báo cáo [Người 2, Người 3]
- **Task:** Báo cáo phải giải thích được *tại sao* game này cần CFR (do không tính được Nash trực tiếp vì thông tin không hoàn hảo). Giải thích rõ Prior, Likelihood trong Bayes, và Chance node trong Expectimax.
