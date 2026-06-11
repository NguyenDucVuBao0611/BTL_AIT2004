# Báo cáo Bài tập lớn — Liar's Dice AI

**Môn:** Cơ sở Trí tuệ Nhân tạo (INT3401)
**Đề tài:** Tác tử chơi Liar's Dice 2 người dựa trên xác suất và lý thuyết trò chơi

---

## 1. Giới thiệu bài toán

Liar's Dice là trò chơi **thông tin không hoàn hảo, tổng bằng không (zero‑sum)** giữa 2
người. Mỗi người giữ kín 5 xúc xắc (mặt 1 là *wild*, được tính cho mọi mặt). Người chơi
lần lượt **cược** "có ít nhất *Q* viên mặt *F*" trên toàn bàn, mỗi cược phải cao hơn cược
trước (tăng số lượng, hoặc giữ số lượng và tăng mặt). Thay vì cược, người chơi có thể hô
**Challenge ("Liar!")** để lật xúc xắc: cược đúng thì người tố cáo thua, cược sai thì
người cược thua; người thua mất 1 xúc xắc. Ai hết xúc xắc trước là thua chung cuộc.

Đây là môi trường tốt để minh hoạ **tìm kiếm bất định (Bài 5)**, **suy luận xác suất /
Bayes (Bài 16‑17)**, **tìm kiếm đối kháng & lý thuyết trò chơi (Bài 4)**, và **học qua
tương tác / regret minimization (Bài 8‑9)**.

## 2. Kiến trúc hệ thống

```
game/      GameState, actions (Bid/Challenge), engine (luật, legal_actions), referee
agents/    base_agent (API chung) + random, probabilistic, bayesian, cfr
core/      probability.py — nhị thức tích lũy P(X >= k)
evaluation/ tournament (seed + cân bằng ghế), plots, exploitability
ui/        cli.py (rich), gui.py (pygame)
tests/     unit test engine + agents (pytest)
```

**Hợp đồng Agent (API chung)** — `agents/base_agent.py`:
- `act(observation, legal_actions) -> Action`: chọn nước đi.
- `observe(action, acting_player)`: cập nhật trạng thái sau mỗi nước đi của bất kỳ ai.
- `reset()`: xoá trạng thái nội bộ trước mỗi ván.

**Chống rò rỉ thông tin:** `GameState.get_observation()` chỉ trả `opponent_dice_count`,
không bao giờ lộ giá trị xúc xắc của đối thủ — bất biến được kiểm thử.

## 3. Các tác tử AI

### 3.1 RandomAgent (baseline)
Chọn ngẫu nhiên một nước đi hợp lệ. Dùng làm mốc tham chiếu.

### 3.2 ProbabilisticAgent — kỳ vọng tham lam 1 bước
Ước lượng xác suất đối thủ "nói thật" cho cược hiện tại bằng **phân phối nhị thức tích
luỹ**. Gọi `k = Q − (số mặt F ta đang giữ, tính wild)` là số viên đối thủ cần có, `n` là
số xúc xắc ẩn của đối thủ, và xác suất một viên khớp `p = 1/3` cho mặt ≠ 1 (mặt F hoặc
wild 1) hay `p = 1/6` cho mặt 1. Khi đó:

$$P(\text{nói thật}) = P(X \ge k),\quad X \sim \mathrm{Binomial}(n, p).$$

Nếu `P(nói thật) < ngưỡng` (mặc định 0.5) thì **Challenge**; ngược lại nâng cược an toàn
nhất (số lượng nhỏ nhất của mặt mình giữ nhiều nhất).

**Giới hạn (đã ghi đúng trong mã):** đây là heuristic **1 bước**, *không* phải expectimax
khai triển cây. Chiến lược thuần tuý, không bao giờ bluff ⇒ dễ đoán và về lý thuyết có
thể bị khai thác bởi một đối thủ học được khuôn mẫu.

### 3.3 BayesianAgent — cập nhật niềm tin Beta‑Bernoulli
Coi việc "một cược *đã được lật tẩy* của đối thủ có phải bluff không" là phép thử
Bernoulli với xác suất θ chưa biết. Đặt tiên nghiệm **Beta(α₀, β₀) = Beta(1, 2)** (kỳ
vọng 1/3) và cập nhật hậu nghiệm:

$$\hat\theta = \mathbb{E}[\theta \mid \text{dữ liệu}] = \frac{k + \alpha_0}{n + \alpha_0 + \beta_0},$$

với `n` = số cược của đối thủ đã được Challenge phân định, `k` = số trong đó là bluff
(cược sai). Từ θ̂ suy ra **ngưỡng Challenge động** ∈ [0.4, 0.6]: đối thủ càng hay bluff ⇒
ngưỡng càng cao ⇒ ta càng dễ tố cáo.

**Sửa lỗi thiên lệch quan trọng (so với bản đầu):** bản trước đếm *mọi* cược vượt kỳ vọng
vào mẫu số nhưng chỉ đếm bluff *bị bắt* vào tử số ⇒ luôn đánh giá đối thủ thật thà hơn
thực tế. Bản này chỉ học từ các cược **đã được phân định** và đếm cả hai kết cục (bluff bị
bắt *và* cược đúng bị tố oan) trên **cùng một tập mẫu** ⇒ ước lượng không lệch. Mỗi vòng
Liar's Dice luôn kết thúc bằng một Challenge nên ta thu được đúng một mẫu có nhãn mỗi vòng.

**Lưu ý trung thực:** đây là *cập nhật niềm tin Bayes* (Bài 16‑17), **không** phải bộ phân
loại Naive Bayes (Bài 10).

### 3.4 CFRAgent — lấy cảm hứng từ Counterfactual Regret Minimization
Tự chơi (self‑play) duyệt cây trò chơi đã **trừu tượng hoá hành động** (giữ Challenge +
bid tối thiểu mỗi mặt + bid tăng 1 đơn vị; tối đa 6 hành động), dùng **regret matching**
để cập nhật chiến lược hỗn hợp, và lưu chiến lược trung bình. Khi gặp trạng thái chưa học,
tự động **fallback** sang ProbabilisticAgent.

**Giới hạn lý thuyết (nêu rõ để defend đúng mực):**
1. **Giới hạn độ sâu** (`max_depth = 4`) với ước lượng nút lá bằng heuristic.
2. **Nút lá đánh giá với thông tin hoàn hảo** (biết cả xúc xắc đối thủ trong mỗi ván
   self‑play) thay vì lấy kỳ vọng đúng trên thông tin ẩn.

Do đó đây là biến thể **xấp xỉ**, *không* đảm bảo hội tụ về cân bằng Nash chính xác. Mục
tiêu thực tế là học chiến lược hỗn hợp **ít bị khai thác** trong trò chơi trừu tượng hoá.

## 4. Phương pháp đánh giá

### 4.1 Giải đấu round‑robin (tái lập + cân bằng ghế)
`evaluation/tournament.py` gieo hạt RNG một lần (`--seed`) và **luân phiên ghế đi trước**
trong mỗi cặp đấu, nhờ đó khử lợi thế đi trước khỏi từng ô của ma trận win‑rate. Lệnh:
```bash
python main.py --mode tournament --games 50 --seed 0
```

### 4.2 Đo hội tụ / exploitability của CFR
`evaluation/exploitability.py` huấn luyện CFR theo từng đợt và ghi lại tại mỗi mốc:
- **`avg_regret`** = regret dương trung bình / vòng lặp (`CFRAgent.average_regret()`).
  Theo lý thuyết CFR, đại lượng này **chặn trên exploitability** và phải tiến về 0.
- **`winrate_vs_prob`** = win‑rate thực nghiệm của chiến lược CFR hiện tại khi đấu
  ProbabilisticAgent (đối thủ cố định), đo bằng giải đấu cân bằng ghế.
```bash
python main.py --mode exploit --iters 20000 --seed 0
```

## 5. Kết quả thực nghiệm

> Chạy với `--seed 0`. Lệnh tái lập ở §8. Biểu đồ trong `results/`.

### 5.1 Giải đấu round‑robin (200 ván/cặp, cân bằng ghế)

| Agent | Win‑rate tổng | Ghi chú đối đầu |
|---|---:|---|
| BayesianAgent | **75.7%** | Thắng Probabilistic 51.5%, thắng CFR 75.5% |
| ProbabilisticAgent | 74.0% | Thắng CFR 73.5% |
| CFRAgent (xấp xỉ) | 50.3% | Thua Probabilistic 26.5%, thua Bayesian 24.5% |
| RandomAgent | 0.0% | Thua mọi agent có chiến lược |

Nhận xét: cả hai agent dựa trên xác suất đè bẹp RandomAgent (100%). **BayesianAgent nhỉnh
hơn ProbabilisticAgent** nhờ ngưỡng Challenge động học được — đúng kỳ vọng thiết kế. CFR
**xấp xỉ** chỉ ở mức trung bình, phù hợp với giới hạn lý thuyết ở §3.4. Biểu đồ:
`results/win_matrix.png`, `results/agent_stats.png`.

### 5.2 Hội tụ CFR (20000 vòng lặp, bước 2000, eval 200 ván/mốc)

| Vòng lặp | avg_regret (↓) | Win‑rate vs Probabilistic |
|---:|---:|---:|
| 2000  | 0.000141 | 19.0% |
| 6000  | 0.000075 | 24.0% |
| 10000 | 0.000057 | 25.5% |
| 14000 | 0.000047 | 29.5% |
| 20000 | 0.000039 | 39.5% |

Regret trung bình **giảm đơn điệu** `0.000141 → 0.000039` ⇒ xác nhận hội tụ (đại lượng
chặn trên exploitability tiến về 0). Song song, **win‑rate thực nghiệm cải thiện**
`19% → 39.5%` ⇒ chiến lược ngày càng mạnh hơn theo huấn luyện, dù vẫn dưới 50% trước một
agent xác suất chắc chắn (hệ quả của xấp xỉ giới hạn độ sâu + nút lá thông tin hoàn hảo).
Đây là kết luận trung thực, củng cố phần giới hạn ở §3.4. Biểu đồ:
`results/cfr_convergence.png`.

## 6. Mức độ phủ chương trình môn học

| Bài giảng | Mức độ chạm | Thể hiện ở |
|---|---|---|
| B5 — Tìm kiếm bất định | ✅ | Ước lượng nhị thức + quyết định theo kỳ vọng (ProbabilisticAgent) |
| B16‑17 — Suy luận Bayes | ✅ | Hậu nghiệm Beta‑Bernoulli (BayesianAgent) |
| B4 — Tìm kiếm đối kháng | ✅ | Duyệt cây + regret matching (CFRAgent) |
| B8‑9 — Học qua tương tác | ⚠️ một phần | Self‑play regret minimization (không phải MDP/Q‑learning) |
| B2‑3, B6‑7, B10‑13 | ❌ | Ngoài phạm vi đề tài |

## 7. Hạn chế & hướng phát triển
- Nâng CFR lên đúng chuẩn: lấy kỳ vọng trên thông tin ẩn ở nút lá (chance sampling theo
  xúc xắc đối thủ) thay vì thông tin hoàn hảo; bỏ giới hạn độ sâu hoặc dùng đánh giá nút
  lá tốt hơn ⇒ kỳ vọng cải thiện win‑rate.
- ProbabilisticAgent có thể thêm thành phần **bluff với tần suất** để bớt bị khai thác.
- Mở rộng đo **exploitability bằng best‑response** thực thụ (thay vì chỉ regret bound).

## 8. Cách tái lập
```bash
pip install -r requirements.txt
python -m pytest -q                                  # 20 test pass
python main.py --mode tournament --games 50 --seed 0 # ma trận + biểu đồ
python main.py --mode exploit --iters 20000 --seed 0 # đường hội tụ CFR
```
