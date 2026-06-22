# Liar's Dice AI — Bài tập lớn Cơ sở Trí tuệ Nhân tạo (INT3401)

Cài đặt trò chơi **Liar's Dice 2 người** (mỗi người 5 xúc xắc, mặt **1 là wild**) cùng
một bộ tác tử (agent) AI tăng dần độ khó, hệ thống giải đấu, đo hội tụ, và giao diện
chơi người‑vs‑AI (CLI + GUI).

## Trò chơi tóm tắt
Mỗi người giữ kín xúc xắc của mình. Lần lượt **cược** "có ít nhất *Q* viên mặt *F*"
(tính trên toàn bàn, mặt 1 là wild được tính cho mọi mặt). Cược sau phải **cao hơn**:
tăng số lượng, hoặc giữ số lượng và tăng mặt. Thay vì cược, người chơi có thể hô
**"Liar!" (Challenge)**. Khi đó lật xúc xắc: nếu cược đúng (đủ số lượng) thì người tố
cáo thua, nếu sai thì người cược thua. Người thua **mất 1 xúc xắc**. Hết xúc xắc là thua chung cuộc.

## Cài đặt
```bash
cd liars-dice-ai
pip install -r requirements.txt
```
Yêu cầu Python 3.10+. `rich`/`pygame` chỉ cần cho chế độ chơi tương tác; phần lõi và
giải đấu chỉ cần `matplotlib` (vẽ biểu đồ) và `pytest` (test).

## Chạy
Chạy từ thư mục `liars-dice-ai/` (mã nguồn nằm trong thư mục con này):
```bash
cd liars-dice-ai
python main.py --mode demo                 # 2 ván demo có log chi tiết
python main.py --mode tournament --games 50 --seed 0   # giải đấu round-robin (tái lập)
python main.py --mode exploit --iters 20000            # đo hội tụ/exploitability CFR
python main.py --mode cli  --agent bayesian            # chơi người-vs-AI trên terminal
python main.py --mode gui  --agent cfr                 # chơi người-vs-AI bằng pygame
python main.py --mode all                              # demo + tournament
```
Chạy test:
```bash
python -m pytest -q
```

> 💡 **Weights CFR đã train sẵn:** bản nén `experiments/cfr_heavy.weights.json.gz` (~7MB)
> được commit kèm repo, nên clone về là chơi/đấu với CFR **mạnh** ngay, không cần train lại
> 40k vòng. `main.py` tự nạp file `.gz` này. Nếu xóa file đi, các mode sẽ tự train fallback
> (yếu hơn). File `.json` chưa nén (~51MB) không được track (xem `.gitignore`).

## Các tác tử AI (theo độ khó)
| Agent | Ý tưởng | Liên quan bài giảng |
|---|---|---|
| `RandomAgent` | Chọn ngẫu nhiên một nước đi hợp lệ (baseline) | — |
| `ProbabilisticAgent` | Kỳ vọng tham lam 1 bước: ước lượng P(đối thủ nói thật) bằng phân phối **nhị thức tích lũy**, Challenge khi P < ngưỡng cố định | B5 (bất định), B16‑17 (xác suất) |
| `BayesianAgent` | Cập nhật **hậu nghiệm Beta‑Bernoulli** về xác suất bluff của đối thủ từ các cược đã bị lật tẩy → ngưỡng Challenge **động** | B16‑17 (suy luận Bayes) |
| `CFRAgent` | **CFR+** (regret matching+ + linear averaging) self‑play full‑game; chơi bằng argmax + gating, nạp weights đã train nặng. Khi train đủ (~50k vòng) **dẫn đầu giải đấu, thắng cả Prob lẫn Bayes** (§5.1) | B4 (đối kháng), B8‑9 (regret minimization) |

> ⚠️ Tên gọi đã được hiệu chỉnh cho đúng bản chất cài đặt: `ProbabilisticAgent` là
> heuristic 1 bước (không phải expectimax đầy đủ); `BayesianAgent` là cập nhật niềm tin
> Bayes (không phải Naive Bayes phân loại); `CFRAgent` là biến thể CFR **xấp xỉ**.
> Xem chi tiết và giới hạn lý thuyết trong [docs/report.md](liars-dice-ai/docs/report.md).

## Cấu trúc thư mục
```
liars-dice-ai/
├── game/         # GameState, actions, engine (luật), referee (vòng lặp đấu)
├── agents/       # base_agent + 4 agent (random, probabilistic, bayesian, cfr)
├── core/         # probability.py — nhị thức tích lũy P(X >= k)
├── evaluation/   # tournament (seed + cân bằng ghế), plots, exploitability
├── ui/           # cli.py (rich), gui.py (pygame) — chơi người-vs-AI
├── tests/        # unit test cho engine + các agent (pytest)
├── docs/report.md
├── requirements.txt
└── main.py       # entry: --mode demo|tournament|exploit|cli|gui|all
```

## Tính tái lập
Giải đấu và phép đo hội tụ đều nhận `--seed` và gieo hạt RNG một lần ở đầu. Giải đấu
**cân bằng ghế** (luân phiên ai đi trước trong mỗi cặp) để khử lợi thế đi trước khỏi
từng ô của ma trận win‑rate.

## Mức độ phủ chương trình môn học
Dự án tập trung phần **xác suất – bất định – lý thuyết trò chơi**: B4 (tìm kiếm đối
kháng), B5 (tìm kiếm bất định / chance node), B16‑17 (suy luận xác suất / Bayes), và
một phần B8‑9 (học qua tương tác / regret minimization). Dự án **không** đụng tới tìm
kiếm mù/heuristic (B2‑3), MDP (B6‑7), hay học có giám sát / mạng nơ‑ron (B10‑13).
