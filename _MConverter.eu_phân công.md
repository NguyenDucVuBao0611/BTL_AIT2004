Chuẩn --- Liar\'s Dice là lựa chọn tốt. Trước khi chia việc, tao chốt 1 quyết định scope quan trọng để nhóm khỏi sa lầy:

Làm bản 2 người chơi (2-player), mỗi người 5 xúc xắc, số 1 là wild.\
Lý do: lý thuyết trò chơi 2-player zero-sum có lời giải sạch (cân bằng/CFR well-defined). Multi-player thì game theory rối

Phân công 3 người (gắn với bài trong môn để defend vấn đáp)

| Người | Vai trò | Chạm bài nào của môn |
|----|----|----|
| Người 1 | Game Engine & Referee | Mô hình hóa trò chơi, trạng thái (nền cho B4-5) |
| Người 2 | AI Agents (lõi thuật toán) | Expectimax/bất định (B5), Xác suất & Bayes (B16-17), RL (B8-9) |
| Người 3 | Evaluation, UI/Demo & Báo cáo | Phân tích chiến lược, demo, slide |

Người 1 --- Game Core (làm xong SỚM nhất, cả nhóm phụ thuộc)

- GameState: xúc xắc giấu của mỗi bên, bid hiện tại, lịch sử, lượt ai.

- Luật: sinh nước đi hợp lệ (tăng số lượng / tăng mặt), xử lý challenge (\"Liar!\"), wild số 1, mất xúc xắc, thắng/thua.

- Referee: vòng lặp cho 2 agent bất kỳ đánh nhau tới khi kết thúc.

- Unit test luật --- cực quan trọng, engine sai thì mọi kết quả vô nghĩa.

- Định dạng log ván đấu để Người 3 phân tích.

- Milestone tuần 1: 2 RandomAgent chơi được trọn 1 ván.

Người 2 --- AI Engineer (lõi \"trí tuệ\", coder mạnh nhất)

Làm 4 agent theo độ khó tăng dần:

1.  RandomAgent --- baseline, bid ngẫu nhiên hợp lệ.

2.  ProbabilisticAgent (lõi đúng môn) --- dùng phân phối nhị thức tính P(bid đúng) từ xúc xắc của mình + số xúc xắc ẩn của địch → challenge khi P \< ngưỡng, ngược lại bid theo expectimax (chance node = xúc xắc ẩn). Đây là phần defend được với thầy (B5 + B16).

3.  BayesianAgent --- cập nhật niềm tin về xu hướng bluff của đối thủ từ các bid của nó (suy luận Bayes, B16-17), rồi khai thác.

4.  CFRAgent (nâng cao/wow) --- Counterfactual Regret Minimization, tự học chiến lược hỗn hợp gần tối ưu qua self-play (như poker AI). *Đây là stretch goal --- cắt được nếu thiếu thời gian.*

Người 3 --- Evaluation + UI + Báo cáo

- Tournament: cho các agent đánh vòng tròn vài nghìn ván → ma trận win-rate, ai thắng ai.

- Biểu đồ: win-rate, tần suất bluff, độ hội tụ của CFR (exploitability giảm theo vòng lặp) → phần \"wow\" trên slide.

- Demo người vs AI: CLI trước (dùng thư viện rich cho đẹp), rồi GUI đơn giản (pygame hoặc web) nếu kịp.

- Slide + báo cáo + video demo, test tích hợp cuối.

Hợp đồng quan trọng nhất: Agent API (chốt NGÀY 1)

Đây là điểm để 3 người làm song song không chặn nhau. Người 1 định nghĩa, cả nhóm tuân theo:

*\# agents/base_agent.py*

class Agent:

def act(self, observation, legal_actions) -\> Action:

\"\"\"observation = (my_dice, current_bid, history, dice_count_per_player, turn)

legal_actions = danh sách Bid hợp lệ + Challenge

return: một Action\"\"\"

\...

Chốt cái này xong, Người 2 code agent với engine giả, Người 3 code tournament với agent giả --- không ai phải đợi ai.

Cấu trúc thư mục

liars-dice-ai/

├── README.md

├── requirements.txt *\# numpy, matplotlib, rich, pytest, (pygame)*

├── .gitignore *\# logs/, \_\_pycache\_\_/, results/\*.png*

│

├── game/ *\# ── Người 1 ──*

│ ├── \_\_init\_\_.py

│ ├── state.py *\# GameState, Dice*

│ ├── actions.py *\# Bid, Challenge*

│ ├── engine.py *\# luật, legal_actions, xử lý challenge*

│ └── referee.py *\# vòng lặp match giữa 2 agent*

│

├── agents/ *\# ── Người 2 ──*

│ ├── \_\_init\_\_.py

│ ├── base_agent.py *\# interface Agent (hợp đồng chung)*

│ ├── random_agent.py

│ ├── probabilistic_agent.py

│ ├── bayesian_agent.py

│ └── cfr_agent.py *\# nâng cao*

│

├── core/ *\# dùng chung*

│ ├── \_\_init\_\_.py

│ └── probability.py *\# nhị thức: P(≥k mặt f trong n xúc xắc ẩn)*

│

├── evaluation/ *\# ── Người 3 ──*

│ ├── \_\_init\_\_.py

│ ├── tournament.py *\# round-robin, ma trận win-rate*

│ └── plots.py *\# biểu đồ*

│

├── ui/ *\# ── Người 3 ──*

│ ├── cli.py *\# người vs AI trong terminal*

│ └── gui.py *\# tùy chọn (pygame/web)*

│

├── tests/ *\# Người 1 chủ trì*

│ ├── test_engine.py

│ ├── test_probability.py

│ └── test_agents.py

│

├── logs/ *\# log ván đấu (gitignore)*

├── results/ *\# biểu đồ, thống kê cuối*

├── docs/

│ └── report.md

└── main.py *\# entry: \--mode play\|tournament\|demo*

Timeline 3 tuần (present 27/6)

Tuần 1 --- Nền tảng

- N1: Engine + Referee + RandomAgent + test. *Milestone: 2 random chơi trọn ván.*

- N2: Học luật/chiến thuật; viết core/probability.py; xong ProbabilisticAgent.

- N3: Dựng repo + README + tournament skeleton + CLI sơ khai.

Tuần 2 --- Lõi AI

- N1: Xử lý edge case (wild 1, exact call), đóng băng API, hỗ trợ tích hợp.

- N2: BayesianAgent xong; bắt đầu CFR/self-play.

- N3: Tournament đầy đủ + ma trận win-rate; CLI người-vs-AI chạy; bắt đầu biểu đồ.

Tuần 3 --- Hoàn thiện & demo

- N1: Fix bug, test cuối, viết phần luật trong báo cáo.

- N2: Hoàn thiện CFR + phân tích exploitability; tinh chỉnh; viết phần thuật toán.

- N3: Kết quả tournament cuối, biểu đồ, demo người-vs-AI mượt, slide, video.

Tech stack & cắt gọt nếu thiếu thời gian

- Bắt buộc: Python + numpy + matplotlib + pytest. Demo CLI bằng rich.

- Cắt được: GUI pygame (CLI là đủ để demo), và CFRAgent (3 agent đầu đã là project mạnh --- ProbabilisticAgent + BayesianAgent phủ đúng B5 + B16-17).

- Insight để defend (giống bài Blotto): với 2-player zero-sum + đối thủ tối ưu, chiến lược tốt nhất là hỗn hợp/bluff có tần suất chính xác để bất khả khai thác; agent xác suất đơn thuần sẽ bị BayesianAgent đọc vị → đó là lý do cần tới CFR.
