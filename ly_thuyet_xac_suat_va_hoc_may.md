# CƠ SỞ LÝ THUYẾT XÁC SUẤT VÀ HỌC MÁY TRONG DỰ ÁN LIAR'S DICE AI

Tài liệu này trình bày chi tiết các công thức toán học, mô hình xác suất và thuật toán học máy được triển khai trong mã nguồn thực tế của dự án.

---

## PHẦN I: LÝ THUYẾT XÁC SUẤT (PROBABILITY THEORY)

Mô hình xác suất trong Liar's Dice được sử dụng để tính toán **khả năng đối thủ nói thật** đối với nước cược hiện tại của họ.

### 1. Hàm phân phối nhị thức (Binomial Distribution)

Giả sử đối thủ đưa ra một nước cược (Bid) gồm số lượng $Q$ quân xúc xắc có mặt giá trị là $F$. 
Từ góc nhìn của AI:
* AI đang cầm trên tay bộ xúc xắc của mình và đếm được có $my\_count$ quân xúc xắc có mặt $F$ (hoặc mặt $1$ - quân vạn năng).
* Đối thủ đang sở hữu $n$ quân xúc xắc ẩn (`opponent_dice_count`).
* AI cần tính xác suất để đối thủ có **ít nhất** $k$ quân xúc xắc mặt $F$ trong số $n$ quân ẩn của họ, với:
  $$k = Q - my\_count$$

Nếu $k \le 0$, điều này có nghĩa là lượng xúc xắc trên tay AI đã đủ để đáp ứng nước cược $Q$ của đối thủ $\Rightarrow$ Xác suất đối thủ nói thật là chắc chắn:
$$P_{truth} = 1.0$$

Nếu $k > 0$, việc đối thủ gieo được bao nhiêu quân mặt $F$ trong số $n$ quân xúc xắc ẩn tuân theo **Phân phối nhị thức**:
$$X \sim Binomial(n, p)$$

Trong đó:
* $n$: Số lượng xúc xắc ẩn của đối thủ.
* $p$: Xác suất một quân xúc xắc gieo trúng mặt $F$. Theo luật chơi Liar's Dice có mặt 1 là Wildcard (vạn năng):
  * Nếu mặt cược $F = 1$ (chỉ đếm mặt 1):
    $$p = \frac{1}{6}$$
  * Nếu mặt cược $F \in \{2, 3, 4, 5, 6\}$ (đếm mặt $F$ và mặt 1):
    $$p = \frac{1}{6} (\text{mặt } F) + \frac{1}{6} (\text{mặt } 1) = \frac{2}{6} = \frac{1}{3}$$

Xác suất để đối thủ gieo được **đúng** $i$ quân xúc xắc mặt $F$ là:
$$P(X = i) = \binom{n}{i} p^i (1-p)^{n-i}$$

Với hệ số tổ hợp:
$$\binom{n}{i} = \frac{n!}{i!(n-i)!}$$

### 2. Xác suất đối thủ nói thật ($P_{truth}$)
Xác suất để đối thủ nói thật (tức là tổng số xúc xắc mặt $F$ thực tế trên bàn $\ge Q$, tương đương số xúc xắc ẩn của đối thủ $\ge k$) là xác suất tích lũy ở đuôi phải (cumulative distribution function - tail probability):
$$P_{truth} = P(X \ge k) = \sum_{i=k}^{n} \binom{n}{i} p^i (1-p)^{n-i}$$

*Đây chính là thuật toán được cài đặt trong hàm `binomial_probability` tại file [probability.py](file:///c:/Users/Nguyen Duc Vu Bao/Desktop/ki2/co so AI/BTL_AIT2004/liars-dice-ai/core/probability.py).*

---

## PHẦN II: MÔ HÌNH HÓA ĐỐI THỦ VÀ SUY LUẬN BAYES (BAYESIAN OPPONENT MODELING)

Để không chỉ dựa vào xác suất lý thuyết tĩnh, AI sử dụng suy luận Bayes để ước lượng xu hướng nói dối (bluff rate) của bạn dựa trên lịch sử đối đầu.

### 1. Phép cập nhật liên hợp Beta-Bernoulli (Conjugate Prior)
AI coi hành vi *"đối thủ có nói dối (bluff) khi cược hay không"* là một biến ngẫu nhiên tuân theo phép thử Bernoulli với xác suất $\theta \in [0, 1]$ chưa biết.

* **Tiên nghiệm (Prior Distribution):** 
  AI giả định $\theta$ tuân theo phân phối Beta:
  $$\theta \sim Beta(\alpha_0, \beta_0)$$
  Hàm mật độ xác suất của phân phối Beta là:
  $$f(\theta) = \frac{1}{B(\alpha, \beta)} \theta^{\alpha - 1} (1-\theta)^{\beta - 1}$$
  Dự án thiết lập tiên nghiệm mặc định là $Beta(1, 2)$, tương ứng kỳ vọng tiên nghiệm về tỷ lệ nói dối ban đầu là:
  $$E[\theta] = \frac{\alpha_0}{\alpha_0 + \beta_0} = \frac{1}{1 + 2} = \frac{1}{3} \approx 33.3\%$$

* **Cập nhật hậu nghiệm (Posterior Update):**
  Mỗi lần một lượt cược của đối thủ bị lật tẩy bởi Challenge, ta thu được 1 quan sát:
  * Nói dối (Bluff / cược sai thực tế) $\rightarrow$ Thành công ($x = 1$).
  * Thật thà (Cược đúng thực tế) $\rightarrow$ Thất bại ($x = 0$).
  
  Sau $n$ lượt cược được kiểm chứng, trong đó có $k$ lần đối thủ thực sự nói dối (bluff), theo định lý Bayes, phân phối hậu nghiệm của $\theta$ vẫn là phân phối Beta (tính chất liên hợp):
  $$\theta | \text{dữ liệu} \sim Beta(\alpha_{new}, \beta_{new})$$
  Với công thức cập nhật cực kỳ đơn giản:
  $$\alpha_{new} = \alpha_0 + k$$
  $$\beta_{new} = \beta_0 + (n - k)$$

* **Ước lượng tỷ lệ nói dối thực tế ($\hat{\theta}$):**
  Kỳ vọng toán học của phân phối hậu nghiệm chính là tỷ lệ nói dối ước lượng của đối thủ:
  $$\hat{\theta} = E[\theta | \text{dữ liệu}] = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n}$$
  *(Công thức này tương ứng hàm `_calc_rate` trong [bayesian_agent.py](file:///c:/Users/Nguyen Duc Vu Bao/Desktop/ki2/co so AI/BTL_AIT2004/liars-dice-ai/agents/bayesian_agent.py)).*

### 2. Thiết lập ngưỡng nghi ngờ thích nghi động
Từ tỷ lệ nói dối ước lượng của đối thủ $\hat{\theta}$, AI tính toán **Ngưỡng nghi ngờ động** ($dynamic\_threshold$):
$$Base\_Threshold = 0.3 + \hat{\theta} \times 0.5$$

Để tránh việc AI quá khờ khạo khi đối thủ quá thật thà, hoặc quá đa nghi, ngưỡng này được kẹp trong đoạn $[0.25, 0.65]$:
$$dynamic\_threshold = \max\left(0.25, \min\left(0.65, Base\_Threshold\right)\right)$$

### 3. Hiệu chỉnh theo hành vi tức thời (Modifier)
Ngưỡng nghi ngờ tiếp tục được nhân với hệ số điều chỉnh hành vi bước nhảy cược ($history\_bluff\_modifier$):
$$Threshold_{final} = dynamic\_threshold \times history\_bluff\_modifier$$

Trong đó, hệ số $history\_bluff\_modifier$ được cập nhật dựa trên luật:
* Nếu đối thủ cược nhảy bước lớn (tăng lượng cược $\ge 2$ con xúc xắc):
  $$history\_bluff\_modifier = 1.3 \quad (\text{Dễ là bluff hơn } \rightarrow \text{Tăng ngưỡng } \rightarrow \text{Dễ thách thức hơn})$$
* Nếu đối thủ cược bước nhỏ nhất (tăng tối thiểu $+1$ hoặc chỉ tăng mặt xúc xắc):
  $$history\_bluff\_modifier = 0.8 \quad (\text{Ít nghi ngờ hơn } \rightarrow \text{Giảm ngưỡng})$$
* Nếu đối thủ mở màn vòng đấu cược quá lớn ($> \frac{1}{3}$ tổng xúc xắc trên bàn):
  $$history\_bluff\_modifier = 1.2$$

---

## PHẦN III: THUẬT TOÁN HỌC MÁY CFR (COUNTERFACTUAL REGRET MINIMIZATION)

CFR là thuật toán học máy củng cố (Reinforcement Learning) được sử dụng để tối ưu hóa chiến thuật trong các trò chơi thông tin không hoàn hảo nhằm tìm kiếm **Cân bằng Nash**.

### 1. Khái niệm Trạng thái thông tin (Infoset)
Trạng thái thông tin $I$ chứa tất cả các trạng thái trò chơi cụ thể mà AI không thể phân biệt do thông tin bị ẩn. Trong Liar's Dice, ta định nghĩa khóa trạng thái $I$ là:
$$I = \big(my\_dice, opponent\_dice\_count, current\_bid\big)$$

### 2. Độ hối tiếc ảo (Counterfactual Regret)
Tại mỗi trạng thái thông tin $I$, nếu chọn hành động $a \in A(I)$ tại vòng lặp thứ $t$, lợi ích ảo thu được là $u_i(\sigma|_{I \to a}, h)$ (lợi ích nhận được khi cố định hành động $a$ tại $I$, các trạng thái khác đi theo chiến thuật $\sigma$).

Độ hối tiếc ảo (Counterfactual Regret) của hành động $a$ tại bước $t$ là hiệu số lợi ích:
$$r^t(I, a) = u_i(\sigma^t|_{I \to a}) - u_i(\sigma^t)$$

Tích lũy hối tiếc (Accumulated Regret) sau $T$ vòng tự chơi là:
$$R^T(I, a) = \sum_{t=1}^T r^t(I, a)$$

Để áp dụng quy tắc Regret Matching, ta chỉ lấy các giá trị hối tiếc dương (hối tiếc cho hành động mang lại kết quả tốt hơn chiến thuật hiện tại):
$$R_+^T(I, a) = \max\left(0, R^T(I, a)\right)$$

### 3. Quy tắc cập nhật chiến thuật (Regret Matching)
Chiến thuật ở vòng lặp tiếp theo $\sigma^{T+1}(I, a)$ (tức xác suất chọn hành động $a$ tại trạng thái $I$) được cập nhật tỷ lệ thuận với độ hối tiếc tích lũy của hành động đó:

$$\sigma^{T+1}(I, a) = 
\begin{cases} 
\frac{R_+^T(I, a)}{\sum_{a' \in A(I)} R_+^T(I, a')}, & \text{nếu } \sum_{a' \in A(I)} R_+^T(I, a') > 0 \\
\frac{1}{|A(I)|}, & \text{ngược lại (chọn ngẫu nhiên đều)}
\end{cases}$$

### 4. Chiến thuật trung bình (Average Strategy)
Chiến thuật cuối cùng dùng để chơi không phải là chiến thuật tại vòng lặp cuối cùng, mà là **chiến thuật trung bình** của tất cả các vòng lặp đã qua, được chuẩn hóa theo số lượt ghé thăm trạng thái:
$$\bar{\sigma}^T(I, a) = \frac{\sum_{t=1}^T s^t(I) \sigma^t(I, a)}{\sum_{t=1}^T s^t(I)}$$
Với $s^t(I)$ là trọng số đóng góp của bước $t$. 

*Chiến thuật trung bình này hội tụ về Cân bằng Nash khi $T \to \infty$ và được lưu trữ trong bảng `strategy_table` của file [cfr_agent.py](file:///c:/Users/Nguyen Duc Vu Bao/Desktop/ki2/co so AI/BTL_AIT2004/liars-dice-ai/agents/cfr_agent.py).*
