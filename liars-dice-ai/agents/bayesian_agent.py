from typing import List, Optional
from agents.base_agent import Agent
from game.actions import Action, Bid, Challenge
from game.engine import count_matching_dice
from core.probability import binomial_probability

class BayesianAgent(Agent):
    """Agent suy luận Bayes trực tuyến (online) về xu hướng nói dối (bluff) của đối thủ.

    Mô hình: coi "đối thủ có bluff hay không ở một cược ĐÃ ĐƯỢC LẬT TẨY" là một phép
    thử Bernoulli với xác suất θ chưa biết. Đặt tiên nghiệm Beta(α0, β0) cho θ và cập
    nhật hậu nghiệm sau mỗi lần một cược của đối thủ được Challenge phân định đúng/sai:

        θ̂ = E[θ | dữ liệu] = (k + α0) / (n + α0 + β0)

    với n = số cược của đối thủ đã được lật tẩy, k = số trong đó là bluff (cược sai).

    KHÁC với cách đếm thiên lệch trước đây: tử số (k) và mẫu số (n) đếm trên CÙNG một
    tập (các cược đã được phân định), nên ước lượng không bị lệch thấp. Mỗi vòng Liar's
    Dice luôn kết thúc bằng một Challenge, nên ta thu được đúng một mẫu có nhãn mỗi vòng.

    Từ θ̂ suy ra ngưỡng Challenge động: đối thủ càng hay bluff ⇒ ngưỡng càng cao ⇒ ta
    càng dễ tố cáo. Liên quan Bài 16-17 (suy luận xác suất / cập nhật niềm tin).
    """
    def __init__(self, name: str = "BayesianAgent",
                 prior_alpha: float = 1.0, prior_beta: float = 2.0):
        super().__init__(name)
        # ID của ta và đối thủ
        self.player_id: Optional[int] = None
        self.opponent_id: Optional[int] = None

        # Tiên nghiệm Beta(α0, β0). Mặc định Beta(1, 2) ⇒ kỳ vọng tiên nghiệm 1/3.
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Thống kê hậu nghiệm: n cược đã phân định, k trong đó là bluff
        self.resolved_opp_bids: int = 0
        self.opp_bluffs: int = 0

        # Hàng đợi quan sát thu được trước khi xác định được player_id
        self.pending_observations: List[tuple] = []

        # Lưu vết lượt Challenge vừa xảy ra để đối chiếu kết quả ở lượt act kế tiếp
        self.last_challenge: Optional[dict] = None

        # Số xúc xắc gần nhất của hai bên để phát hiện ai mất xúc xắc sau challenge
        self.last_dice_counts: Optional[dict] = None

    def reset(self):
        """Reset trạng thái nội bộ khi bắt đầu một game đấu mới."""
        self.player_id = None
        self.opponent_id = None
        self.resolved_opp_bids = 0
        self.opp_bluffs = 0
        self.pending_observations.clear()
        self.last_challenge = None
        self.last_dice_counts = None

    def bluff_rate(self) -> float:
        """Kỳ vọng hậu nghiệm θ̂ của xác suất đối thủ bluff (Beta-Bernoulli)."""
        return ((self.opp_bluffs + self.prior_alpha)
                / (self.resolved_opp_bids + self.prior_alpha + self.prior_beta))

    def observe(self, action: Action, acting_player: int):
        """Theo dõi hành động của người chơi để thu thập dữ liệu suy luận online."""
        if self.player_id is None:
            self.pending_observations.append((action, acting_player))
            return
        self._process_observation(action, acting_player)

    def _process_observation(self, action: Action, acting_player: int):
        # Chỉ cần ghi nhận thời điểm Challenge; kết quả (ai mất xúc xắc) sẽ được
        # đối chiếu ở lượt act() kế tiếp dựa trên thay đổi số xúc xắc.
        if isinstance(action, Challenge):
            self.last_challenge = {
                "bidder": 1 - acting_player,      # người bị tố cáo
                "challenger": acting_player,
            }

    def _resolve_last_challenge(self, observation: dict):
        """Cập nhật hậu nghiệm Beta từ kết quả của lượt Challenge gần nhất.

        Một Challenge luôn lật tẩy cược đang đứng (current_bid) của `bidder`. Nếu
        bidder chính là đối thủ thì đây là một mẫu có nhãn về xu hướng bluff của họ:
          - đối thủ mất xúc xắc ⇒ cược của họ SAI ⇒ bluff bị bắt   (k += 1, n += 1)
          - ta mất xúc xắc       ⇒ cược của họ ĐÚNG ⇒ không bluff   (n += 1)
        """
        if self.last_challenge is None or self.last_dice_counts is None:
            return

        # Chỉ học khi người bị tố cáo là đối thủ (cược của họ được phân định)
        if self.last_challenge["bidder"] == self.opponent_id:
            old_opp = self.last_dice_counts[self.opponent_id]
            new_opp = observation["opponent_dice_count"]

            self.resolved_opp_bids += 1
            if new_opp < old_opp:
                # Đối thủ (bidder) mất xúc xắc ⇒ cược sai ⇒ đó là bluff
                self.opp_bluffs += 1

        self.last_challenge = None

    def act(self, observation: dict, legal_actions: List[Action]) -> Action:
        # Khởi tạo player_id/opponent_id và xử lý các quan sát pending
        if self.player_id is None:
            self.player_id = observation["player_id"]
            self.opponent_id = 1 - self.player_id
            for action, acting_player in self.pending_observations:
                self._process_observation(action, acting_player)
            self.pending_observations.clear()

        # Đối chiếu kết quả Challenge của lượt trước (nếu có) để cập nhật niềm tin
        self._resolve_last_challenge(observation)

        # Cập nhật số xúc xắc mới nhất của cả hai bên
        self.last_dice_counts = {
            self.player_id: len(observation["my_dice"]),
            self.opponent_id: observation["opponent_dice_count"],
        }

        # Ngưỡng động ∈ [0.4, 0.6] tỉ lệ thuận với xác suất bluff hậu nghiệm:
        # đối thủ hay bluff ⇒ ngưỡng cao ⇒ dễ Challenge; thật thà ⇒ ngưỡng thấp ⇒ an toàn.
        dynamic_threshold = max(0.4, min(0.6, 0.4 + self.bluff_rate() * 0.2))

        # Logic ra quyết định (giống ProbabilisticAgent nhưng dùng ngưỡng động)
        my_dice = observation["my_dice"]
        counts = {f: count_matching_dice([my_dice], f) for f in range(1, 7)}
        current_bid = observation["current_bid"]

        # Trường hợp đi đầu vòng (chưa có ai cược)
        if current_bid is None:
            best_face = max(range(1, 7), key=lambda f: (counts[f], f))
            preferred_bids = [
                action for action in legal_actions
                if isinstance(action, Bid) and action.face_value == best_face
            ]
            if preferred_bids:
                return min(preferred_bids, key=lambda b: b.quantity)

            all_bids = [action for action in legal_actions if isinstance(action, Bid)]
            if all_bids:
                return min(all_bids, key=lambda b: b.quantity)
            return legal_actions[0]

        # Trường hợp đối thủ đã cược
        Q = current_bid.quantity
        F = current_bid.face_value
        my_count = counts[F]
        k = Q - my_count
        n = observation["opponent_dice_count"]
        p = 1/6 if F == 1 else 1/3

        # Xác suất đối thủ nói thật
        P_truth = binomial_probability(k, n, p)

        # Quyết định Challenge dựa trên ngưỡng động
        if P_truth < dynamic_threshold and any(isinstance(a, Challenge) for a in legal_actions):
            return Challenge()

        # Ngược lại, nâng cược an toàn nhất theo mặt ta giữ nhiều nhất
        sorted_faces = sorted(range(1, 7), key=lambda f: (counts[f], f), reverse=True)
        for face in sorted_faces:
            matching_bids = [
                action for action in legal_actions
                if isinstance(action, Bid) and action.face_value == face
            ]
            if matching_bids:
                return min(matching_bids, key=lambda b: b.quantity)

        all_bids = [action for action in legal_actions if isinstance(action, Bid)]
        if all_bids:
            return min(all_bids, key=lambda b: b.quantity)

        challenge_actions = [a for a in legal_actions if isinstance(a, Challenge)]
        if challenge_actions:
            return challenge_actions[0]

        return legal_actions[0]
