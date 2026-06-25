import os
import json
import sys
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

        # Thống kê hậu nghiệm theo 6 ngữ cảnh:
        self.resolved_opp_bids: int = 0
        self.opp_bluffs: int = 0

        self.resolved_opp_bids_low_dice: int = 0
        self.opp_bluffs_low_dice: int = 0

        self.resolved_opp_bids_winning: int = 0
        self.opp_bluffs_winning: int = 0

        self.resolved_opp_bids_losing: int = 0
        self.opp_bluffs_losing: int = 0

        self.resolved_opp_bids_high_qty: int = 0
        self.opp_bluffs_high_qty: int = 0

        self.resolved_opp_bids_face1: int = 0
        self.opp_bluffs_face1: int = 0

        # Thống kê phản ứng của đối thủ khi ta cược mặt lớn (>= 4)
        self.our_bids_high_face: int = 0
        self.opp_challenges_high_face: int = 0
        self._last_our_bid_high: bool = False

        # Hàng đợi quan sát thu được trước khi xác định được player_id
        self.pending_observations: List[tuple] = []

        # Lưu vết lượt Challenge vừa xảy ra để đối chiếu kết quả ở lượt act kế tiếp
        self.last_challenge: Optional[dict] = None

        # Số xúc xắc gần nhất của hai bên để phát hiện ai mất xúc xắc sau challenge
        self.last_dice_counts: Optional[dict] = None

        # Các thuộc tính log phục vụ vẽ đồ thị và Web Dashboard
        self.last_p_truth: Optional[float] = None
        self.last_threshold: Optional[float] = None
        self.last_modifier: Optional[float] = None
        self.last_base_rate: Optional[float] = None
        self.last_action_type: str = "cược"

        self._skip_file_io = "pytest" in sys.modules
        self.load_profile()

    def load_profile(self):
        if getattr(self, "_skip_file_io", False):
            return
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile_path = os.path.join(base_dir, "results", "user_habit_profile.json")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.resolved_opp_bids = data.get("resolved_opp_bids", 0)
                self.opp_bluffs = data.get("opp_bluffs", 0)
                self.resolved_opp_bids_low_dice = data.get("resolved_opp_bids_low_dice", 0)
                self.opp_bluffs_low_dice = data.get("opp_bluffs_low_dice", 0)
                self.resolved_opp_bids_winning = data.get("resolved_opp_bids_winning", 0)
                self.opp_bluffs_winning = data.get("opp_bluffs_winning", 0)
                self.resolved_opp_bids_losing = data.get("resolved_opp_bids_losing", 0)
                self.opp_bluffs_losing = data.get("opp_bluffs_losing", 0)
                self.resolved_opp_bids_high_qty = data.get("resolved_opp_bids_high_qty", 0)
                self.opp_bluffs_high_qty = data.get("opp_bluffs_high_qty", 0)
                self.resolved_opp_bids_face1 = data.get("resolved_opp_bids_face1", 0)
                self.opp_bluffs_face1 = data.get("opp_bluffs_face1", 0)
                self.our_bids_high_face = data.get("our_bids_high_face", 0)
                self.opp_challenges_high_face = data.get("opp_challenges_high_face", 0)
            except Exception as e:
                print(f"Lỗi đọc file user_habit_profile.json: {e}")

    def save_profile(self):
        if getattr(self, "_skip_file_io", False):
            return
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        profile_path = os.path.join(results_dir, "user_habit_profile.json")
        data = {
            "resolved_opp_bids": self.resolved_opp_bids,
            "opp_bluffs": self.opp_bluffs,
            "resolved_opp_bids_low_dice": self.resolved_opp_bids_low_dice,
            "opp_bluffs_low_dice": self.opp_bluffs_low_dice,
            "resolved_opp_bids_winning": self.resolved_opp_bids_winning,
            "opp_bluffs_winning": self.opp_bluffs_winning,
            "resolved_opp_bids_losing": self.resolved_opp_bids_losing,
            "opp_bluffs_losing": self.opp_bluffs_losing,
            "resolved_opp_bids_high_qty": self.resolved_opp_bids_high_qty,
            "opp_bluffs_high_qty": self.opp_bluffs_high_qty,
            "resolved_opp_bids_face1": self.resolved_opp_bids_face1,
            "opp_bluffs_face1": self.opp_bluffs_face1,
            "our_bids_high_face": self.our_bids_high_face,
            "opp_challenges_high_face": self.opp_challenges_high_face
        }
        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Lỗi ghi file user_habit_profile.json: {e}")

    def reset(self):
        """Reset trạng thái nội bộ khi bắt đầu một game đấu mới."""
        self.player_id = None
        self.opponent_id = None
        self.resolved_opp_bids = 0
        self.opp_bluffs = 0
        self.resolved_opp_bids_low_dice = 0
        self.opp_bluffs_low_dice = 0
        self.resolved_opp_bids_winning = 0
        self.opp_bluffs_winning = 0
        self.resolved_opp_bids_losing = 0
        self.opp_bluffs_losing = 0
        self.resolved_opp_bids_high_qty = 0
        self.opp_bluffs_high_qty = 0
        self.resolved_opp_bids_face1 = 0
        self.opp_bluffs_face1 = 0
        self.our_bids_high_face = 0
        self.opp_challenges_high_face = 0
        self._last_our_bid_high = False
        self.pending_observations.clear()
        self.last_challenge = None
        self.last_dice_counts = None
        self.save_profile()



    def get_best_bluff_rate(self, obs: dict, current_bid: Optional[Bid]) -> float:
        """Lấy tỷ lệ bluff phù hợp nhất dựa trên phân cấp phân loại ngữ cảnh."""
        opp_dice = obs["opponent_dice_count"]
        my_dice = len(obs["my_dice"])
        total_dice = opp_dice + my_dice

        # 1. Ngữ cảnh đối thủ ít xúc xắc (<= 2)
        if opp_dice <= 2 and self.resolved_opp_bids_low_dice >= 2:
            return self._calc_rate(self.opp_bluffs_low_dice, self.resolved_opp_bids_low_dice)
        
        # 2. Ngữ cảnh cược mặt 1
        if current_bid and current_bid.face_value == 1 and self.resolved_opp_bids_face1 >= 2:
            return self._calc_rate(self.opp_bluffs_face1, self.resolved_opp_bids_face1)

        # 3. Ngữ cảnh cược số lượng lớn (>= 50% tổng xúc xắc)
        if current_bid and current_bid.quantity >= total_dice / 2.0 and self.resolved_opp_bids_high_qty >= 2:
            return self._calc_rate(self.opp_bluffs_high_qty, self.resolved_opp_bids_high_qty)

        # 4. Ngữ cảnh thế thắng/thua
        if opp_dice > my_dice and self.resolved_opp_bids_winning >= 2:
            return self._calc_rate(self.opp_bluffs_winning, self.resolved_opp_bids_winning)
        elif opp_dice < my_dice and self.resolved_opp_bids_losing >= 2:
            return self._calc_rate(self.opp_bluffs_losing, self.resolved_opp_bids_losing)

        # 5. Ngữ cảnh mặc định
        return self._calc_rate(self.opp_bluffs, self.resolved_opp_bids)

    def _calc_rate(self, bluffs: int, resolved: int) -> float:
        return (bluffs + self.prior_alpha) / (resolved + self.prior_alpha + self.prior_beta)

    def bluff_rate(self) -> float:
        """Kỳ vọng hậu nghiệm θ̂ của xác suất đối thủ bluff (Beta-Bernoulli)."""
        return self._calc_rate(self.opp_bluffs, self.resolved_opp_bids)

    def observe(self, action: Action, acting_player: int):
        """Theo dõi hành động của người chơi để thu thập dữ liệu suy luận online."""
        if self.player_id is None:
            self.pending_observations.append((action, acting_player))
            return
        self._process_observation(action, acting_player)

    def _process_observation(self, action: Action, acting_player: int):
        # Theo dõi nếu ta thực hiện cược lớn (mặt >= 4)
        if acting_player == self.player_id and isinstance(action, Bid) and action.face_value >= 4:
            self._last_our_bid_high = True
            
        if isinstance(action, Challenge):
            self.last_challenge = {
                "bidder": 1 - acting_player,      # người bị tố cáo
                "challenger": acting_player,
            }
            if acting_player == self.opponent_id and getattr(self, "_last_our_bid_high", False):
                self.opp_challenges_high_face += 1
                self.our_bids_high_face += 1
                self._last_our_bid_high = False
                self.save_profile()
        elif acting_player == self.opponent_id and isinstance(action, Bid):
            # Nếu đối thủ không challenge nước cược mặt lớn của ta mà cược tiếp
            if getattr(self, "_last_our_bid_high", False):
                self.our_bids_high_face += 1
                self._last_our_bid_high = False
                self.save_profile()

    def _resolve_last_challenge(self, observation: dict):
        if self.last_challenge is None or self.last_dice_counts is None:
            return

        # Chỉ học khi người bị tố cáo là đối thủ (cược của họ được phân định)
        if self.last_challenge["bidder"] == self.opponent_id:
            old_opp = self.last_dice_counts[self.opponent_id]
            new_opp = observation["opponent_dice_count"]
            old_my = self.last_dice_counts[self.player_id]

            # Thử tìm nước cược cuối cùng của đối thủ từ lịch sử
            history = observation.get("history", [])
            last_bid = None
            if len(history) >= 2:
                for i in range(len(history) - 1, -1, -1):
                    if history[i][0] == self.opponent_id and isinstance(history[i][1], Bid):
                        last_bid = history[i][1]
                        break
            if last_bid is None:
                last_bid = observation.get("current_bid")

            is_bluff = new_opp < old_opp
            total_dice = old_opp + old_my

            # Cập nhật Multi-context:
            # 1. General
            self.resolved_opp_bids += 1
            if is_bluff:
                self.opp_bluffs += 1

            # 2. Opponent Low Dice (<= 2)
            if old_opp <= 2:
                self.resolved_opp_bids_low_dice += 1
                if is_bluff:
                    self.opp_bluffs_low_dice += 1

            # 3. Opponent Winning (Đối thủ nhiều xúc xắc hơn AI)
            if old_opp > old_my:
                self.resolved_opp_bids_winning += 1
                if is_bluff:
                    self.opp_bluffs_winning += 1

            # 4. Opponent Losing (Đối thủ ít xúc xắc hơn AI)
            if old_opp < old_my:
                self.resolved_opp_bids_losing += 1
                if is_bluff:
                    self.opp_bluffs_losing += 1

            # 5. High Bid Quantity (>= 50% tổng xúc xắc)
            if last_bid and last_bid.quantity >= total_dice / 2.0:
                self.resolved_opp_bids_high_qty += 1
                if is_bluff:
                    self.opp_bluffs_high_qty += 1

            # 6. Face 1 Bid
            if last_bid and last_bid.face_value == 1:
                self.resolved_opp_bids_face1 += 1
                if is_bluff:
                    self.opp_bluffs_face1 += 1

            self.save_profile()

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

        current_bid = observation["current_bid"]

        # 1. Tính toán modifier cho độ khả nghi dựa trên lịch sử bước nhảy cược
        history = observation.get("history", [])
        history_bluff_modifier = 1.0
        opp_bid_indices = [i for i, (pid, action) in enumerate(history) if pid == self.opponent_id and isinstance(action, Bid)]
        
        if opp_bid_indices:
            last_idx = opp_bid_indices[-1]
            last_opp_bid = history[last_idx][1]
            
            # Tìm cược ngay trước đó để tính bước nhảy
            prev_bid = None
            for i in range(last_idx - 1, -1, -1):
                if isinstance(history[i][1], Bid):
                    prev_bid = history[i][1]
                    break
                    
            if prev_bid:
                dq = last_opp_bid.quantity - prev_bid.quantity
                if dq >= 2:
                    history_bluff_modifier = 1.3  # Giật cược đột biến -> dễ là bluff hơn
                elif dq == 1 or (dq == 0 and last_opp_bid.face_value > prev_bid.face_value):
                    history_bluff_modifier = 0.8  # Tăng cược tối thiểu -> cẩn thận/thật thà hơn
            else:
                # Đối thủ đi đầu tiên của vòng chơi
                total_dice = len(observation["my_dice"]) + observation["opponent_dice_count"]
                if last_opp_bid.quantity > total_dice / 3.0:
                    history_bluff_modifier = 1.2  # Mở màn hô quá to -> có độ khả nghi cao hơn

        # 2. Ngưỡng động cân bằng: cho phép giảm xuống tối thiểu 0.25 khi đối thủ cực kỳ trung thực (rate thấp)
        # để tránh AI quá khờ khạo, và tăng lên tối đa 0.65 khi đối thủ rất hay bluff.
        base_bluff_rate = self.get_best_bluff_rate(observation, current_bid)
        rate = base_bluff_rate * history_bluff_modifier
        dynamic_threshold = max(0.25, min(0.65, 0.3 + rate * 0.5))

        # Lưu thông số để hiển thị trên Web Dashboard
        self.last_base_rate = base_bluff_rate
        self.last_modifier = history_bluff_modifier
        self.last_threshold = dynamic_threshold

        # Logic ra quyết định
        my_dice = observation["my_dice"]
        counts = {f: count_matching_dice([my_dice], f) for f in range(1, 7)}

        def choose(act_obj: Action) -> Action:
            self.last_action_type = "tố cáo" if isinstance(act_obj, Challenge) else "cược"
            return act_obj

        # Trường hợp đi đầu vòng (chưa có ai cược)
        if current_bid is None:
            self.last_p_truth = None
            best_face = max(range(1, 7), key=lambda f: (counts[f], f))
            preferred_bids = [
                action for action in legal_actions
                if isinstance(action, Bid) and action.face_value == best_face
            ]
            if preferred_bids:
                return choose(min(preferred_bids, key=lambda b: b.quantity))

            all_bids = [action for action in legal_actions if isinstance(action, Bid)]
            if all_bids:
                return choose(min(all_bids, key=lambda b: b.quantity))
            return choose(legal_actions[0])

        # Trường hợp đối thủ đã cược
        Q = current_bid.quantity
        F = current_bid.face_value
        my_count = counts[F]
        k = Q - my_count
        n = observation["opponent_dice_count"]
        p = 1/6 if F == 1 else 1/3

        # Xác suất đối thủ nói thật
        P_truth = binomial_probability(k, n, p)
        self.last_p_truth = P_truth

        # Quyết định Challenge dựa trên ngưỡng động đã hiệu chỉnh
        if P_truth < dynamic_threshold and any(isinstance(a, Challenge) for a in legal_actions):
            return choose(Challenge())

        # Ngược lại, nâng cược
        sorted_faces = sorted(range(1, 7), key=lambda f: (counts[f], f), reverse=True)
        
        # BÓC LỘT: Nếu đối thủ hiếm khi challenge khi ta cược mặt lớn (>= 4), ưu tiên đưa mặt lớn lên đầu
        if self.our_bids_high_face >= 3:
            chal_rate = self.opp_challenges_high_face / self.our_bids_high_face
            if chal_rate < 0.25:
                sorted_faces = [f for f in sorted_faces if f >= 4] + [f for f in sorted_faces if f < 4]

        for face in sorted_faces:
            matching_bids = [
                action for action in legal_actions
                if isinstance(action, Bid) and action.face_value == face
            ]
            if matching_bids:
                return choose(min(matching_bids, key=lambda b: b.quantity))

        all_bids = [action for action in legal_actions if isinstance(action, Bid)]
        if all_bids:
            return choose(min(all_bids, key=lambda b: b.quantity))

        challenge_actions = [a for a in legal_actions if isinstance(a, Challenge)]
        if challenge_actions:
            return choose(challenge_actions[0])

        return choose(legal_actions[0])
