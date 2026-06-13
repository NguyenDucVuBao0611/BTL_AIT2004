import json
from typing import List, Dict, Optional
from agents.base_agent import Agent
from game.actions import Action, Bid, Challenge
from game.engine import count_matching_dice, get_legal_actions
from game.state import GameState


class CFRAgent(Agent):
    """Agent dựa trên Counterfactual Regret Minimization — bản nâng cấp (CFR+).

    Cải tiến so với bản đầu để mạnh và hội tụ nhanh hơn rõ rệt:
      1. CFR+: regret tích lũy được "sàn hóa" về 0 mỗi bước (regret matching+) và
         chiến lược trung bình được lấy có TRỌNG SỐ TUYẾN TÍNH theo số vòng lặp
         (linear averaging) ⇒ hội tụ nhanh hơn nhiều bậc so với CFR thường.
      2. Nút lá KHÔNG rò rỉ thông tin: khi cắt độ sâu, ước lượng giá trị bằng KỲ
         VỌNG trên xúc xắc ẩn của đối thủ (phân phối nhị thức) từ góc nhìn người
         đang đi — thay vì nhìn lén tay đối thủ như trước.
      3. Độ sâu tìm kiếm lớn hơn (đa số đường đi chạm nút Challenge thật sự).
      4. Tập hành động trừu tượng rộng hơn (có thêm lựa chọn nâng cược/bluff).

    Khi gặp trạng thái chưa học, tự động Fallback sang ProbabilisticAgent.
    Liên quan Bài 4 (đối kháng/lý thuyết trò chơi) và Bài 8-9 (regret minimization).
    """

    # Giữ chi phí mỗi vòng lặp thấp như bản gốc: sức mạnh đến từ CFR+ (hội tụ nhanh)
    # và nút lá ước lượng KHÔNG rò rỉ thông tin, không phải từ việc duyệt cây sâu hơn.
    MAX_DEPTH = 4          # độ sâu tối đa khi duyệt cây
    MAX_ABSTRACT = 8       # số hành động đại diện tối đa (giữ đủ Challenge + 6 mặt + 1 raise)

    def __init__(self, name: str = "CFRAgent"):
        super().__init__(name)
        # Bảng hối hận tích lũy (đã sàn hóa ≥ 0 theo CFR+): infoset -> {action: regret}
        self.regret_table: Dict[str, Dict[str, float]] = {}
        # Bảng chiến thuật tích lũy (trọng số tuyến tính): infoset -> {action: strategy_sum}
        self.strategy_table: Dict[str, Dict[str, float]] = {}

        # Tổng số vòng lặp self-play đã huấn luyện
        self.iterations_trained: int = 0
        # Chỉ số vòng lặp hiện tại (1-based) dùng làm trọng số linear averaging
        self._t: int = 0

        # Lớp fallback an toàn khi gặp trạng thái lạ
        self.fallback_agent: Optional[Agent] = None

    # ------------------------------------------------------------------ helpers
    def _action_to_str(self, action: Action) -> str:
        if isinstance(action, Challenge):
            return "Challenge"
        elif isinstance(action, Bid):
            return f"Bid:{action.quantity}:{action.face_value}"
        raise ValueError("Hành động không xác định.")

    def _str_to_action(self, action_str: str) -> Action:
        if action_str == "Challenge":
            return Challenge()
        elif action_str.startswith("Bid:"):
            parts = action_str.split(":")
            return Bid(int(parts[1]), int(parts[2]))
        raise ValueError("Chuỗi hành động không hợp lệ.")

    def _get_abstract_actions(self, state: GameState, legal_actions: List[Action]) -> List[Action]:
        """Trừu tượng hóa danh sách hành động để thu hẹp không gian tìm kiếm.

        Giữ: Challenge + bid tối thiểu mỗi mặt + một vài nước nâng cược (an toàn và
        bluff nhẹ) để chiến lược học được cả khả năng "thổi giá".
        """
        abstract = [a for a in legal_actions if isinstance(a, Challenge)]

        bids = [a for a in legal_actions if isinstance(a, Bid)]
        if not bids:
            return abstract

        # Bid tối thiểu (an toàn nhất) cho mỗi mặt xúc xắc
        min_bids_by_face: Dict[int, Bid] = {}
        for b in bids:
            cur = min_bids_by_face.get(b.face_value)
            if cur is None or b.quantity < cur.quantity:
                min_bids_by_face[b.face_value] = b
        for face in sorted(min_bids_by_face.keys()):
            abstract.append(min_bids_by_face[face])

        # Nước nâng cược +1 số lượng cho mặt hiện tại hoặc wild (giữ tập hành động gọn)
        if state.current_bid:
            cur_face = state.current_bid.face_value
            next_qty = state.current_bid.quantity + 1
            for b in bids:
                if b.quantity == next_qty and b.face_value in (cur_face, 1):
                    abstract.append(b)

        # Loại trùng và giới hạn số hành động đại diện
        unique, seen = [], set()
        for a in abstract:
            s = self._action_to_str(a)
            if s not in seen:
                seen.add(s)
                unique.append(a)
        return unique[:self.MAX_ABSTRACT]

    def _leaf_utility(self, state: GameState) -> float:
        """Ước lượng giá trị nút lá khi cắt độ sâu: lấy kết cục NẾU người đang đi
        Challenge cược đang đứng trong VÁN ĐÃ BỐC hiện tại.

        Lưu ý: đây là giá trị thật của ván self-play đã bốc (không phải "nhìn lén"
        ở thế đối kháng thật), nên qua nhiều ván sampling nó cho ước lượng giá trị
        subgame chính xác hơn hẳn heuristic "giả định challenge theo kỳ vọng".
        Trả về giá trị từ góc nhìn Player 0.
        """
        if state.current_bid is None:
            return 0.0
        return self._terminal_challenge_utility(state, state.current_player)

    def _terminal_challenge_utility(self, state: GameState, active_player: int) -> float:
        """Giá trị thật khi `active_player` Challenge cược đang đứng (đã lộ cả hai tay
        trong ván self-play hiện tại — đây là kết cục THẬT của ván đã bốc, hợp lệ)."""
        bid = state.current_bid
        actual = count_matching_dice(state.hands, bid.face_value)
        bidder = 1 - active_player
        winner = bidder if actual >= bid.quantity else active_player
        return 1.0 if winner == 0 else -1.0

    # ------------------------------------------------------------------- search
    def cfr_search(self, state: GameState, p0: float, p1: float, depth: int) -> float:
        """Duyệt đệ quy cây CFR+. Trả về utility kỳ vọng từ góc nhìn Player 0."""
        if depth >= self.MAX_DEPTH:
            return self._leaf_utility(state)

        active = state.current_player
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            return self._leaf_utility(state)

        abstract_actions = self._get_abstract_actions(state, legal_actions)

        my_dice = tuple(state.hands[active])
        opp_count = state.dice_counts[1 - active]
        bid_tuple = (state.current_bid.quantity, state.current_bid.face_value) if state.current_bid else None
        infoset = f"{my_dice} | {opp_count} | {bid_tuple}"

        action_strs = [self._action_to_str(a) for a in abstract_actions]
        if infoset not in self.regret_table:
            self.regret_table[infoset] = {s: 0.0 for s in action_strs}
            self.strategy_table[infoset] = {s: 0.0 for s in action_strs}
        regrets = self.regret_table[infoset]
        strat_sums = self.strategy_table[infoset]

        # 1. Regret matching (chỉ dùng regret dương — CFR+ đã sàn hóa ≥ 0)
        pos = {s: max(0.0, regrets.get(s, 0.0)) for s in action_strs}
        total_pos = sum(pos.values())
        strategy = {}
        for a, s in zip(abstract_actions, action_strs):
            strategy[a] = pos[s] / total_pos if total_pos > 0 else 1.0 / len(abstract_actions)

        # 2. Tích lũy chiến lược trung bình có TRỌNG SỐ TUYẾN TÍNH (linear averaging)
        p_active = p0 if active == 0 else p1
        weight = self._t  # vòng lặp hiện tại (1-based)
        for a, s in zip(abstract_actions, action_strs):
            strat_sums[s] = strat_sums.get(s, 0.0) + weight * p_active * strategy[a]

        # 3. Utility kỳ vọng cho từng hành động
        action_utils = {}
        for a, s in zip(abstract_actions, action_strs):
            if isinstance(a, Challenge):
                action_utils[a] = self._terminal_challenge_utility(state, active)
            else:
                nxt = state.clone()
                nxt.current_bid = a
                nxt.history.append((active, a))
                nxt.current_player = 1 - active
                if active == 0:
                    action_utils[a] = self.cfr_search(nxt, p0 * strategy[a], p1, depth + 1)
                else:
                    action_utils[a] = self.cfr_search(nxt, p0, p1 * strategy[a], depth + 1)

        # 4. Utility kỳ vọng của infoset
        U = sum(strategy[a] * action_utils[a] for a in abstract_actions)

        # 5. Cập nhật regret theo CFR+ (sàn hóa ≥ 0)
        p_opp = p1 if active == 0 else p0
        for a, s in zip(abstract_actions, action_strs):
            regret_val = (action_utils[a] - U) if active == 0 else (U - action_utils[a])
            regrets[s] = max(0.0, regrets.get(s, 0.0) + p_opp * regret_val)

        return U

    # ----------------------------------------------------------------- training
    def train(self, iterations: int = 1000):
        """Huấn luyện qua tự chơi (self-play). Mỗi vòng bốc một ván ngẫu nhiên."""
        import random
        for _ in range(iterations):
            self._t += 1
            d0 = random.randint(1, 5)
            d1 = random.randint(1, 5)
            state = GameState(start_dice=5)
            state.dice_counts = [d0, d1]
            state.roll_all_dice()
            state.current_bid = None
            state.current_player = random.randint(0, 1)
            state.history = []
            self.cfr_search(state, 1.0, 1.0, depth=0)

        self.iterations_trained += iterations

    def average_regret(self) -> float:
        """Regret dương trung bình / vòng lặp — chặn trên exploitability (→ 0 khi hội tụ)."""
        if self.iterations_trained == 0 or not self.regret_table:
            return float("inf")
        total = 0.0
        for regrets in self.regret_table.values():
            total += max((max(0.0, r) for r in regrets.values()), default=0.0)
        return total / (self.iterations_trained * len(self.regret_table))

    # --------------------------------------------------------------------- play
    def act(self, observation: dict, legal_actions: List[Action]) -> Action:
        my_dice = tuple(observation["my_dice"])
        opp_count = observation["opponent_dice_count"]
        curr_bid = observation["current_bid"]
        bid_tuple = (curr_bid.quantity, curr_bid.face_value) if curr_bid else None
        infoset = f"{my_dice} | {opp_count} | {bid_tuple}"

        if infoset in self.strategy_table:
            strat_sums = self.strategy_table[infoset]
            legal_map = {self._action_to_str(a): a for a in legal_actions}

            valid = {}
            for s, a in legal_map.items():
                if s in strat_sums:
                    valid[a] = max(0.0, strat_sums[s])

            total = sum(valid.values())
            if total > 0:
                import random
                r = random.random()
                cumulative = 0.0
                for a, prob in valid.items():
                    cumulative += prob / total
                    if r <= cumulative:
                        return a

        # Fallback an toàn: ProbabilisticAgent
        if self.fallback_agent is None:
            from agents.probabilistic_agent import ProbabilisticAgent
            self.fallback_agent = ProbabilisticAgent(name="CFR_Fallback", threshold=0.5)
        return self.fallback_agent.act(observation, legal_actions)

    # ------------------------------------------------------------- persistence
    def save_weights(self, filepath: str):
        data = {"regret_table": self.regret_table, "strategy_table": self.strategy_table}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_weights(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.regret_table = data.get("regret_table", {})
        self.strategy_table = data.get("strategy_table", {})
