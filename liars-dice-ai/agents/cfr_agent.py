import json
import gzip
from typing import List, Dict, Optional
from agents.base_agent import Agent
from game.actions import Action, Bid, Challenge
from game.engine import count_matching_dice, get_legal_actions, apply_action
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
    # Chọn DEPTH 4 cho TRAIN NẶNG: vì nút thắt là data-starvation (mỗi infoset thăm quá
    # ít), dồn ngân sách vào SỐ VÒNG quan trọng hơn duyệt sâu. Depth 4 nhanh ~6× depth 5
    # ⇒ nhiều vòng hơn ⇒ mỗi infoset được thăm dày hơn ⇒ hội tụ tới WR ≥ 50% nhanh hơn.
    MAX_DEPTH = 4          # độ sâu tối đa khi duyệt cây
    # MỞ RỘNG ABSTRACTION: thêm các nước nâng cược +1/+2 (mặt hiện tại & wild) bên cạnh
    # bid tối thiểu mỗi mặt ⇒ chiến lược học được cả khả năng "thổi giá" mạnh hơn.
    MAX_ABSTRACT = 10      # số hành động đại diện tối đa (trước: 8)
    # GATING ĐỘ TIN CẬY: không gian infoset rất lớn (bộ xúc xắc chính xác) nên đa số
    # infoset chỉ được thăm vài lần ⇒ chiến lược học còn nhiễu, đánh dở hơn cả fallback.
    # Chỉ DÙNG chiến lược đã học khi infoset được thăm ĐỦ NHIỀU; chưa đủ thì fallback
    # sang ProbabilisticAgent (vốn ổn định) thay vì đè bằng nước học dở dang.
    MIN_VISITS = 10        # số lượt thăm tối thiểu để tin dùng chiến lược đã học

    def __init__(self, name: str = "CFRAgent"):
        super().__init__(name)
        # Bảng hối hận tích lũy (đã sàn hóa ≥ 0 theo CFR+): infoset -> {action: regret}
        self.regret_table: Dict[str, Dict[str, float]] = {}
        # Bảng chiến thuật tích lũy (trọng số tuyến tính): infoset -> {action: strategy_sum}
        self.strategy_table: Dict[str, Dict[str, float]] = {}
        # Số lượt thăm mỗi infoset (đo độ tin cậy để gating trong act())
        self.visit_counts: Dict[str, int] = {}

        # Tổng số vòng lặp self-play đã huấn luyện
        self.iterations_trained: int = 0
        # Chỉ số vòng lặp hiện tại (1-based) dùng làm trọng số linear averaging
        self._t: int = 0
        # Trạng thái ván self-play đang chạy (dùng cho train FULL-GAME): các vòng đấu
        # được lấy mẫu theo đúng phân phối số xúc xắc xuất hiện trong ván thật.
        self._game_state: Optional[GameState] = None

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

    def _make_infoset_key(self, my_dice, opp_count, current_bid) -> str:
        """Khóa infoset — DÙNG CHUNG cho cfr_search, rollout và act() để không lệch khóa.

        Gồm đúng những gì người chơi được phép biết: tay mình (đầy đủ), số xúc xắc đối
        thủ, cược đang đứng. (Đã thử bucket hóa tay theo "số viên khớp mặt cược" để giảm
        số infoset, nhưng làm vậy vứt mất thông tin các mặt khác ⇒ nâng cược mù ⇒ WR sụp.
        Trừu tượng hóa đúng cần làm ĐỒNG THỜI với trừu tượng hóa hành động — xem docs.)
        """
        bid_tuple = (current_bid.quantity, current_bid.face_value) if current_bid else None
        return f"{tuple(my_dice)} | {opp_count} | {bid_tuple}"

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

        # Nước nâng cược +1 và +2 số lượng cho mặt hiện tại hoặc wild (mở rộng abstraction
        # để học được cả các nước "thổi giá" mạnh, không chỉ bid tối thiểu an toàn).
        if state.current_bid:
            cur_face = state.current_bid.face_value
            for delta in (1, 2):
                next_qty = state.current_bid.quantity + delta
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
        infoset = self._make_infoset_key(my_dice, opp_count, state.current_bid)

        action_strs = [self._action_to_str(a) for a in abstract_actions]
        if infoset not in self.regret_table:
            self.regret_table[infoset] = {s: 0.0 for s in action_strs}
            self.strategy_table[infoset] = {s: 0.0 for s in action_strs}
        self.visit_counts[infoset] = self.visit_counts.get(infoset, 0) + 1
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
        """Huấn luyện qua tự chơi FULL-GAME.

        Trước đây mỗi vòng lặp bốc một ván với số xúc xắc NGẪU NHIÊN [1,5]×[1,5] — phân
        phối này lệch so với ván thật (luôn bắt đầu 5-5, chênh lệch nhỏ dần). Nay ta nuôi
        MỘT ván đầy đủ: mỗi vòng lặp cập nhật CFR cho subgame vòng hiện tại RỒI chơi hết
        vòng đó để số xúc xắc tiến triển đúng như ván thật; hết ván thì bắt đầu ván mới.
        Nhờ vậy CFR được huấn luyện trên đúng phân phối trạng thái của game đầy đủ.

        `iterations` = số VÒNG đấu được cập nhật (giữ cùng đơn vị với bản cũ).
        """
        import random
        for _ in range(iterations):
            self._t += 1

            # Bắt đầu ván mới nếu chưa có hoặc ván trước đã kết thúc
            if self._game_state is None or self._game_state.is_game_over():
                self._game_state = GameState(start_dice=5)
                self._game_state.current_player = random.randint(0, 1)

            st = self._game_state

            # 1) Cập nhật CFR cho subgame VÒNG hiện tại (trên một bản sao sạch)
            root = st.clone()
            root.current_bid = None
            root.history = []
            self.cfr_search(root, 1.0, 1.0, depth=0)

            # 2) Chơi hết vòng hiện tại để ván tiến triển (trừ xúc xắc người thua, sang vòng sau)
            self._play_one_round(st)

        self.iterations_trained += iterations

    def _rollout_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """Chọn hành động khi 'chơi out' một vòng trong self-play full-game.

        Dùng chiến lược regret-matching hiện tại nếu đã học infoset này, ngược lại chọn
        ngẫu nhiên trong tập hành động đại diện — chỉ nhằm sinh quỹ đạo ván thực tế.
        """
        import random
        abstract = self._get_abstract_actions(state, legal_actions)
        if not abstract:
            return random.choice(legal_actions)

        active = state.current_player
        my_dice = tuple(state.hands[active])
        opp_count = state.dice_counts[1 - active]
        infoset = self._make_infoset_key(my_dice, opp_count, state.current_bid)

        regrets = self.regret_table.get(infoset)
        action_strs = [self._action_to_str(a) for a in abstract]
        if regrets:
            pos = {s: max(0.0, regrets.get(s, 0.0)) for s in action_strs}
            total = sum(pos.values())
            r = random.random()
            cumulative = 0.0
            for a, s in zip(abstract, action_strs):
                prob = pos[s] / total if total > 0 else 1.0 / len(abstract)
                cumulative += prob
                if r <= cumulative:
                    return a
        return random.choice(abstract)

    def _play_one_round(self, state: GameState):
        """Chơi (in-place) một vòng từ trạng thái bid=None cho đến khi có Challenge.

        engine.apply_action tự xử lý trừ 1 xúc xắc người thua và reset_round (bốc lại,
        người thua đi trước) nếu ván chưa kết thúc — nên sau hàm này state đã ở đầu vòng
        kế tiếp hoặc đã game-over.
        """
        guard = 0
        while not state.is_game_over():
            legal = get_legal_actions(state)
            if not legal:
                break
            action = self._rollout_action(state, legal)
            outcome = apply_action(state, action)
            if outcome["type"] == "challenge":
                break
            guard += 1
            if guard > 200:  # bảo hiểm chống vòng lặp vô hạn
                break

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
        infoset = self._make_infoset_key(my_dice, opp_count, curr_bid)

        # GATING: chỉ tin dùng chiến lược đã học nếu infoset được thăm đủ nhiều; ngược lại
        # rơi xuống fallback (tránh đè nước tốt của fallback bằng nước học dở dang).
        if infoset in self.strategy_table and self.visit_counts.get(infoset, 0) >= self.MIN_VISITS:
            strat_sums = self.strategy_table[infoset]
            legal_map = {self._action_to_str(a): a for a in legal_actions}

            valid = {}
            for s, a in legal_map.items():
                if s in strat_sums:
                    valid[a] = max(0.0, strat_sums[s])

            # ARGMAX: trước đối thủ CỐ ĐỊNH (Random/Prob/Bayes), best-response thường là
            # nước đi CỨNG. Chọn hành động có xác suất trung bình cao nhất thay vì sample
            # chiến lược hỗn hợp ⇒ sắc bén hơn, không tự "làm mềm" nước đi tốt.
            if valid and sum(valid.values()) > 0:
                return max(valid, key=valid.get)

        # Fallback an toàn: ProbabilisticAgent
        if self.fallback_agent is None:
            from agents.probabilistic_agent import ProbabilisticAgent
            self.fallback_agent = ProbabilisticAgent(name="CFR_Fallback", threshold=0.5)
        return self.fallback_agent.act(observation, legal_actions)

    # ------------------------------------------------------------- persistence
    def save_weights(self, filepath: str):
        data = {
            "regret_table": self.regret_table,
            "strategy_table": self.strategy_table,
            "visit_counts": self.visit_counts,
        }
        # File .gz được nén gzip (nhỏ ~7× ⇒ commit được); ngược lại ghi JSON thường.
        opener = gzip.open if filepath.endswith(".gz") else open
        with opener(filepath, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_weights(self, filepath: str):
        # Tự nhận diện file nén gzip qua đuôi .gz để nạp được cả bản nén lẫn bản thường.
        opener = gzip.open if filepath.endswith(".gz") else open
        with opener(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)
        self.regret_table = data.get("regret_table", {})
        self.strategy_table = data.get("strategy_table", {})
        self.visit_counts = data.get("visit_counts", {})
