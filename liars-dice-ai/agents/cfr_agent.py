import json
from typing import List, Dict, Optional
from agents.base_agent import Agent
from game.actions import Action, Bid, Challenge
from game.engine import count_matching_dice, get_legal_actions, apply_action
from game.state import GameState

class CFRAgent(Agent):
    """Agent LẤY CẢM HỨNG từ Counterfactual Regret Minimization (CFR) qua tự chơi
    (self-play). Đây là biến thể XẤP XỈ, không đảm bảo hội tụ Nash chính xác vì:
      (1) giới hạn độ sâu (max_depth=4) và ước lượng nút lá bằng heuristic;
      (2) nút lá đánh giá với thông tin hoàn hảo (biết cả xúc xắc đối thủ trong
          mỗi ván self-play), thay vì lấy kỳ vọng đúng trên thông tin ẩn.
    Mục tiêu thực tế: học chiến lược HỖN HỢP ít bị khai thác (low-exploitability)
    trong trò chơi đã trừu tượng hóa. Liên quan Bài 4 (đối kháng/lý thuyết trò chơi)
    và Bài 8-9 (học qua tương tác / regret minimization).
    Khi gặp trạng thái chưa học, tự động Fallback sang ProbabilisticAgent.
    """
    def __init__(self, name: str = "CFRAgent"):
        super().__init__(name)
        # Bảng hối hận tích lũy: infoset_key -> {action_str: regret_value}
        self.regret_table: Dict[str, Dict[str, float]] = {}
        # Bảng chiến thuật tích lũy: infoset_key -> {action_str: strategy_sum}
        self.strategy_table: Dict[str, Dict[str, float]] = {}

        # Tổng số vòng lặp self-play đã huấn luyện (để tính regret trung bình)
        self.iterations_trained: int = 0

        # Lớp fallback an toàn khi gặp trạng thái lạ
        self.fallback_agent: Optional[Agent] = None

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
        """Trừu tượng hóa danh sách hành động hợp lệ để thu hẹp không gian tìm kiếm,
        giúp thuật toán duyệt cây nhanh hơn mà không làm mất tính tổng quát.
        """
        # Luôn giữ hành động tố cáo (Challenge)
        abstract = [a for a in legal_actions if isinstance(a, Challenge)]
        
        bids = [a for a in legal_actions if isinstance(a, Bid)]
        if not bids:
            return abstract
            
        # Thêm bid tối thiểu (an toàn nhất) cho mỗi mặt xúc xắc (1-6)
        min_bids_by_face = {}
        for b in bids:
            if b.face_value not in min_bids_by_face or b.quantity < min_bids_by_face[b.face_value].quantity:
                min_bids_by_face[b.face_value] = b
                
        for face in sorted(min_bids_by_face.keys()):
            abstract.append(min_bids_by_face[face])
            
        # Thêm bid tăng số lượng lên đúng 1 viên cho mặt hiện tại hoặc mặt 1 (wild)
        if state.current_bid:
            current_face = state.current_bid.face_value
            next_qty = state.current_bid.quantity + 1
            for b in bids:
                if b.quantity == next_qty and b.face_value in (current_face, 1):
                    abstract.append(b)
                    
        # Loại bỏ trùng lặp
        unique_abstract = []
        seen = set()
        for a in abstract:
            a_str = self._action_to_str(a)
            if a_str not in seen:
                seen.add(a_str)
                unique_abstract.append(a)
                
        # Giới hạn tối đa 6 hành động đại diện tốt nhất
        return unique_abstract[:6]

    def _estimate_utility(self, state: GameState) -> float:
        """Ước lượng giá trị phần thưởng (utility) tại nút lá độ sâu giới hạn.
        Trả về +1.0 nếu Player 0 thắng, và -1.0 nếu Player 0 thua.
        """
        if state.current_bid is None:
            return 0.0
            
        bid = state.current_bid
        actual_count = count_matching_dice(state.hands, bid.face_value)
        # Người cược là người vừa đi (1 - current_player)
        bidder = 1 - state.current_player
        
        if actual_count >= bid.quantity:
            # Cược đúng -> bidder thắng. Nếu bidder là Player 0 -> +1.0, ngược lại -1.0
            return 1.0 if bidder == 0 else -1.0
        else:
            # Cược sai -> bidder thua. Nếu bidder là Player 0 -> -1.0, ngược lại +1.0
            return -1.0 if bidder == 0 else 1.0

    def cfr_search(self, state: GameState, p0: float, p1: float, depth: int) -> float:
        """Duyệt đệ quy cây game CFR để tính toán Regret và Strategy.
        Trả về utility của nút từ góc nhìn của Player 0.
        """
        # Giới hạn độ sâu tìm kiếm để tránh bùng nổ tổ hợp trạng thái
        max_depth = 4
        if depth >= max_depth:
            return self._estimate_utility(state)
            
        active_player = state.current_player
        legal_actions = get_legal_actions(state)
        
        if not legal_actions:
            return self._estimate_utility(state)
            
        # Thu hẹp hành động
        abstract_actions = self._get_abstract_actions(state, legal_actions)
        
        # Tạo khóa infoset
        my_dice = tuple(state.hands[active_player])
        opp_dice_count = state.dice_counts[1 - active_player]
        curr_bid_tuple = (state.current_bid.quantity, state.current_bid.face_value) if state.current_bid else None
        infoset_key = f"{my_dice} | {opp_dice_count} | {curr_bid_tuple}"
        
        # Khởi tạo bảng nếu chưa có
        if infoset_key not in self.regret_table:
            self.regret_table[infoset_key] = {self._action_to_str(a): 0.0 for a in abstract_actions}
        if infoset_key not in self.strategy_table:
            self.strategy_table[infoset_key] = {self._action_to_str(a): 0.0 for a in abstract_actions}
            
        regrets = self.regret_table[infoset_key]
        
        # 1. Regret Matching
        positive_regrets = {a_str: max(0.0, regrets.get(a_str, 0.0)) for a_str in regrets}
        sum_pos_regrets = sum(positive_regrets.values())
        
        strategy = {}
        for a in abstract_actions:
            a_str = self._action_to_str(a)
            if sum_pos_regrets > 0:
                strategy[a] = positive_regrets[a_str] / sum_pos_regrets
            else:
                strategy[a] = 1.0 / len(abstract_actions)
                
        # 2. Tích lũy strategy_sum
        p_active = p0 if active_player == 0 else p1
        strat_sums = self.strategy_table[infoset_key]
        for a, prob in strategy.items():
            a_str = self._action_to_str(a)
            strat_sums[a_str] = strat_sums.get(a_str, 0.0) + p_active * prob
            
        # 3. Tính expected utility cho từng hành động
        action_utilities = {}
        for a in abstract_actions:
            if isinstance(a, Challenge):
                actual_count = count_matching_dice(state.hands, state.current_bid.face_value)
                bidder = 1 - active_player
                if actual_count >= state.current_bid.quantity:
                    winner = bidder
                else:
                    winner = active_player
                action_utilities[a] = 1.0 if winner == 0 else -1.0
            else:
                # Tạo bản sao giả lập hành động Bid
                next_state = state.clone()
                next_state.current_bid = a
                next_state.history.append((active_player, a))
                next_state.current_player = 1 - active_player
                
                if active_player == 0:
                    action_utilities[a] = self.cfr_search(next_state, p0 * strategy[a], p1, depth + 1)
                else:
                    action_utilities[a] = self.cfr_search(next_state, p0, p1 * strategy[a], depth + 1)
                    
        # 4. Tính expected utility tổng thể của infoset
        U = sum(strategy[a] * action_utilities[a] for a in abstract_actions)
        
        # 5. Cập nhật regret tích lũy
        p_opp = p1 if active_player == 0 else p0
        for a in abstract_actions:
            a_str = self._action_to_str(a)
            u_a = action_utilities[a]
            # Hối hận = Utility hành động - Utility trung bình của infoset (đảo chiều đối với Player 1)
            regret_val = (u_a - U) if active_player == 0 else (U - u_a)
            regrets[a_str] = regrets.get(a_str, 0.0) + p_opp * regret_val
            
        return U

    def train(self, iterations: int = 1000):
        """Huấn luyện Agent thông qua tự chơi (Self-play) trong số vòng lặp chỉ định."""
        import random
        for _ in range(iterations):
            # Lắc xúc xắc ngẫu nhiên với số lượng xúc xắc từ [1, 5] cho mỗi người chơi
            d0 = random.randint(1, 5)
            d1 = random.randint(1, 5)
            state = GameState(start_dice=5)
            state.dice_counts = [d0, d1]
            state.roll_all_dice()
            state.current_bid = None
            state.current_player = random.randint(0, 1)
            state.history = []

            self.cfr_search(state, 1.0, 1.0, depth=0)

        # Ghi nhận tổng số vòng lặp để tính regret trung bình (chặn trên exploitability)
        self.iterations_trained += iterations

    def average_regret(self) -> float:
        """Trả về regret dương trung bình trên mỗi infoset, chuẩn hóa theo số vòng lặp.

        Theo lý thuyết CFR, regret trung bình R^T/T chặn trên khoảng cách tới cân
        bằng (exploitability) và phải tiến về 0 khi số vòng lặp T tăng. Đây là chỉ
        số hội tụ thực nghiệm: giá trị càng nhỏ ⇒ chiến lược càng khó bị khai thác.
        """
        if self.iterations_trained == 0 or not self.regret_table:
            return float("inf")

        total = 0.0
        for regrets in self.regret_table.values():
            # Regret dương lớn nhất tại infoset này (phần "đáng tiếc" còn lại)
            total += max((max(0.0, r) for r in regrets.values()), default=0.0)

        return total / (self.iterations_trained * len(self.regret_table))

    def act(self, observation: dict, legal_actions: List[Action]) -> Action:
        my_dice = tuple(observation["my_dice"])
        opp_dice_count = observation["opponent_dice_count"]
        curr_bid = observation["current_bid"]
        curr_bid_tuple = (curr_bid.quantity, curr_bid.face_value) if curr_bid else None
        
        infoset_key = f"{my_dice} | {opp_dice_count} | {curr_bid_tuple}"
        
        # Nếu có thông tin chiến thuật đã được huấn luyện cho trạng thái này
        if infoset_key in self.strategy_table:
            strat_sums = self.strategy_table[infoset_key]
            legal_action_strs = {self._action_to_str(a): a for a in legal_actions}
            
            # Lấy xác suất của các hành động hợp lệ hiện tại
            valid_probs = {}
            for a_str, a in legal_action_strs.items():
                if a_str in strat_sums:
                    valid_probs[a] = max(0.0, strat_sums[a_str])
                    
            sum_valid = sum(valid_probs.values())
            if sum_valid > 0:
                # Chuẩn hóa phân phối xác suất
                normalized_probs = {a: p / sum_valid for a, p in valid_probs.items()}
                
                # Lấy ngẫu nhiên hành động theo phân phối xác suất (Mixed Strategy)
                import random
                r = random.random()
                cumulative = 0.0
                for a, prob in normalized_probs.items():
                    cumulative += prob
                    if r <= cumulative:
                        return a
                        
        # Hậu phòng an toàn (Fallback): chuyển sang dùng ProbabilisticAgent
        if self.fallback_agent is None:
            from agents.probabilistic_agent import ProbabilisticAgent
            self.fallback_agent = ProbabilisticAgent(name="CFR_Fallback", threshold=0.5)
        return self.fallback_agent.act(observation, legal_actions)

    def save_weights(self, filepath: str):
        """Lưu trọng số của regret_table và strategy_table ra file JSON."""
        data = {
            "regret_table": self.regret_table,
            "strategy_table": self.strategy_table
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_weights(self, filepath: str):
        """Tải trọng số đã lưu từ file JSON vào Agent."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.regret_table = data.get("regret_table", {})
        self.strategy_table = data.get("strategy_table", {})
