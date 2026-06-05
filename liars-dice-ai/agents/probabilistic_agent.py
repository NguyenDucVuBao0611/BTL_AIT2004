from typing import List, Optional
from agents.base_agent import Agent
from game.actions import Action, Bid, Challenge
from game.engine import count_matching_dice
from core.probability import binomial_probability

class ProbabilisticAgent(Agent):
    """Agent sử dụng xác suất tích lũy (Expectimax Search Model - Chance Nodes) 
    để đưa ra quyết định Challenge hoặc nâng cược tối ưu kỳ vọng.
    """
    def __init__(self, name: str = "ProbabilisticAgent", threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold

    def act(self, observation: dict, legal_actions: List[Action]) -> Action:
        my_dice = observation["my_dice"]
        opponent_dice_count = observation["opponent_dice_count"]
        current_bid = observation["current_bid"]

        # Đếm số lượng xúc xắc trong tay của bản thân cho từng mặt (1-6)
        counts = {f: count_matching_dice([my_dice], f) for f in range(1, 7)}

        # 1. Trường hợp đi đầu vòng (chưa có ai cược)
        if current_bid is None:
            # Chọn mặt xúc xắc xuất hiện nhiều nhất trên tay (ưu tiên mặt lớn hơn nếu hòa)
            best_face = max(range(1, 7), key=lambda f: (counts[f], f))
            
            # Lọc các nước cược cho mặt xúc xắc này và chọn nước cược có số lượng nhỏ nhất (an toàn nhất)
            preferred_bids = [
                action for action in legal_actions 
                if isinstance(action, Bid) and action.face_value == best_face
            ]
            if preferred_bids:
                return min(preferred_bids, key=lambda b: b.quantity)
            
            # Hậu phòng: chọn nước cược có số lượng nhỏ nhất trong tất cả các nước cược hợp lệ
            all_bids = [action for action in legal_actions if isinstance(action, Bid)]
            if all_bids:
                return min(all_bids, key=lambda b: b.quantity)
            
            # Nếu không có hành động cược nào hợp lệ (không xảy ra theo luật)
            return legal_actions[0]

        # 2. Trường hợp đối thủ đã cược trước đó
        # Cược hiện tại của đối thủ là Q viên mặt F
        Q = current_bid.quantity
        F = current_bid.face_value

        # Đếm số lượng xúc xắc khớp mặt F trên tay ta (đã tính cả mặt 1 là wild nếu F != 1)
        my_count = counts[F]

        # Số lượng mặt F đối thủ cần có tối thiểu để cược đúng
        k = Q - my_count

        # Số xúc xắc ẩn của đối thủ
        n = opponent_dice_count

        # Xác suất một viên xúc xắc đơn lẻ khớp với mặt F
        # Nếu F == 1 thì p = 1/6 (mặt 1 chỉ tính là chính nó)
        # Nếu F != 1 thì p = 1/3 (mặt F hoặc mặt 1 đều được tính, 2/6 = 1/3)
        p = 1/6 if F == 1 else 1/3

        # Tính xác suất đối thủ cược đúng (nói thật)
        P_truth = binomial_probability(k, n, p)

        # Ra quyết định Challenge nếu xác suất đối thủ nói thật nhỏ hơn ngưỡng threshold
        if P_truth < self.threshold and any(isinstance(a, Challenge) for a in legal_actions):
            return Challenge()

        # Ngược lại, ta chọn nâng cược tối ưu kỳ vọng: ưu tiên mặt ta giữ nhiều nhất
        # Sắp xếp các mặt xúc xắc theo thứ tự ưu tiên giảm dần về số lượng ta nắm giữ
        sorted_faces = sorted(range(1, 7), key=lambda f: (counts[f], f), reverse=True)

        for face in sorted_faces:
            matching_bids = [
                action for action in legal_actions 
                if isinstance(action, Bid) and action.face_value == face
            ]
            if matching_bids:
                # Chọn nước cược có số lượng nhỏ nhất của mặt đó để giữ an toàn tối đa
                return min(matching_bids, key=lambda b: b.quantity)

        # Hậu phòng nếu không tìm thấy nước cược theo mặt ưu tiên
        all_bids = [action for action in legal_actions if isinstance(action, Bid)]
        if all_bids:
            return min(all_bids, key=lambda b: b.quantity)

        # Nếu không còn nước cược nào hợp lệ, bắt buộc phải Challenge
        challenge_actions = [a for a in legal_actions if isinstance(a, Challenge)]
        if challenge_actions:
            return challenge_actions[0]

        # Mặc định trả về nước đi hợp lệ đầu tiên nếu mọi logic trên không tìm được hành động (an toàn)
        return legal_actions[0]
