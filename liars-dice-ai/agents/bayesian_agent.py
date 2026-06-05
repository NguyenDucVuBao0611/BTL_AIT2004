from typing import List, Optional
from agents.base_agent import Agent
from game.actions import Action, Bid, Challenge
from game.engine import count_matching_dice
from core.probability import binomial_probability

class BayesianAgent(Agent):
    """BayesianAgent áp dụng mô hình Naive Bayes để theo dõi và cập nhật niềm tin 
    về xu hướng nói dối (bluff) của đối thủ trực tuyến (online) qua các vòng đấu.
    Từ đó tính toán ngưỡng quyết định động (dynamic_threshold) để bắt bài đối thủ.
    """
    def __init__(self, name: str = "BayesianAgent"):
        super().__init__(name)
        # ID của ta và đối thủ
        self.player_id: Optional[int] = None
        self.opponent_id: Optional[int] = None
        
        # Thống kê suy luận Bayes tích lũy trong trận đấu
        self.bluff_opportunities: int = 0
        self.actual_bluffs: int = 0
        
        # Hàng đợi lưu các quan sát thu được trước khi xác định được player_id
        self.pending_observations: List[tuple] = []
        
        # Danh sách cược của đối thủ trong vòng này để xử lý
        self.observed_bids: List[dict] = []
        
        # Lưu vết thông tin lượt Challenge vừa qua để kiểm tra kết quả
        self.last_challenge: Optional[dict] = None
        
        # Lưu vết số xúc xắc gần nhất để phát hiện mất xúc xắc
        self.last_dice_counts: Optional[dict] = None
        
        # Xúc xắc hiện tại của ta
        self.my_dice: Optional[List[int]] = None
        
        # Trạng thái vòng đấu hiện tại
        self.is_current_round_active: bool = False

    def reset(self):
        """Reset trạng thái nội bộ khi bắt đầu một game đấu mới."""
        self.player_id = None
        self.opponent_id = None
        self.bluff_opportunities = 0
        self.actual_bluffs = 0
        self.pending_observations.clear()
        self.observed_bids.clear()
        self.last_challenge = None
        self.last_dice_counts = None
        self.my_dice = None
        self.is_current_round_active = False

    def observe(self, action: Action, acting_player: int):
        """Theo dõi hành động của người chơi để thu thập dữ liệu huấn luyện online."""
        if self.player_id is None:
            self.pending_observations.append((action, acting_player))
            return
        self._process_observation(action, acting_player)

    def _process_observation(self, action: Action, acting_player: int):
        # 1. Nếu đối thủ thực hiện Bid
        if isinstance(action, Bid) and acting_player == self.opponent_id:
            opp_dice = 5
            if self.last_dice_counts is not None:
                opp_dice = self.last_dice_counts[self.opponent_id]
                
            bid_info = {
                "bid": action,
                "opponent_dice_count": opp_dice,
                "processed": False
            }
            self.observed_bids.append(bid_info)
            
            # Nếu ta đã có thông tin xúc xắc của vòng này và vòng đấu đang hoạt động
            if self.my_dice is not None and self.is_current_round_active:
                self._process_single_bid(bid_info)
                
        # 2. Nếu có hành động Challenge
        elif isinstance(action, Challenge):
            self.is_current_round_active = False
            
            # Người cược bị tố cáo là 1 - acting_player
            bidder = 1 - acting_player
            
            self.last_challenge = {
                "bidder": bidder,
                "challenger": acting_player
            }
            
            # Xóa các cược của vòng này để chuẩn bị cho vòng tiếp theo
            self.observed_bids.clear()

    def _process_single_bid(self, bid_info: dict):
        bid = bid_info["bid"]
        opp_dice_count = bid_info["opponent_dice_count"]
        
        # Số lượng mặt cược trên tay ta (đã tính wild 1 nếu cược mặt 2-6)
        my_count = count_matching_dice([self.my_dice], bid.face_value)
        
        # Xác suất một viên đơn lẻ khớp
        p_single = 1/6 if bid.face_value == 1 else 1/3
        
        # Số lượng cược kỳ vọng an toàn E
        E = my_count + p_single * opp_dice_count
        
        # Nếu cược vượt quá kỳ vọng, đây là cơ hội nói dối (Bluff Opportunity)
        if bid.quantity > E:
            self.bluff_opportunities += 1
            
        bid_info["processed"] = True

    def act(self, observation: dict, legal_actions: List[Action]) -> Action:
        # Khởi tạo player_id và opponent_id nếu đi đầu tiên
        if self.player_id is None:
            self.player_id = observation["player_id"]
            self.opponent_id = 1 - self.player_id
            
            # Xử lý các quan sát pending thu thập trước đó
            for action, acting_player in self.pending_observations:
                self._process_observation(action, acting_player)
            self.pending_observations.clear()

        # Đánh dấu vòng đấu hiện tại bắt đầu hoạt động
        self.is_current_round_active = True
        self.my_dice = observation["my_dice"]

        # Phát hiện kết quả của lượt Challenge trước đó để cập nhật actual_bluffs
        if self.last_dice_counts is not None and self.last_challenge is not None:
            old_opp_dice = self.last_dice_counts[self.opponent_id]
            new_opp_dice = observation["opponent_dice_count"]
            
            # Nếu đối thủ là người cược (bidder) và bị mất xúc xắc sau challenge
            if new_opp_dice < old_opp_dice and self.last_challenge["bidder"] == self.opponent_id:
                self.actual_bluffs += 1
            
            # Đã đối chiếu xong
            self.last_challenge = None

        # Cập nhật số lượng xúc xắc mới nhất của cả hai bên
        self.last_dice_counts = {
            self.player_id: len(observation["my_dice"]),
            self.opponent_id: observation["opponent_dice_count"]
        }

        # Xử lý hồi tố (retrospective) các cược của đối thủ chưa được tính bluff opportunity của vòng này
        for bid_info in self.observed_bids:
            if not bid_info["processed"]:
                self._process_single_bid(bid_info)

        # 4. Tính toán bluff_rate và dynamic_threshold
        if self.bluff_opportunities > 0:
            bluff_rate = self.actual_bluffs / self.bluff_opportunities
        else:
            bluff_rate = 0.3 # Giá trị mặc định ban đầu khi chưa thu thập đủ dữ liệu

        # Ngưỡng động dao động từ [0.4, 0.6] tỷ lệ thuận với bluff_rate
        # Nếu bluff_rate cao (đối thủ hay nói dối) -> dynamic_threshold cao (tối đa 0.6) -> Dễ Challenge hơn
        # Nếu bluff_rate thấp (đối thủ thật thà) -> dynamic_threshold thấp (tối thiểu 0.4) -> Khó Challenge hơn (an toàn)
        dynamic_threshold = max(0.4, min(0.6, 0.4 + (bluff_rate * 0.2)))

        # 5. Logic ra quyết định tương tự ProbabilisticAgent nhưng sử dụng dynamic_threshold
        counts = {f: count_matching_dice([self.my_dice], f) for f in range(1, 7)}
        current_bid = observation["current_bid"]

        # 5.1. Trường hợp đi đầu vòng (chưa có ai cược)
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

        # 5.2. Trường hợp đối thủ đã cược
        Q = current_bid.quantity
        F = current_bid.face_value
        my_count = counts[F]
        k = Q - my_count
        n = observation["opponent_dice_count"]
        p = 1/6 if F == 1 else 1/3

        # Tính xác suất đối thủ nói thật
        P_truth = binomial_probability(k, n, p)

        # Quyết định Challenge dựa trên ngưỡng động dynamic_threshold
        if P_truth < dynamic_threshold and any(isinstance(a, Challenge) for a in legal_actions):
            return Challenge()

        # Ngược lại, nâng cược tối ưu kỳ vọng
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
