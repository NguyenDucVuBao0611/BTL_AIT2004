import random
from typing import List, Optional
from game.actions import Action, Bid

class GameState:
    """Đại diện cho toàn bộ trạng thái của một trận đấu Liar's Dice 2 người chơi."""
    def __init__(self, start_dice: int = 5):
        self.num_players = 2
        # Số lượng xúc xắc còn lại của mỗi người chơi
        self.dice_counts = [start_dice, start_dice]
        # Giá trị xúc xắc được lắc trong vòng hiện tại của mỗi người chơi (ví dụ: [[2, 3, 1, 5, 6], [4, 4, 1, 2, 3]])
        self.hands: List[List[int]] = [[], []]
        # Lượt cược hiện tại trong vòng này
        self.current_bid: Optional[Bid] = None
        # ID người chơi đang đến lượt (0 hoặc 1)
        self.current_player: int = 0
        # Lịch sử các hành động trong vòng hiện tại: danh sách các tuple (player_id, action)
        self.history: List[tuple] = []
        
        # Khởi tạo vòng chơi đầu tiên
        self.roll_all_dice()

    def roll_all_dice(self):
        """Lắc xúc xắc mới cho tất cả người chơi còn hoạt động dựa trên số xúc xắc còn lại của họ."""
        self.hands = [
            sorted([random.randint(1, 6) for _ in range(self.dice_counts[i])])
            for i in range(self.num_players)
        ]

    def reset_round(self, starting_player: int):
        """Đặt lại trạng thái vòng đấu, lắc xúc xắc mới, reset cược và lịch sử.
        Người chơi thua vòng trước (hoặc được chỉ định) sẽ đi trước.
        """
        self.roll_all_dice()
        self.current_bid = None
        self.current_player = starting_player
        self.history = []

    def is_game_over(self) -> bool:
        """Trả về True nếu có bất kỳ người chơi nào hết xúc xắc."""
        return any(count <= 0 for count in self.dice_counts)

    def get_winner(self) -> Optional[int]:
        """Trả về ID của người thắng cuộc (người vẫn còn xúc xắc),
        hoặc None nếu trò chơi chưa kết thúc.
        """
        if not self.is_game_over():
            return None
        return 0 if self.dice_counts[0] > 0 else 1

    def clone(self) -> 'GameState':
        """Tạo một bản sao sâu (deep copy) của trạng thái game."""
        new_state = GameState.__new__(GameState)
        new_state.num_players = self.num_players
        new_state.dice_counts = list(self.dice_counts)
        new_state.hands = [list(hand) for hand in self.hands]
        new_state.current_bid = self.current_bid
        new_state.current_player = self.current_player
        new_state.history = list(self.history)
        return new_state

    def get_observation(self, player_id: int) -> dict:
        """Trả về thông tin quan sát được cho một người chơi cụ thể (ẩn xúc xắc của đối thủ)."""
        return {
            "player_id": player_id,
            "my_dice": list(self.hands[player_id]),
            "opponent_dice_count": self.dice_counts[1 - player_id],
            "current_bid": self.current_bid,
            "history": list(self.history)
        }
