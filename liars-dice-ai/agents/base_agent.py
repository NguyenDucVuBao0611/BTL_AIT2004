from typing import List
from game.actions import Action

class Agent:
    """Lớp cơ sở (Interface) cho tất cả các Agent chơi Liar's Dice."""
    def __init__(self, name: str):
        self.name = name

    def act(self, observation: dict, legal_actions: List[Action]) -> Action:
        """Quyết định hành động tiếp theo.
        
        Args:
            observation (dict): Thông tin mà người chơi này được phép biết.
                - "player_id": ID của người chơi này (0 hoặc 1).
                - "my_dice": Danh sách xúc xắc hiện tại của bản thân.
                - "opponent_dice_count": Số lượng xúc xắc còn lại của đối phương.
                - "current_bid": Đối tượng Bid cược hiện tại của đối thủ hoặc None.
                - "history": Lịch sử các cặp (player_id, action) trong vòng chơi này.
            legal_actions (List[Action]): Danh sách các hành động hợp lệ có thể chọn.
            
        Returns:
            Action: Hành động được chọn (Bid hoặc Challenge).
        """
        raise NotImplementedError("Phương thức act() cần được cài đặt ở lớp con.")

    def __str__(self):
        return self.name
