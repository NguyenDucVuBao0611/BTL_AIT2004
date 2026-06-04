from typing import List, Dict, Any
from game.state import GameState
from game.actions import Bid, Challenge
from game.engine import get_legal_actions, apply_action
from agents.base_agent import Agent

class Referee:
    """Referee quản lý luồng điều phối vòng lặp game đấu giữa 2 Agent trong Liar's Dice."""
    def __init__(self, agent_0: Agent, agent_1: Agent, start_dice: int = 5, verbose: bool = True):
        self.agents = [agent_0, agent_1]
        self.start_dice = start_dice
        self.verbose = verbose
        self.state = GameState(start_dice=self.start_dice)
        self.game_log: List[Dict[str, Any]] = []

    def log(self, message: str):
        """In thông báo ra terminal nếu chế độ verbose là True."""
        if self.verbose:
            print(message)

    def play_game(self) -> int:
        """Chạy toàn bộ một trận đấu Liar's Dice cho đến khi một người chơi hết xúc xắc.
        Trả về ID của người chơi giành chiến thắng chung cuộc (0 hoặc 1).
        """
        self.log(f"=== Bắt đầu trận đấu Liar's Dice ===")
        self.log(f"Người chơi 0: {self.agents[0]}")
        self.log(f"Người chơi 1: {self.agents[1]}")
        self.log(f"Mỗi người bắt đầu với {self.start_dice} xúc xắc. Mặt 1 là Wild!\n")

        round_num = 1
        
        while not self.state.is_game_over():
            self.log(f"--- VÒNG {round_num} ---")
            self.log(f"Số xúc xắc còn lại: {self.agents[0].name} ({self.state.dice_counts[0]}) vs {self.agents[1].name} ({self.state.dice_counts[1]})")
            
            # Log xúc xắc giấu của cả 2 bên (dành cho trọng tài theo dõi hoặc phục vụ debug)
            self.log(f"[Referee] Xúc xắc giấu: {self.agents[0].name}: {self.state.hands[0]} | {self.agents[1].name}: {self.state.hands[1]}")
            
            round_active = True
            
            while round_active:
                curr_player_id = self.state.current_player
                agent = self.agents[curr_player_id]
                
                # Tạo thông tin quan sát được của người chơi hiện tại
                obs = self.state.get_observation(curr_player_id)
                legal_actions = get_legal_actions(self.state)
                
                # Agent đưa ra quyết định hành động
                action = agent.act(obs, legal_actions)
                
                # Bảo vệ: kiểm tra agent có trả về hành động hợp lệ hay không
                if action not in legal_actions:
                    import random as _rand
                    self.log(f"[Referee] CẢNH BÁO: {agent.name} trả về hành động không hợp lệ ({action}). Chọn ngẫu nhiên thay thế.")
                    action = _rand.choice(legal_actions)
                
                # Áp dụng hành động lên game
                outcome = apply_action(self.state, action)
                self.game_log.append(outcome)
                
                # Hiển thị hành động ra màn hình
                if outcome["type"] == "bid":
                    self.log(f"{agent.name} cược: \"Có ít nhất {action.quantity} viên mặt {action.face_value}\"")
                elif outcome["type"] == "challenge":
                    self.log(f"{agent.name} hô: \"LIAR!\" (Tố cáo cược của {self.agents[outcome['bidder']].name})")
                    
                    # Chi tiết kết quả tố cáo
                    bid = outcome["bid"]
                    actual = outcome["actual_count"]
                    winner = self.agents[outcome["winner"]].name
                    loser = self.agents[outcome["loser"]].name
                    
                    self.log(f"  -> Xúc xắc thực tế của cả 2 bên: {self.agents[0].name}: {outcome['hands'][0]} | {self.agents[1].name}: {outcome['hands'][1]}")
                    self.log(f"  -> Tổng số mặt {bid.face_value} (tính cả mặt 1 là wild): {actual} viên.")
                    
                    if outcome["challenge_success"]:
                        self.log(f"  -> {agent.name} tố cáo ĐÚNG! {self.agents[outcome['bidder']].name} đã nói dối.")
                    else:
                        self.log(f"  -> {agent.name} tố cáo SAI! Có đủ hoặc nhiều hơn số lượng cược.")
                        
                    self.log(f"  -> Kết quả: {loser} mất 1 xúc xắc.\n")
                    round_active = False # Kết thúc vòng chơi
            
            round_num += 1

        # Game kết thúc
        winner_id = self.state.get_winner()
        winner_name = self.agents[winner_id].name
        self.log(f"=== TRẬN ĐẤU KẾT THÚC ===")
        self.log(f"Người chiến thắng chung cuộc là: {winner_name} 🎉")
        return winner_id
