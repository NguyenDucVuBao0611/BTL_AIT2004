import sys
import os

# Thêm thư mục dự án vào python path để tránh lỗi import chéo
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from game.referee import Referee
from agents.random_agent import RandomAgent

def main():
    print("Khởi chạy demo game Liar's Dice...")
    # Khởi tạo hai Random Agent chơi thử
    agent_0 = RandomAgent("Random_0")
    agent_1 = RandomAgent("Random_1")
    
    # Tạo referee và chạy game đấu
    referee = Referee(agent_0, agent_1, start_dice=5, verbose=True)
    referee.play_game()

if __name__ == "__main__":
    main()
