import sys
import os

# Thêm thư mục dự án vào python path để tránh lỗi import chéo
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from game.referee import Referee
from agents.random_agent import RandomAgent
from agents.probabilistic_agent import ProbabilisticAgent
from agents.bayesian_agent import BayesianAgent
from agents.cfr_agent import CFRAgent

def run_demo():
    print("\n" + "=" * 50)
    print("PHẦN 1: CHẠY DEMO CÁC TRẬN ĐẤU CÓ CHI TIẾT LOG (VERBOSE)")
    print("=" * 50)
    
    print("\n=========================================")
    print("TRẬN 1: ProbabilisticAgent vs RandomAgent")
    print("=========================================")
    agent_prob = ProbabilisticAgent("Probabilistic_0", threshold=0.5)
    agent_rand = RandomAgent("Random_1")
    referee_1 = Referee(agent_prob, agent_rand, start_dice=5, verbose=True)
    referee_1.play_game()
    
    print("\n=========================================")
    print("TRẬN 2: BayesianAgent vs ProbabilisticAgent")
    print("=========================================")
    agent_bayes = BayesianAgent("Bayesian_0")
    agent_prob_2 = ProbabilisticAgent("Probabilistic_1", threshold=0.5)
    referee_2 = Referee(agent_bayes, agent_prob_2, start_dice=5, verbose=True)
    referee_2.play_game()

def run_tournament(num_games=30):
    print("\n" + "=" * 50)
    print("PHẦN 2: GIẢI ĐẤU VÒNG TRÒN (ROUND-ROBIN TOURNAMENT)")
    print("=" * 50)
    
    # 1. Khởi tạo CFR Agent và huấn luyện nhanh qua tự chơi
    print("Đang huấn luyện nhanh CFR Agent bằng Self-play (5000 iterations)...")
    agent_cfr = CFRAgent("CFRAgent")
    agent_cfr.train(5000)
    print(f"Hoàn tất huấn luyện! Đã học được {len(agent_cfr.strategy_table)} trạng thái.")
    
    agents = [
        RandomAgent("RandomAgent"),
        ProbabilisticAgent("ProbabilisticAgent", threshold=0.5),
        BayesianAgent("BayesianAgent"),
        agent_cfr
    ]
    
    # Ma trận kết quả: win_matrix[player][opponent] = số trận player thắng opponent
    win_matrix = {a.name: {opp.name: 0 for opp in agents} for a in agents}
    
    print(f"\nKhởi chạy giải đấu: mỗi cặp đấu {num_games} trận...")
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i == j:
                continue
            a1 = agents[i]
            a2 = agents[j]
            print(f"Đang đấu: {a1.name} vs {a2.name}... ", end="", flush=True)
            
            a1_wins = 0
            for _ in range(num_games):
                # Tạo referee và chạy đấu tắt verbose
                ref = Referee(a1, a2, start_dice=5, verbose=False)
                winner_id = ref.play_game()
                if winner_id == 0:
                    a1_wins += 1
                    
            win_matrix[a1.name][a2.name] = a1_wins
            win_rate = (a1_wins / num_games) * 100
            print(f"Thắng {a1_wins}/{num_games} trận ({win_rate:.1f}%)")
            
    # Hiển thị bảng kết quả tỉ lệ thắng trực quan
    print("\n" + "=" * 80)
    print("BẢNG THỐNG KÊ TỶ LỆ CHIẾN THẮNG TRONG GIẢI ĐẤU (WIN-RATE MATRIX)")
    print("=" * 80)
    # In Header
    print(f"{'Đội cược (Hàng)':<20} | ", end="")
    for a in agents:
        print(f"{a.name[:12]:<12} | ", end="")
    print("\n" + "-" * 78)
    
    for a1 in agents:
        print(f"{a1.name:<20} | ", end="")
        for a2 in agents:
            if a1.name == a2.name:
                print(f"{'-':<12} | ", end="")
            else:
                wins = win_matrix[a1.name][a2.name]
                wr = (wins / num_games) * 100
                print(f"{wr:>5.1f}%      | ", end="")
        print()
    print("=" * 80 + "\n")

def main():
    run_demo()
    # Chạy giải đấu thu nhỏ với 20 trận để phản hồi nhanh
    run_tournament(num_games=20)

if __name__ == "__main__":
    main()
