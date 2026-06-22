import sys
import os

# Ép stdout/stderr về UTF-8 để in tiếng Việt không lỗi trên console Windows (cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Thêm thư mục dự án vào python path để tránh lỗi import chéo
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from game.referee import Referee
from agents.random_agent import RandomAgent
from agents.probabilistic_agent import ProbabilisticAgent
from agents.bayesian_agent import BayesianAgent
from agents.cfr_agent import CFRAgent

# Weights CFR đã huấn luyện nặng (mạnh, thắng Prob/Bayes). Bản nén .gz (~7MB) ĐƯỢC commit
# để ai clone về là chạy được CFR mạnh ngay, khỏi train lại 40k vòng. Ưu tiên nạp bản .gz;
# nếu có sẵn bản .json chưa nén cục bộ thì cũng dùng được; không có gì thì train (fallback).
_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
CFR_WEIGHTS_GZ = os.path.join(_WEIGHTS_DIR, "cfr_heavy.weights.json.gz")
CFR_WEIGHTS_JSON = os.path.join(_WEIGHTS_DIR, "cfr_heavy.weights.json")


def _find_cfr_weights():
    """Trả về đường dẫn weights khả dụng (ưu tiên .gz đã commit), hoặc None nếu không có."""
    for path in (CFR_WEIGHTS_GZ, CFR_WEIGHTS_JSON):
        if os.path.exists(path):
            return path
    return None


def make_cfr_agent(name="CFRAgent", train_iters=40000):
    """Tạo CFRAgent mạnh: NẠP weights đã huấn luyện nếu có sẵn, ngược lại tự train."""
    agent = CFRAgent(name)
    weights_path = _find_cfr_weights()
    if weights_path:
        agent.load_weights(weights_path)
        print(f"Đã nạp CFR weights ({len(agent.strategy_table)} trạng thái) từ "
              f"{os.path.relpath(weights_path)} — bỏ qua huấn luyện.")
    else:
        print(f"Không thấy weights sẵn, huấn luyện CFR {train_iters} vòng self-play...")
        agent.train(train_iters)
        print(f"Hoàn tất huấn luyện! Đã học {len(agent.strategy_table)} trạng thái.")
    return agent

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

def run_tournament(num_games=30, seed=0, make_plots=True, cfr_iters=40000):
    print("\n" + "=" * 50)
    print("PHẦN 2: GIẢI ĐẤU VÒNG TRÒN (ROUND-ROBIN, có seed + cân bằng ghế)")
    print("=" * 50)
    from evaluation.tournament import run_tournament as _run, print_win_matrix

    # 1. Khởi tạo CFR Agent: nạp weights mạnh nếu có, ngược lại train (CFR+ cần nhiều vòng)
    agent_cfr = make_cfr_agent("CFRAgent", train_iters=cfr_iters)

    agents = [
        RandomAgent("RandomAgent"),
        ProbabilisticAgent("ProbabilisticAgent", threshold=0.5),
        BayesianAgent("BayesianAgent"),
        agent_cfr,
    ]

<<<<<<< HEAD
    def create_fresh_agent(agent_name):
        if agent_name == "RandomAgent":
            return RandomAgent("RandomAgent")
        elif agent_name == "ProbabilisticAgent":
            return ProbabilisticAgent("ProbabilisticAgent", threshold=0.5)
        elif agent_name == "BayesianAgent":
            return BayesianAgent("BayesianAgent")
        elif agent_name == "CFRAgent":
            new_cfr = CFRAgent("CFRAgent")
            new_cfr.regret_table = agent_cfr.regret_table
            new_cfr.strategy_table = agent_cfr.strategy_table
            return new_cfr
        raise ValueError(f"Unknown agent: {agent_name}")
    
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
                # Tạo mới instance cho mỗi trận đấu để tránh lỗi "ám trạng thái" (State Bleeding)
                fresh_a1 = create_fresh_agent(a1.name)
                fresh_a2 = create_fresh_agent(a2.name)
                ref = Referee(fresh_a1, fresh_a2, start_dice=5, verbose=False)
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
=======
    print(f"\nKhởi chạy giải đấu: mỗi cặp đấu {num_games} ván (seed={seed})...")
    result = _run(agents, num_games=num_games, seed=seed)
    print_win_matrix(result)

    if make_plots:
        try:
            from evaluation.plots import plot_win_matrix, plot_overall_winrate
            p1 = plot_win_matrix(result)
            p2 = plot_overall_winrate(result)
            print(f"Đã lưu biểu đồ: {p1}, {p2}")
        except Exception as e:  # pragma: no cover - thiếu matplotlib không nên làm hỏng giải đấu
            print(f"(Bỏ qua vẽ biểu đồ: {e})")

    return result


def run_exploit(total_iterations=20000, step=2000, eval_games=200, seed=0, make_plots=True):
    print("\n" + "=" * 50)
    print("PHẦN 3: ĐO HỘI TỤ / EXPLOITABILITY CỦA CFR THEO SỐ VÒNG LẶP")
    print("=" * 50)
    from evaluation.exploitability import measure_convergence

    history = measure_convergence(total_iterations=total_iterations, step=step,
                                  eval_games=eval_games, seed=seed)

    if make_plots:
        try:
            from evaluation.plots import plot_convergence, plot_infoset_growth
            p = plot_convergence(history)
            p2 = plot_infoset_growth(history)
            print(f"Đã lưu biểu đồ hội tụ: {p}, {p2}")
        except Exception as e:  # pragma: no cover
            print(f"(Bỏ qua vẽ biểu đồ: {e})")

    return history
>>>>>>> 60744c8a5f5dadb435e766181c81ba8f4134e9f1

def make_agent(name):
    """Tạo agent AI đối thủ cho chế độ chơi người-vs-AI (tên hiển thị = tên loại agent)."""
    key = (name or "random").lower()
    if key in ("random", "rand"):
        return RandomAgent("RandomAgent")
    if key in ("prob", "probabilistic"):
        return ProbabilisticAgent("ProbabilisticAgent", threshold=0.5)
    if key in ("bayes", "bayesian"):
        return BayesianAgent("BayesianAgent")
    if key == "cfr":
        # Nạp weights mạnh nếu có; nếu không, train nhanh 1000 vòng làm fallback
        return make_cfr_agent("CFRAgent", train_iters=1000)
    raise ValueError(f"Agent không hợp lệ: {name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Liar's Dice AI — demo, giải đấu, hoặc chơi vs AI")
    parser.add_argument("--mode", default="all",
                        choices=["all", "demo", "tournament", "exploit", "cli", "gui"],
                        help="Chế độ chạy (mặc định: all = demo + tournament)")
    parser.add_argument("--agent", default="bayesian",
                        help="AI đối thủ cho cli/gui: random | probabilistic | bayesian | cfr")
    parser.add_argument("--seed", type=int, default=0, help="Seed ngẫu nhiên (tournament/exploit/cli)")
    parser.add_argument("--games", type=int, default=20, help="Số trận mỗi cặp trong tournament")
    parser.add_argument("--iters", type=int, default=20000, help="Tổng vòng lặp self-play cho mode exploit")
    parser.add_argument("--cfr-iters", type=int, default=40000, help="Số vòng self-play huấn luyện CFR trước tournament")
    args = parser.parse_args()

    if args.mode == "all":
        run_demo()
        run_tournament(num_games=args.games, seed=args.seed, cfr_iters=args.cfr_iters)
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "tournament":
        run_tournament(num_games=args.games, seed=args.seed, cfr_iters=args.cfr_iters)
    elif args.mode == "exploit":
        run_exploit(total_iterations=args.iters, seed=args.seed)
    elif args.mode == "cli":
        from ui.cli import play_vs_ai
        play_vs_ai(make_agent(args.agent), seed=args.seed)
    elif args.mode == "gui":
        from ui.gui import play_gui
        play_gui(make_agent(args.agent), seed=args.seed)


if __name__ == "__main__":
    main()
