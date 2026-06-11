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

def run_tournament(num_games=30, seed=0, make_plots=True):
    print("\n" + "=" * 50)
    print("PHẦN 2: GIẢI ĐẤU VÒNG TRÒN (ROUND-ROBIN, có seed + cân bằng ghế)")
    print("=" * 50)
    from evaluation.tournament import run_tournament as _run, print_win_matrix

    # 1. Khởi tạo CFR Agent và huấn luyện nhanh qua tự chơi
    print("Đang huấn luyện nhanh CFR Agent bằng Self-play (5000 iterations)...")
    agent_cfr = CFRAgent("CFRAgent")
    agent_cfr.train(5000)
    print(f"Hoàn tất huấn luyện! Đã học được {len(agent_cfr.strategy_table)} trạng thái.")

    agents = [
        RandomAgent("RandomAgent"),
        ProbabilisticAgent("ProbabilisticAgent", threshold=0.5),
        BayesianAgent("BayesianAgent"),
        agent_cfr,
    ]

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
            from evaluation.plots import plot_convergence
            p = plot_convergence(history)
            print(f"Đã lưu biểu đồ hội tụ: {p}")
        except Exception as e:  # pragma: no cover
            print(f"(Bỏ qua vẽ biểu đồ: {e})")

    return history

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
        agent = CFRAgent("CFRAgent")
        print("Đang huấn luyện nhanh CFR (1000 ván self-play)...")
        agent.train(1000)
        return agent
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
    args = parser.parse_args()

    if args.mode == "all":
        run_demo()
        run_tournament(num_games=args.games, seed=args.seed)
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "tournament":
        run_tournament(num_games=args.games, seed=args.seed)
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
