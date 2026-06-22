"""Giải đấu round-robin có TÁI LẬP (seed) và CÂN BẰNG GHẾ (seat-swap).

Hai cải tiến so với bản tournament cũ trong main.py:
  1. `seed`: gieo hạt ngẫu nhiên một lần ⇒ kết quả lặp lại được giữa các lần chạy.
  2. Cân bằng ghế: mỗi cặp đấu chia đôi số ván, luân phiên ai đi trước (P0). Nhờ đó
     lợi thế đi trước (first-move advantage) bị khử khỏi từng ô của ma trận win-rate.
"""
import random
from typing import Dict, List

from game.referee import Referee
from agents.base_agent import Agent


def play_match(agent_a: Agent, agent_b: Agent, num_games: int,
               start_dice: int = 5) -> int:
    """Đấu `num_games` ván giữa A và B, LUÂN PHIÊN ghế đi trước. Trả về số ván A thắng."""
    a_wins = 0
    for g in range(num_games):
        if g % 2 == 0:
            # A là P0, B là P1
            ref = Referee(agent_a, agent_b, start_dice=start_dice, verbose=False)
            if ref.play_game() == 0:
                a_wins += 1
        else:
            # Đổi ghế: B là P0, A là P1
            ref = Referee(agent_b, agent_a, start_dice=start_dice, verbose=False)
            if ref.play_game() == 1:
                a_wins += 1
    return a_wins


def run_tournament(agents: List[Agent], num_games: int = 30, seed: int = 0,
                   start_dice: int = 5, progress: bool = True) -> Dict:
    """Chạy round-robin cân bằng ghế. Trả về dict gồm ma trận thắng và win-rate tổng.

    Returns:
        {
          "names": [...],
          "win_matrix": {a: {b: số ván a thắng b (cân bằng ghế)}},
          "num_games": num_games,
          "overall_winrate": {a: tỉ lệ thắng trung bình trên mọi đối thủ},
        }
    """
    random.seed(seed)
    names = [a.name for a in agents]
    win_matrix: Dict[str, Dict[str, int]] = {a.name: {b.name: 0 for b in agents} for a in agents}

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            if progress:
                print(f"Đang đấu: {a.name} vs {b.name} ... ", end="", flush=True)

            a_wins = play_match(a, b, num_games, start_dice=start_dice)
            win_matrix[a.name][b.name] = a_wins
            win_matrix[b.name][a.name] = num_games - a_wins

            if progress:
                print(f"{a.name} {a_wins}/{num_games} ({a_wins/num_games*100:.1f}%)")

    # Win-rate tổng: trung bình tỉ lệ thắng trên tất cả đối thủ
    overall = {}
    for a in agents:
        opps = [b for b in agents if b.name != a.name]
        total = sum(win_matrix[a.name][b.name] for b in opps)
        overall[a.name] = total / (len(opps) * num_games) if opps else 0.0

    return {
        "names": names,
        "win_matrix": win_matrix,
        "num_games": num_games,
        "overall_winrate": overall,
    }


def print_win_matrix(result: Dict) -> None:
    """In ma trận win-rate ra terminal (ô [hàng][cột] = % hàng thắng cột)."""
    names = result["names"]
    wm = result["win_matrix"]
    ng = result["num_games"]

    print("\n" + "=" * 80)
    print("MA TRẬN TỶ LỆ THẮNG (cân bằng ghế) — ô [hàng][cột] = % hàng thắng cột")
    print("=" * 80)
    print(f"{'':<20} | " + " | ".join(f"{n[:12]:<12}" for n in names))
    print("-" * 80)
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append(f"{'—':<12}")
            else:
                cells.append(f"{wm[a][b]/ng*100:>5.1f}%      ")
        print(f"{a:<20} | " + " | ".join(cells))
    print("-" * 80)
    print("Win-rate tổng (trung bình mọi đối thủ):")
    for a in names:
        print(f"  {a:<22} {result['overall_winrate'][a]*100:5.1f}%")
    print("=" * 80 + "\n")
