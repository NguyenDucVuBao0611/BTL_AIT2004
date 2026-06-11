"""CLI người-vs-AI bằng `rich` (NGƯỜI 3).

Chạy:  python main.py --mode cli
Nhập:  'c'        -> hô Liar (Challenge)
       '<q> <f>'  -> bid quantity q, face f  (vd: 3 5)

Nếu không có `rich`, tự rớt về chế độ text thuần để vẫn chạy được.
"""
from __future__ import annotations

import random

from game.actions import Action, Bid, Challenge
from game.engine import apply_action, get_legal_actions
from game.state import GameState
from agents.base_agent import Agent
from agents.random_agent import RandomAgent

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    _RICH = True
    _console = Console()
except ImportError:  # pragma: no cover - fallback hiếm khi dùng
    _RICH = False
    _console = None

HUMAN, AI = 0, 1

# Mặt xúc xắc bằng ký tự (1 là wild → tô khác).
_PIPS = {1: "⚀", 2: "⚁", 3: "⚂", 4: "⚃", 5: "⚄", 6: "⚅"}


def _render_dice(dice) -> str:
    parts = []
    for d in dice:
        face = _PIPS.get(d, str(d))
        parts.append(f"[bold yellow]{face}(wild)[/]" if d == 1 else f"[cyan]{face}[/]")
    return " ".join(parts)


def _show_state(obs: dict) -> None:
    my_dice = obs["my_dice"]
    opp_count = obs["opponent_dice_count"]
    current_bid = obs["current_bid"]
    if not _RICH:
        print(f"\nXúc xắc của bạn: {my_dice}")
        print(f"Số xúc xắc (bạn / AI): {len(my_dice)} / {opp_count}")
        print(f"Bid hiện tại: {current_bid}")
        return

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Xúc xắc của bạn:", _render_dice(my_dice))
    table.add_row("Số xúc xắc (bạn, AI):", f"({len(my_dice)}, {opp_count})")
    bid_txt = (
        f"[bold]{current_bid.quantity} con mặt {current_bid.face_value}[/]"
        if current_bid else "[dim]chưa có[/]"
    )
    table.add_row("Bid hiện tại:", bid_txt)
    _console.print(Panel(table, title="[bold]Lượt của bạn[/]", border_style="green"))


def _read_human_action(obs: dict, legal_actions: list[Action]) -> Action:
    can_challenge = any(isinstance(a, Challenge) for a in legal_actions)
    _show_state(obs)

    while True:
        hint = "Nhập 'q f' để bid" + (" hoặc 'c' để hô Liar" if can_challenge else "")
        try:
            raw = (Prompt.ask(hint) if _RICH else input(hint + ": ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            _say("\nĐã thoát ván chơi.", style="bold red")
            raise SystemExit(0)

        if raw == "c" and can_challenge:
            return Challenge()
        try:
            q, f = (int(x) for x in raw.split())
            bid = Bid(q, f)
        except (ValueError, TypeError):
            _warn("Sai định dạng. Ví dụ: '3 5' nghĩa là 3 con mặt 5.")
            continue
        if any(isinstance(a, Bid) and a == bid for a in legal_actions):
            return bid
        _warn("Nước đi không hợp lệ (phải cao hơn bid hiện tại).")


def _warn(msg: str) -> None:
    if _RICH:
        _console.print(f"[red]» {msg}[/]")
    else:
        print("  " + msg)


def _say(msg: str, style: str = "") -> None:
    if _RICH:
        _console.print(f"[{style}]{msg}[/]" if style else msg)
    else:
        print(msg)


class HumanAgent(Agent):
    def act(self, observation: dict, legal_actions: list[Action]) -> Action:
        return _read_human_action(observation, legal_actions)


def play_vs_ai(ai_agent: Agent | None = None, seed: int | None = None,
               dice_per_player: int = 5) -> None:
    if seed is not None:
        random.seed(seed)
    ai = ai_agent or RandomAgent(name="AI")
    human = HumanAgent(name="Bạn")
    agents = [human, ai]
    for a in agents:
        a.reset()

    _say("=== Liar's Dice: Bạn (P0) vs AI (P1) ===", style="bold magenta")

    state = GameState(start_dice=dice_per_player)
    state.reset_round(starting_player=HUMAN)
    rounds = 0

    while not state.is_game_over():
        rounds += 1
        _say(f"\n----- VÒNG {rounds} | xúc xắc (bạn, AI) = "
             f"({state.dice_counts[HUMAN]}, {state.dice_counts[AI]}) -----", style="dim")
        round_over = False
        while not round_over:
            pid = state.current_player
            agent = agents[pid]
            obs = state.get_observation(pid)
            legal = get_legal_actions(state)
            action = agent.act(obs, legal)
            if action not in legal:
                action = random.choice(legal)

            outcome = apply_action(state, action)
            for a in agents:
                a.observe(action, pid)

            if outcome["type"] == "bid":
                if pid == AI:
                    _say(f"AI cược: {action.quantity} con mặt {action.face_value}", style="dim")
            else:  # challenge
                bid = outcome["bid"]
                actual = outcome["actual_count"]
                loser = outcome["loser"]
                who = "Bạn" if pid == HUMAN else "AI"
                _say(f"{who} hô LIAR! mặt {bid.face_value}: thực {actual} / cược {bid.quantity}",
                     style="bold yellow")
                _say(f"→ {'Bạn' if loser == HUMAN else 'AI'} thua vòng, mất 1 xúc xắc "
                     f"(còn lại {state.dice_counts})", style="dim")
                round_over = True

    winner = "Bạn 🎉" if state.get_winner() == HUMAN else "AI 🤖"
    _say(f"\n*** {winner} THẮNG sau {rounds} vòng! ***", style="bold")


if __name__ == "__main__":
    play_vs_ai()
