"""GUI pygame cho Liar's Dice — người vs AI (NGƯỜI 3, phần "wow", tuỳ chọn).

Chạy:  python main.py --mode gui --agent bayesian

Thiết kế: KHÔNG dùng thread. Tái sử dụng game engine (GameState) + agent có sẵn,
điều phối ván bằng một máy trạng thái (state machine) trong vòng lặp pygame:

  human_turn → (bid) → ai_turn → (bid) → human_turn → ...
            ↘ (Liar!) ┐                ┌ (Liar!) ↙
                      reveal → next round / gameover

Agent đối thủ chỉ cần tuân thủ Agent.act(observation, legal_actions) như mọi nơi.
"""
from __future__ import annotations

import os

# Khi chạy trực tiếp `python ui/gui.py` (chứ không phải `python -m ui.gui`), thêm gốc dự án
# vào sys.path để import được các package `agents`, `game`... (giống các script experiments/).
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import pygame

from agents.base_agent import Agent
from agents.random_agent import RandomAgent
from game.actions import Bid, Challenge
from game.engine import count_matching_dice, get_legal_actions, validate_action
from game.state import GameState

# ---- Hằng số mặt xúc xắc (API duc không export, tự định nghĩa) ----
MIN_FACE, MAX_FACE = 1, 6
NUM_FACES = 6

# ---- Kích thước & màu ----
W, H = 800, 720
FPS = 60
AI_THINK_MS = 900

BG = (7, 92, 40)
PANEL = (6, 70, 31)
PANEL_LIGHT = (9, 110, 49)
WHITE = (244, 244, 244)
INK = (28, 28, 28)
GOLD = (214, 178, 64)
RED = (188, 62, 52)
GREEN_BTN = (46, 160, 90)
BLUE_BTN = (52, 122, 196)
GREY_BTN = (90, 96, 104)
TEXT = (238, 238, 238)
DIM = (170, 190, 175)

HUMAN, AI = 0, 1


# ======================= Vẽ phần tử =======================

_PIP_GRID = {
    "TL": (0.27, 0.27), "TC": (0.5, 0.27), "TR": (0.73, 0.27),
    "ML": (0.27, 0.5), "C": (0.5, 0.5), "MR": (0.73, 0.5),
    "BL": (0.27, 0.73), "BC": (0.5, 0.73), "BR": (0.73, 0.73),
}
_PIPS = {
    1: ["C"],
    2: ["TL", "BR"],
    3: ["TL", "C", "BR"],
    4: ["TL", "TR", "BL", "BR"],
    5: ["TL", "TR", "C", "BL", "BR"],
    6: ["TL", "TR", "ML", "MR", "BL", "BR"],
}


def draw_die(surf, x, y, size, value, hidden=False, highlight=False):
    r = int(size * 0.16)
    rect = pygame.Rect(x, y, size, size)
    if hidden:
        pygame.draw.rect(surf, (132, 46, 44), rect, border_radius=r)
        pygame.draw.rect(surf, (225, 225, 225), rect, width=2, border_radius=r)
        f = pygame.font.SysFont("Segoe UI", int(size * 0.55), bold=True)
        q = f.render("?", True, (235, 220, 220))
        surf.blit(q, q.get_rect(center=rect.center))
        return
    is_wild = value == 1
    body = GOLD if is_wild else WHITE
    pygame.draw.rect(surf, body, rect, border_radius=r)
    border = (150, 110, 20) if is_wild else (45, 45, 45)
    if highlight:
        pygame.draw.rect(surf, (40, 130, 220), rect, width=4, border_radius=r)
    else:
        pygame.draw.rect(surf, border, rect, width=2, border_radius=r)
    pip_color = (60, 45, 10) if is_wild else (40, 40, 40)
    pr = max(3, int(size * 0.085))
    for key in _PIPS[value]:
        fx, fy = _PIP_GRID[key]
        pygame.draw.circle(surf, pip_color, (int(x + fx * size), int(y + fy * size)), pr)
    if is_wild:
        f = pygame.font.SysFont("Segoe UI", int(size * 0.2), bold=True)
        lbl = f.render("WILD", True, (90, 65, 10))
        surf.blit(lbl, lbl.get_rect(center=(x + size // 2, y + int(size * 0.86))))


class Button:
    def __init__(self, rect, label, color=GREY_BTN, font=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color
        self.font = font
        self.enabled = True

    def draw(self, surf, mouse):
        hot = self.enabled and self.rect.collidepoint(mouse)
        base = self.color if self.enabled else (70, 74, 80)
        col = tuple(min(255, c + 28) for c in base) if hot else base
        pygame.draw.rect(surf, col, self.rect, border_radius=10)
        pygame.draw.rect(surf, (0, 0, 0, 60), self.rect, width=2, border_radius=10)
        txt = self.font.render(self.label, True, WHITE if self.enabled else (190, 190, 190))
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.enabled and self.rect.collidepoint(pos)


# ======================= Trò chơi =======================

class LiarsDiceGUI:
    def __init__(self, screen, ai: Agent, seed=None, dice_per_player=5, weights_path=None):
        self.screen = screen
        self.ai = ai
        if seed is not None:
            random.seed(seed)
        self.dice_per_player = dice_per_player

        self.f_big = pygame.font.SysFont("Segoe UI", 40, bold=True)
        self.f_mid = pygame.font.SysFont("Segoe UI", 26, bold=True)
        self.f_sm = pygame.font.SysFont("Segoe UI", 20)
        self.f_log = pygame.font.SysFont("Consolas", 18)

        self.counts = [dice_per_player, dice_per_player]
        self.start_player = HUMAN
        self.log_lines: list[str] = []
        self.round_history: list[dict] = []
        self.message = ""
        self.reveal = None          # dict thông tin lúc lật bài
        self.winner = None
        self.ai_think_at = None
        self.sel_q = 1
        self.sel_f = MIN_FACE
        self.player_streak = 0
        self.buttons: dict[str, Button] = {}
        self.weights_path = weights_path  # đường dẫn để auto-save sau mỗi ván

        # Khởi tạo thông tin cho Bayesian Agent (hoặc fallback) ngay lập tức
        bayes_agent = None
        from agents.bayesian_agent import BayesianAgent
        from agents.cfr_agent import CFRAgent
        if isinstance(self.ai, BayesianAgent):
            bayes_agent = self.ai
        elif isinstance(self.ai, CFRAgent) and self.ai.fallback_agent:
            bayes_agent = self.ai.fallback_agent
        if bayes_agent:
            bayes_agent.player_id = AI
            bayes_agent.opponent_id = HUMAN
            bayes_agent.last_dice_counts = {
                AI: dice_per_player,
                HUMAN: dice_per_player
            }


        # GameState là nguồn sự thật cho ván hiện tại (hands/current_bid/...).
        self.state = GameState(start_dice=dice_per_player)
        if hasattr(self.ai, "reset"):
            self.ai.reset()
        self._build_buttons()
        self._new_round()

    # ---------- vòng/ván ----------
    def _new_round(self):
        # Đồng bộ số xúc xắc còn lại rồi lắc vòng mới (GameState tự roll hands).
        self.state.dice_counts = list(self.counts)
        self.state.reset_round(starting_player=self.start_player)
        self.reveal = None
        self._reset_selection()
        self.phase = "human_turn" if self.start_player == HUMAN else "ai_turn"
        if self.phase == "ai_turn":
            self.ai_think_at = pygame.time.get_ticks() + AI_THINK_MS
        self._log(f"— Vòng mới: bạn {self.counts[HUMAN]} xúc xắc, "
                  f"AI {self.counts[AI]} xúc xắc —")
        
        # Đồng bộ thông tin Bayesian Agent
        bayes_agent = None
        from agents.bayesian_agent import BayesianAgent
        from agents.cfr_agent import CFRAgent
        if isinstance(self.ai, BayesianAgent):
            bayes_agent = self.ai
        elif isinstance(self.ai, CFRAgent) and self.ai.fallback_agent:
            bayes_agent = self.ai.fallback_agent
        if bayes_agent:
            bayes_agent.player_id = AI
            bayes_agent.opponent_id = HUMAN
            bayes_agent.last_dice_counts = {
                AI: self.counts[AI],
                HUMAN: self.counts[HUMAN]
            }

        # Lưu vào lịch sử đấu
        self.round_history.append({
            "round_num": len(self.round_history) + 1,
            "my_dice": self.counts[HUMAN],
            "ai_dice": self.counts[AI],
            "winner": None
        })
        
        self._export_thinking_state()

    def _reset_selection(self):
        cur = self.state.current_bid
        if cur is None:
            self.sel_q, self.sel_f = 1, MIN_FACE
        else:
            # Khởi tạo tới nước nâng nhỏ nhất hợp lệ.
            if cur.face_value < MAX_FACE:
                self.sel_q, self.sel_f = cur.quantity, cur.face_value + 1
            else:
                self.sel_q, self.sel_f = cur.quantity + 1, MIN_FACE

    def _log(self, line):
        self.log_lines.append(line)
        self.log_lines = self.log_lines[-9:]

    def _total(self):
        return sum(self.counts)

    def _sel_is_legal(self):
        bid = Bid(self.sel_q, self.sel_f)
        return validate_action(self.state, bid) and self.sel_q <= self._total()

    def _resolve_challenge(self):
        """Tính kết quả tố cáo mà KHÔNG thay đổi state (để vẽ reveal trước khi trừ xúc xắc).

        Trả về (loser, actual, claimed) — khớp chữ ký cũ resolve_challenge().
        """
        bid = self.state.current_bid
        actual = count_matching_dice(self.state.hands, bid.face_value)
        challenger = self.state.current_player
        bidder = 1 - challenger
        loser = challenger if actual >= bid.quantity else bidder
        return loser, actual, bid.quantity

    def _apply_action(self, actor, action):
        if isinstance(action, Bid):
            self.state.history.append((actor, action))
            self.state.current_bid = action
            self.state.current_player = 1 - actor
            who = "Bạn" if actor == HUMAN else "AI"
            self._log(f"{who} cược: {action.quantity} con mặt {action.face_value}")
            if hasattr(self.ai, "observe"):
                self.ai.observe(action, actor)
            if self.state.current_player == AI:
                self.phase = "ai_turn"
                self.ai_think_at = pygame.time.get_ticks() + AI_THINK_MS
            else:
                self.phase = "human_turn"
                self._reset_selection()
        else:  # Challenge
            loser, actual, claimed = self._resolve_challenge()
            face = self.state.current_bid.face_value
            who = "Bạn" if actor == HUMAN else "AI"
            self._log(f"{who} hô LIAR! (mặt {face}: thực {actual} / cược {claimed})")
            if hasattr(self.ai, "observe"):
                self.ai.observe(action, actor)
            
            # Cập nhật kết quả challenge cho Bayes agent ngay lập tức để hiển thị ngay trên màn hình Reveal
            bayes_agent = None
            from agents.bayesian_agent import BayesianAgent
            from agents.cfr_agent import CFRAgent
            if isinstance(self.ai, BayesianAgent):
                bayes_agent = self.ai
            elif isinstance(self.ai, CFRAgent) and self.ai.fallback_agent:
                bayes_agent = self.ai.fallback_agent
                
            if bayes_agent:
                if bayes_agent.player_id is None:
                    bayes_agent.player_id = AI
                    bayes_agent.opponent_id = HUMAN
                if bayes_agent.last_dice_counts is None:
                    bayes_agent.last_dice_counts = {
                        AI: self.counts[AI],
                        HUMAN: self.counts[HUMAN]
                    }
                if bayes_agent.last_challenge is not None:
                    obs = {
                        "opponent_dice_count": self.counts[HUMAN] - (1 if loser == HUMAN else 0),
                        "my_dice": self.state.hands[AI] if self.state.hands else [],
                        "history": self.state.history,
                        "current_bid": self.state.current_bid
                    }
                    bayes_agent._resolve_last_challenge(obs)

            self.reveal = {
                "loser": loser, "actual": actual, "claimed": claimed,
                "face": face, "challenger": actor,
            }
            self.phase = "reveal"
        self._export_thinking_state(actor, action)


    def _resolve_reveal(self):
        loser = self.reveal["loser"]
        self.counts[loser] -= 1
        self._log(f"{'Bạn' if loser == HUMAN else 'AI'} thua vòng, mất 1 xúc xắc "
                  f"→ {self.counts}")
        
        if self.round_history:
            self.round_history[-1]["winner"] = "Bạn" if loser == AI else "AI"
        
        # Cập nhật kết quả challenge cho Bayes agent ngay lập tức để thống kê không bị trễ
        bayes_agent = None
        from agents.bayesian_agent import BayesianAgent
        from agents.cfr_agent import CFRAgent
        if isinstance(self.ai, BayesianAgent):
            bayes_agent = self.ai
        elif isinstance(self.ai, CFRAgent) and self.ai.fallback_agent:
            bayes_agent = self.ai.fallback_agent
            
        if bayes_agent:
            if bayes_agent.player_id is None:
                bayes_agent.player_id = AI
                bayes_agent.opponent_id = HUMAN
            if bayes_agent.last_dice_counts is None:
                bayes_agent.last_dice_counts = {
                    AI: self.counts[AI] + (1 if loser == AI else 0),
                    HUMAN: self.counts[HUMAN] + (1 if loser == HUMAN else 0)
                }
            if bayes_agent.last_challenge is not None:
                obs = {
                    "opponent_dice_count": self.counts[HUMAN],
                    "my_dice": self.state.hands[AI] if self.state.hands else [],
                    "history": self.state.history,
                    "current_bid": self.state.current_bid
                }
                bayes_agent._resolve_last_challenge(obs)

        if self.counts[loser] == 0:
            self.winner = 1 - loser
            self.phase = "gameover"
            # Cập nhật chuỗi thắng/thua (streak)
            if self.winner == HUMAN:
                self.player_streak = max(1, self.player_streak + 1) if self.player_streak > 0 else 1
            else:
                self.player_streak = min(-1, self.player_streak - 1) if self.player_streak < 0 else -1
            self._learn_and_save()  # AI học thêm từ ván vừa đánh rồi lưu lại
            self._export_thinking_state()
        else:
            self.start_player = loser
            self._new_round()

    def _learn_and_save(self):
        """Sau mỗi ván kết thúc: train thêm 200 vòng CFR rồi auto-save weights.
        Giúp AI ngày càng khôn hơn theo cách chơi của người dùng."""
        from agents.cfr_agent import CFRAgent
        if not isinstance(self.ai, CFRAgent) or self.weights_path is None:
            return
        try:
            # Đưa cả việc tự chơi 200 vòng (train) và lưu file weights vào luồng ngầm (threading)
            # giúp màn hình chờ gameover của Pygame không bị đóng băng/khựng lại.
            import threading
            def bg_train_and_save():
                try:
                    self.ai.train(200)  # tự chơi ảo 200 vòng trong luồng ngầm (mất ~1-2s)
                    self.ai.save_weights(self.weights_path)  # lưu bộ não ngầm (mất ~0.2s)
                except Exception as ex:
                    print(f"Lỗi train/save ngầm: {ex}")
                    
            threading.Thread(target=bg_train_and_save, daemon=True).start()
            
            n = len(self.ai.strategy_table)
            self._log(f"💾 AI đang tự học & lưu ngầm 200 vòng — ngày càng khôn hơn!")
        except Exception as e:
            self._log(f"(Không kích hoạt học ngầm được: {e})")

    def _export_thinking_state(self, last_actor=None, last_action=None):
        import json
        import os
        
        mode = "Bayes (Ứng biến)"
        visits = 0
        distribution = {}
        p_truth = 0.5
        threshold = 0.5
        modifier = 1.0
        base_rate = 0.33
        action_type = "cược"
        
        from agents.cfr_agent import CFRAgent
        from agents.bayesian_agent import BayesianAgent
        
        if isinstance(self.ai, CFRAgent):
            mode = getattr(self.ai, "last_mode", "CFR (Tra bảng)")
            visits = getattr(self.ai, "last_visits", 0)
            distribution = getattr(self.ai, "last_distribution", {})
            
            if mode == "Bayes (Ứng biến)":
                p_truth = getattr(self.ai, "last_fallback_p_truth", 0.5)
                threshold = getattr(self.ai, "last_fallback_threshold", 0.5)
                modifier = getattr(self.ai, "last_fallback_modifier", 1.0)
                if self.ai.fallback_agent:
                    base_rate = getattr(self.ai.fallback_agent, "last_base_rate", 0.33)
                    action_type = getattr(self.ai.fallback_agent, "last_action_type", "cược")
            else:
                p_truth = None
                threshold = None
                modifier = None
                base_rate = None
                action_type = "tố cáo" if (last_action and isinstance(last_action, Challenge)) else "cược"
        elif isinstance(self.ai, BayesianAgent):
            mode = "Bayes (Ứng biến)"
            p_truth = getattr(self.ai, "last_p_truth", 0.5)
            threshold = getattr(self.ai, "last_threshold", 0.5)
            modifier = getattr(self.ai, "last_modifier", 1.0)
            base_rate = getattr(self.ai, "last_base_rate", 0.33)
            action_type = getattr(self.ai, "last_action_type", "cược")

        action_str = ""
        if last_action:
            if isinstance(last_action, Challenge):
                action_str = "LIAR! (Tố cáo)"
            elif isinstance(last_action, Bid):
                action_str = f"Cược {last_action.quantity} con mặt {last_action.face_value}"
        
        phase_map = {
            "human_turn": "Lượt của bạn",
            "ai_turn": "Lượt của AI (Đang suy nghĩ...)",
            "reveal": "Hạ bài phân định",
            "gameover": "Kết thúc ván đấu"
        }
        
        state_data = {
            "timestamp": pygame.time.get_ticks(),
            "phase": phase_map.get(self.phase, self.phase),
            "last_actor": "AI" if last_actor == AI else ("Bạn" if last_actor == HUMAN else None),
            "action_chosen": action_str,
            "mode": mode,
            "visits": visits,
            "p_truth": p_truth,
            "threshold": threshold,
            "modifier": modifier,
            "base_rate": base_rate,
            "distribution": {str(k): float(v) for k, v in distribution.items()},
            "ai_dice": self.state.hands[AI] if self.state.hands else [],
            "opponent_dice_count": self.counts[HUMAN],
            "current_bid": {
                "quantity": self.state.current_bid.quantity,
                "face_value": self.state.current_bid.face_value
            } if self.state.current_bid else None,
            "player_streak": self.player_streak,
            "bayesian_context_metrics": None
        }
        
        bayes_agent = None
        if isinstance(self.ai, BayesianAgent):
            bayes_agent = self.ai
        elif isinstance(self.ai, CFRAgent) and self.ai.fallback_agent:
            bayes_agent = self.ai.fallback_agent
            
        if bayes_agent:
            if bayes_agent.player_id is None:
                bayes_agent.player_id = AI
                bayes_agent.opponent_id = HUMAN
            if bayes_agent.last_dice_counts is None:
                bayes_agent.last_dice_counts = {
                    AI: self.counts[AI],
                    HUMAN: self.counts[HUMAN]
                }
            state_data["bayesian_context_metrics"] = {
                "resolved_opp_bids": bayes_agent.resolved_opp_bids,
                "opp_bluffs": bayes_agent.opp_bluffs,
                "resolved_opp_bids_low_dice": bayes_agent.resolved_opp_bids_low_dice,
                "opp_bluffs_low_dice": bayes_agent.opp_bluffs_low_dice,
                "resolved_opp_bids_winning": bayes_agent.resolved_opp_bids_winning,
                "opp_bluffs_winning": bayes_agent.opp_bluffs_winning,
                "resolved_opp_bids_losing": bayes_agent.resolved_opp_bids_losing,
                "opp_bluffs_losing": bayes_agent.opp_bluffs_losing,
                "resolved_opp_bids_high_qty": bayes_agent.resolved_opp_bids_high_qty,
                "opp_bluffs_high_qty": bayes_agent.opp_bluffs_high_qty,
                "resolved_opp_bids_face1": bayes_agent.resolved_opp_bids_face1,
                "opp_bluffs_face1": bayes_agent.opp_bluffs_face1,
                "our_bids_high_face": bayes_agent.our_bids_high_face,
                "opp_challenges_high_face": bayes_agent.opp_challenges_high_face
            }
            
        try:
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, "ai_thinking_state.json"), "w", encoding="utf-8") as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Lỗi ghi file ai_thinking_state.json: {e}")

    # ---------- sự kiện ----------
    def handle_click(self, pos):
        if self.phase == "gameover":
            return
        if self.phase == "reveal":
            self._resolve_reveal()
            return
        if self.phase != "human_turn":
            return

        b = self.buttons
        if b["q-"].hit(pos):
            self.sel_q = max(1, self.sel_q - 1)
        elif b["q+"].hit(pos):
            self.sel_q = min(self._total(), self.sel_q + 1)
        elif b["f-"].hit(pos):
            self.sel_f = max(MIN_FACE, self.sel_f - 1)
        elif b["f+"].hit(pos):
            self.sel_f = min(MAX_FACE, self.sel_f + 1)
        elif b["bid"].hit(pos) and self._sel_is_legal():
            self._apply_action(HUMAN, Bid(self.sel_q, self.sel_f))
        elif b["liar"].hit(pos) and self.state.current_bid is not None:
            self._apply_action(HUMAN, Challenge())

    def update(self):
        if self.phase == "ai_turn" and pygame.time.get_ticks() >= self.ai_think_at:
            obs = self.state.get_observation(AI)
            
            # Độ khó thích nghi:
            # - Bạn thắng liên tiếp >= 3 ván: AI chơi ngẫu nhiên/lắt léo để phòng thủ (mixed_strategy = True)
            # - Bạn đang thua: AI chơi an toàn tuyệt đối (mixed_strategy = False)
            # - Mặc định khác: True
            use_mixed = True
            if self.player_streak >= 3:
                use_mixed = True
            elif self.player_streak < 0:
                use_mixed = False

            from agents.cfr_agent import CFRAgent
            if isinstance(self.ai, CFRAgent):
                action = self.ai.act(obs, get_legal_actions(self.state), mixed_strategy=use_mixed)
            else:
                action = self.ai.act(obs, get_legal_actions(self.state))
            self._apply_action(AI, action)

    # ---------- vẽ ----------
    def _build_buttons(self):
        self.buttons = {
            "q-": Button((200, 560, 44, 44), "−", BLUE_BTN, self.f_mid),
            "q+": Button((310, 560, 44, 44), "+", BLUE_BTN, self.f_mid),
            "f-": Button((450, 560, 44, 44), "−", BLUE_BTN, self.f_mid),
            "f+": Button((560, 560, 44, 44), "+", BLUE_BTN, self.f_mid),
            "bid": Button((200, 630, 200, 56), "CƯỢC (BID)", GREEN_BTN, self.f_mid),
            "liar": Button((420, 630, 184, 56), "LIAR!", RED, self.f_mid),
        }

    def draw(self):
        s = self.screen
        s.fill(BG)
        mouse = pygame.mouse.get_pos()

        # Tiêu đề
        title = self.f_big.render("Liar's Dice", True, GOLD)
        s.blit(title, (40, 24))

        # --- Khu AI (trên) ---
        ai_reveal = self.phase in ("reveal", "gameover")
        
        # Nhãn AI phát sáng theo nhịp đập khi đang suy nghĩ
        ai_label_color = RED
        if self.phase == "ai_turn":
            import math
            pulse = (math.sin(pygame.time.get_ticks() * 0.007) + 1.0) / 2.0
            ai_label_color = tuple(int(c1 * (1.0 - pulse) + c2 * pulse) for c1, c2 in zip(RED, (255, 120, 120)))
        
        status_suffix = " (Đang suy nghĩ...)" if self.phase == "ai_turn" else ""
        self._draw_label(f"AI{status_suffix}", 40, 120, ai_label_color)
        
        ai_dice = self.state.hands[AI]
        hi_face = self.reveal["face"] if self.reveal else None
        for i in range(self.counts[AI]):
            val = ai_dice[i] if i < len(ai_dice) else 1
            draw_die(s, 168 + i * 96, 150, 80,
                     val if ai_reveal else 0,
                     hidden=not ai_reveal,
                     highlight=ai_reveal and (val == hi_face or val == 1) and hi_face is not None)

        # --- Bid hiện tại (giữa) ---
        cur = self.state.current_bid
        pygame.draw.rect(s, PANEL, (40, 270, 720, 96), border_radius=14)
        if cur is None:
            ct = self.f_mid.render("Chưa có cược — bạn ra cược trước", True, TEXT)
        else:
            ct = self.f_big.render(f"Cược: {cur.quantity} × mặt {cur.face_value}", True, WHITE)
        s.blit(ct, ct.get_rect(center=(400, 318)))

        # --- Khu người chơi (dưới) ---
        player_label_color = (120, 200, 255)
        if self.phase == "human_turn":
            import math
            pulse = (math.sin(pygame.time.get_ticks() * 0.007) + 1.0) / 2.0
            player_label_color = tuple(int(c1 * (1.0 - pulse) + c2 * pulse) for c1, c2 in zip((120, 200, 255), (200, 230, 255)))
            
        status_suffix = " (Lượt của bạn)" if self.phase == "human_turn" else ""
        self._draw_label(f"BẠN{status_suffix}", 40, 396, player_label_color)
        
        for i, val in enumerate(self.state.hands[HUMAN]):
            draw_die(s, 168 + i * 96, 426, 80, val,
                     highlight=hi_face is not None and ai_reveal and (val == hi_face or val == 1))

        # --- Điều khiển / trạng thái ---
        if self.phase == "human_turn":
            self._draw_controls(mouse)
        elif self.phase == "ai_turn":
            dots = "." * ((pygame.time.get_ticks() // 250) % 4)
            self._draw_center_msg(f"AI đang suy nghĩ{dots}", GOLD)
        elif self.phase == "reveal":
            self._draw_reveal()
        elif self.phase == "gameover":
            self._draw_gameover()

        pygame.display.flip()

    def _draw_label(self, text, x, y, color):
        self.screen.blit(self.f_mid.render(text, True, color), (x, y))

    @staticmethod
    def _wrap_text(text, font, max_w):
        """Ngắt chuỗi thành nhiều dòng để mỗi dòng không vượt quá max_w px (word-wrap)."""
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            trial = w if not cur else cur + " " + w
            if font.size(trial)[0] <= max_w:
                cur = trial
                continue
            if cur:
                lines.append(cur)
            # Từ đơn quá dài: cắt cứng theo ký tự cho vừa khung.
            while font.size(w)[0] > max_w and len(w) > 1:
                cut = len(w)
                while cut > 1 and font.size(w[:cut])[0] > max_w:
                    cut -= 1
                lines.append(w[:cut])
                w = w[cut:]
            cur = w
        if cur:
            lines.append(cur)
        return lines

    def _draw_controls(self, mouse):
        s = self.screen
        # Steppers
        s.blit(self.f_sm.render("Số lượng", True, DIM), (200, 532))
        s.blit(self.f_sm.render("Mặt", True, DIM), (450, 532))
        qv = self.f_mid.render(str(self.sel_q), True, WHITE)
        s.blit(qv, qv.get_rect(center=(277, 582)))
        fv = self.f_mid.render(str(self.sel_f), True, WHITE)
        s.blit(fv, fv.get_rect(center=(527, 582)))
        legal = self._sel_is_legal()
        self.buttons["bid"].enabled = legal
        self.buttons["liar"].enabled = self.state.current_bid is not None
        for key in ("q-", "q+", "f-", "f+", "bid", "liar"):
            self.buttons[key].draw(s, mouse)
        if not legal:
            warn = self.f_sm.render("(nước cược phải cao hơn cược hiện tại)", True, (255, 200, 120))
            s.blit(warn, warn.get_rect(center=(400, 696)))

    def _draw_center_msg(self, text, color):
        box = pygame.Rect(200, 600, 400, 90)
        pygame.draw.rect(self.screen, PANEL_LIGHT, box, border_radius=12)
        t = self.f_mid.render(text, True, color)
        self.screen.blit(t, t.get_rect(center=box.center))

    def _draw_reveal(self):
        rv = self.reveal
        loser_txt = "Bạn THUA vòng" if rv["loser"] == HUMAN else "AI THUA vòng"
        color = RED if rv["loser"] == HUMAN else GREEN_BTN
        box = pygame.Rect(40, 600, 720, 96)
        pygame.draw.rect(self.screen, PANEL_LIGHT, box, border_radius=12)
        l1 = self.f_mid.render(
            f"Mặt {rv['face']}: thực tế {rv['actual']} (cược {rv['claimed']}) → {loser_txt}",
            True, color)
        self.screen.blit(l1, l1.get_rect(center=(400, 624)))
        l2 = self.f_sm.render("Bấm chuột để sang vòng tiếp theo", True, DIM)
        self.screen.blit(l2, l2.get_rect(center=(400, 668)))

    def _draw_gameover(self):
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        txt = "🎉 BẠN THẮNG! 🎉" if self.winner == HUMAN else "AI THẮNG 🤖"
        col = GOLD if self.winner == HUMAN else RED
        t = self.f_big.render(txt, True, col)
        self.screen.blit(t, t.get_rect(center=(W // 2, H // 2 - 20)))
        h = self.f_sm.render("R: chơi lại   |   Esc: thoát", True, TEXT)
        self.screen.blit(h, h.get_rect(center=(W // 2, H // 2 + 40)))

    def restart(self):
        self.counts = [self.dice_per_player, self.dice_per_player]
        self.start_player = HUMAN
        self.winner = None
        self.log_lines = []
        self.round_history = []
        if hasattr(self.ai, "reset"):
            self.ai.reset()
        self._new_round()



def play_gui(ai_agent: Agent | None = None, seed=None, dice_per_player=5, weights_path=None):
    # Tự động mở Web Dashboard song song trong tiến trình ngầm
    try:
        import subprocess
        import sys
        import os
        dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "web_dashboard.py"))
        
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            
        subprocess.Popen(
            [sys.executable, dashboard_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs
        )
        print("======================================================================")
        print("🚀 Web Dashboard đã tự động mở song song tại: http://localhost:8000")
        print("======================================================================")
    except Exception as e:
        print(f"Không thể khởi động Web Dashboard tự động: {e}")

    ai = ai_agent or RandomAgent(name="AI")
    pygame.init()
    pygame.display.set_caption("Liar's Dice — Bạn vs AI")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    game = LiarsDiceGUI(screen, ai, seed=seed, dice_per_player=dice_per_player,
                        weights_path=weights_path)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r and game.phase == "gameover":
                    game.restart()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                game.handle_click(e.pos)
        game.update()
        game.draw()
        clock.tick(FPS)

    pygame.quit()


def _smoke_test():
    """Kiểm tra không cần màn hình thật (CI/headless): vẽ các trạng thái chính."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    game = LiarsDiceGUI(screen, RandomAgent(name="AI"), seed=1)
    game.draw()                                   # human_turn
    game._apply_action(HUMAN, Bid(2, 3))
    game.draw()                                   # ai_turn
    game.update()                                 # AI hành động
    game.draw()
    # Ép vào trạng thái reveal để vẽ thử.
    game.state.current_bid = Bid(2, 3)
    game.state.history = [(HUMAN, Bid(2, 3))]
    game.state.current_player = AI
    game._apply_action(AI, Challenge())
    game.draw()                                   # reveal
    pygame.quit()
    print("GUI smoke test OK")


if __name__ == "__main__":
    import sys
    import os
    # Cấu hình đường dẫn để có thể chạy trực tiếp file gui.py từ Editor/IDE
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from agents.cfr_agent import CFRAgent

    # Ưu tiên file .gz đã commit; nếu không có thì lưu vào .gz mới
    _weights_gz   = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'cfr_heavy.weights.json.gz')
    _weights_json = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'cfr_heavy.weights.json')
    _save_path = _weights_gz  # luôn lưu vào bản .gz (nén nhỏ)

    cfr_bot = CFRAgent("CFRAgent")
    for _w in (_weights_gz, _weights_json):
        if os.path.exists(_w):
            cfr_bot.load_weights(_w)
            print(f"Đã nạp {len(cfr_bot.strategy_table)} trạng thái từ {os.path.basename(_w)}")
            break
    else:
        print("Không tìm thấy weights, train nhanh 5000 vòng...")
        cfr_bot.train(5000)

    # weights_path → sau mỗi ván AI sẽ tự học thêm 200 vòng rồi lưu lại
    # Mỗi lần bạn chơi xong, AI ngày càng "hiểu" cách chơi của bạn hơn!
    play_gui(cfr_bot, weights_path=_save_path)
