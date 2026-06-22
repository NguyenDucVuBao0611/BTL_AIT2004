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
W, H = 1024, 720
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
    def __init__(self, screen, ai: Agent, seed=None, dice_per_player=5):
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
        self.message = ""
        self.reveal = None          # dict thông tin lúc lật bài
        self.winner = None
        self.ai_think_at = None
        self.sel_q = 1
        self.sel_f = MIN_FACE
        self.buttons: dict[str, Button] = {}
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
            self.reveal = {
                "loser": loser, "actual": actual, "claimed": claimed,
                "face": face, "challenger": actor,
            }
            self.phase = "reveal"

    def _resolve_reveal(self):
        loser = self.reveal["loser"]
        self.counts[loser] -= 1
        self._log(f"{'Bạn' if loser == HUMAN else 'AI'} thua vòng, mất 1 xúc xắc "
                  f"→ {self.counts}")
        if self.counts[loser] == 0:
            self.winner = 1 - loser
            self.phase = "gameover"
        else:
            self.start_player = loser
            self._new_round()

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
            action = self.ai.act(obs, get_legal_actions(self.state))
            self._apply_action(AI, action)

    # ---------- vẽ ----------
    def _build_buttons(self):
        self.buttons = {
            "q-": Button((360, 560, 44, 44), "−", BLUE_BTN, self.f_mid),
            "q+": Button((470, 560, 44, 44), "+", BLUE_BTN, self.f_mid),
            "f-": Button((610, 560, 44, 44), "−", BLUE_BTN, self.f_mid),
            "f+": Button((720, 560, 44, 44), "+", BLUE_BTN, self.f_mid),
            "bid": Button((360, 630, 200, 56), "CƯỢC (BID)", GREEN_BTN, self.f_mid),
            "liar": Button((580, 630, 184, 56), "LIAR!", RED, self.f_mid),
        }

    def draw(self):
        s = self.screen
        s.fill(BG)
        mouse = pygame.mouse.get_pos()

        # Tiêu đề
        title = self.f_big.render("Liar's Dice", True, GOLD)
        s.blit(title, (40, 24))
        sub = self.f_sm.render(f"Đối thủ: {self.ai.name}", True, DIM)
        s.blit(sub, (44, 78))

        # --- Khu AI (trên) ---
        ai_reveal = self.phase in ("reveal", "gameover")
        self._draw_label("AI", 40, 120, RED)
        ai_dice = self.state.hands[AI]
        hi_face = self.reveal["face"] if self.reveal else None
        for i in range(self.counts[AI]):
            val = ai_dice[i] if i < len(ai_dice) else 1
            draw_die(s, 120 + i * 96, 150, 80,
                     val if ai_reveal else 0,
                     hidden=not ai_reveal,
                     highlight=ai_reveal and (val == hi_face or val == 1) and hi_face is not None)

        # --- Bid hiện tại (giữa) ---
        cur = self.state.current_bid
        pygame.draw.rect(s, PANEL, (40, 270, 600, 96), border_radius=14)
        if cur is None:
            ct = self.f_mid.render("Chưa có cược — bạn ra cược trước", True, TEXT)
        else:
            ct = self.f_big.render(f"Cược: {cur.quantity} × mặt {cur.face_value}", True, WHITE)
        s.blit(ct, ct.get_rect(midleft=(64, 318)))

        # --- Khu người chơi (dưới) ---
        self._draw_label("BẠN", 40, 396, (120, 200, 255))
        for i, val in enumerate(self.state.hands[HUMAN]):
            draw_die(s, 120 + i * 96, 426, 80, val,
                     highlight=hi_face is not None and ai_reveal and (val == hi_face or val == 1))

        # --- Log (phải) ---
        log_x, log_y, log_w, log_h = 680, 120, 304, 470
        pygame.draw.rect(s, PANEL, (log_x, log_y, log_w, log_h), border_radius=14)
        lt = self.f_sm.render("Diễn biến", True, GOLD)
        s.blit(lt, (700, 132))
        max_w = log_x + log_w - 700 - 14    # bề rộng tối đa của 1 dòng chữ
        line_h = 26
        top = 168
        budget = (log_y + log_h - top - 8) // line_h   # số dòng vẽ vừa khung
        wrapped: list[str] = []
        for line in self.log_lines:
            wrapped.extend(self._wrap_text(line, self.f_log, max_w))
        for i, wl in enumerate(wrapped[-budget:]):
            s.blit(self.f_log.render(wl, True, TEXT), (700, top + i * line_h))

        # --- Điều khiển / trạng thái ---
        if self.phase == "human_turn":
            self._draw_controls(mouse)
        elif self.phase == "ai_turn":
            self._draw_center_msg("AI đang suy nghĩ…", GOLD)
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
        s.blit(self.f_sm.render("Số lượng", True, DIM), (360, 532))
        s.blit(self.f_sm.render("Mặt", True, DIM), (610, 532))
        qv = self.f_mid.render(str(self.sel_q), True, WHITE)
        s.blit(qv, qv.get_rect(center=(437, 582)))
        fv = self.f_mid.render(str(self.sel_f), True, WHITE)
        s.blit(fv, fv.get_rect(center=(687, 582)))
        legal = self._sel_is_legal()
        self.buttons["bid"].enabled = legal
        self.buttons["liar"].enabled = self.state.current_bid is not None
        for key in ("q-", "q+", "f-", "f+", "bid", "liar"):
            self.buttons[key].draw(s, mouse)
        if not legal:
            warn = self.f_sm.render("(nước cược phải cao hơn cược hiện tại)", True, (255, 200, 120))
            s.blit(warn, (360, 700 - 4))

    def _draw_center_msg(self, text, color):
        box = pygame.Rect(360, 600, 404, 90)
        pygame.draw.rect(self.screen, PANEL_LIGHT, box, border_radius=12)
        t = self.f_mid.render(text, True, color)
        self.screen.blit(t, t.get_rect(center=box.center))

    def _draw_reveal(self):
        rv = self.reveal
        loser_txt = "Bạn THUA vòng" if rv["loser"] == HUMAN else "AI THUA vòng"
        color = RED if rv["loser"] == HUMAN else GREEN_BTN
        box = pygame.Rect(40, 600, 600, 96)
        pygame.draw.rect(self.screen, PANEL_LIGHT, box, border_radius=12)
        l1 = self.f_mid.render(
            f"Mặt {rv['face']}: thực tế {rv['actual']} (cược {rv['claimed']}) → {loser_txt}",
            True, color)
        self.screen.blit(l1, (60, 612))
        l2 = self.f_sm.render("Bấm chuột để sang vòng tiếp theo", True, DIM)
        self.screen.blit(l2, (60, 656))

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
        if hasattr(self.ai, "reset"):
            self.ai.reset()
        self._new_round()


def play_gui(ai_agent: Agent | None = None, seed=None, dice_per_player=5):
    ai = ai_agent or RandomAgent(name="AI")
    pygame.init()
    pygame.display.set_caption("Liar's Dice — Bạn vs AI")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    game = LiarsDiceGUI(screen, ai, seed=seed, dice_per_player=dice_per_player)

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
    _smoke_test()
