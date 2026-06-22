"""Biểu đồ cho phần đánh giá. Dùng backend 'Agg' để chạy không cần màn hình."""
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)


def plot_win_matrix(result: Dict, path: str = "results/win_matrix.png") -> str:
    """Vẽ heatmap ma trận win-rate (ô [hàng][cột] = % hàng thắng cột)."""
    _ensure_dir(path)
    names = result["names"]
    wm = result["win_matrix"]
    ng = result["num_games"]
    n = len(names)

    grid = [[(wm[a][b] / ng * 100 if a != b else float("nan")) for b in names] for a in names]

    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.6 * n + 2))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("Đối thủ (cột)"); ax.set_ylabel("Agent (hàng)")
    ax.set_title(f"Ma trận tỷ lệ thắng (cân bằng ghế, {ng} ván/cặp)")

    for i in range(n):
        for j in range(n):
            txt = "—" if i == j else f"{grid[i][j]:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=10)

    fig.colorbar(im, ax=ax, label="% thắng")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_overall_winrate(result: Dict, path: str = "results/agent_stats.png") -> str:
    """Vẽ cột win-rate tổng của từng agent."""
    _ensure_dir(path)
    names = result["names"]
    rates = [result["overall_winrate"][a] * 100 for a in names]

    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 4))
    bars = ax.bar(names, rates, color="steelblue")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% (hòa)")
    ax.set_ylabel("Win-rate tổng (%)"); ax.set_ylim(0, 100)
    ax.set_title("Win-rate trung bình trên mọi đối thủ")
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 1, f"{r:.1f}%", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_infoset_growth(history: Dict, path: str = "results/cfr_infoset_growth.png") -> str:
    """Vẽ số infoset tích luỹ theo số vòng lặp self-play.

    Minh hoạ trực quan nút thắt DATA-STARVATION: không gian infoset tăng gần tuyến tính (khoá
    bằng bộ xúc xắc chính xác) trong khi ngân sách huấn luyện hữu hạn ⇒ mỗi infoset chỉ được
    thăm vài lần ⇒ chiến lược học chậm hội tụ. (Xem §4.2, §5.2 của báo cáo.)
    """
    _ensure_dir(path)
    iters = history["iterations"]
    n_infoset = history.get("n_infoset")
    if not n_infoset:
        raise ValueError("history thiếu khoá 'n_infoset' — cần đo lại với phiên bản mới.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iters, n_infoset, marker="o", color="darkorange")
    ax.set_xlabel("Số vòng lặp self-play")
    ax.set_ylabel("Số infoset đã gặp (tích luỹ)")
    ax.set_title("Tăng trưởng không gian infoset (nút thắt data-starvation)")
    ax.grid(True, alpha=0.3)
    # Chú thích số trung bình lượt thăm/infoset ở mốc cuối để nhấn ý data-starvation.
    if iters and n_infoset[-1] > 0:
        avg_visits = iters[-1] / n_infoset[-1]
        ax.annotate(f"≈ {avg_visits:.2f} lượt thăm / infoset ở {iters[-1]} vòng",
                    xy=(iters[-1], n_infoset[-1]), xytext=(0.05, 0.9),
                    textcoords="axes fraction", color="dimgray")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_convergence(history: Dict, path: str = "results/cfr_convergence.png") -> str:
    """Vẽ đường hội tụ CFR: regret trung bình (↓) và win-rate vs Probabilistic."""
    _ensure_dir(path)
    iters = history["iterations"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Dùng raw_bound (cận trên exploitability ĐÚNG CHUẨN) nếu có; tránh avg_regret (gây
    # hiểu nhầm do chia thêm cho #infoset).
    bound = history.get("raw_bound", history["avg_regret"])
    ax1.plot(iters, bound, marker="o", color="crimson")
    ax1.set_xlabel("Số vòng lặp self-play")
    ax1.set_ylabel("Cận trên exploitability: Σ regret⁺ / T")
    ax1.set_title("Hội tụ CFR: cận trên exploitability ↓")
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, [w * 100 for w in history["winrate_vs_prob"]], marker="s", color="seagreen")
    ax2.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Số vòng lặp self-play")
    ax2.set_ylabel("Win-rate vs ProbabilisticAgent (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Sức mạnh thực nghiệm theo huấn luyện")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
