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


def plot_overall_winrate(result: Dict, path: str = "results/agent_stats.png",
                         exclude: tuple = ("RandomAgent",)) -> str:
    """Vẽ cột win-rate đối đầu, LOẠI các đối thủ tầm thường trong `exclude` khỏi trung bình.

    Mỗi cột = win-rate trung bình của agent trên các đối thủ KHÔNG tầm thường, kèm thanh sai
    số 95% (xấp xỉ chuẩn). Loại RandomAgent vì đập một baseline tầm thường (mọi agent thắng
    ~100%) sẽ thổi phồng win-rate của tất cả và che mất khác biệt thật giữa các agent có
    chiến lược — vốn xúm quanh mốc Nash 50%.
    """
    import math
    _ensure_dir(path)
    wm = result["win_matrix"]
    ng = result["num_games"]
    names = [n for n in result["names"] if n not in exclude]

    rates, errs = [], []
    for a in names:
        opps = [b for b in names if b != a]
        n_games = len(opps) * ng
        wins = sum(wm[a][b] for b in opps)
        p = wins / n_games if n_games else 0.0
        rates.append(p * 100)
        se = math.sqrt(p * (1 - p) / n_games) if n_games else 0.0
        errs.append(1.96 * se * 100)  # khoảng tin cậy 95%

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names) + 2), 4.2))
    bars = ax.bar(names, rates, yerr=errs, capsize=6, color="steelblue",
                  error_kw={"ecolor": "dimgray"})
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% (mốc Nash)")
    ax.set_ylabel("Win-rate đối đầu (%)"); ax.set_ylim(0, 100)
    excl = ", ".join(exclude)
    ax.set_title(f"Win-rate đối đầu giữa agent có chiến lược (bỏ {excl}, CI 95%)")
    for bar, r, e in zip(bars, rates, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, r + e + 1.8, f"{r:.1f}%", ha="center")
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
