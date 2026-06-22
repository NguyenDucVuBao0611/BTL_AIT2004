import pytest
from game.actions import Bid, Challenge
from agents.bayesian_agent import BayesianAgent


def test_bayesian_agent_initialization():
    agent = BayesianAgent(name="Bayes")
    # Chưa có dữ liệu nào được phân định
    assert agent.resolved_opp_bids == 0
    assert agent.opp_bluffs == 0
    # Kỳ vọng tiên nghiệm Beta(1, 2) = 1/3
    assert abs(agent.bluff_rate() - 1/3) < 1e-9

    observation = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 5,
        "current_bid": None,
        "history": [],
    }
    action = agent.act(observation, [Bid(1, 2), Bid(1, 3)])
    assert agent.player_id == 0
    assert agent.opponent_id == 1
    # Chưa học gì ⇒ ngưỡng = 0.4 + (1/3)*0.2 ≈ 0.4667
    assert isinstance(action, Bid)


def test_posterior_unbiased_for_caught_bluff():
    """Đối thủ bị tố cáo và mất xúc xắc ⇒ ghi nhận 1 bluff trên 1 cược phân định."""
    agent = BayesianAgent(name="Bayes")

    # Lượt đầu của ta để xác định player_id và ghi nhận số xúc xắc ban đầu (3 vs 3)
    obs1 = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 3,
        "current_bid": None,
        "history": [],
    }
    agent.act(obs1, [Bid(1, 2)])

    # Đối thủ (id=1) cược, rồi ta (id=0) Challenge
    agent.observe(Bid(3, 4), 1)
    agent.observe(Challenge(), 0)   # bidder = 1 (đối thủ)

    # Vòng mới: số xúc xắc đối thủ giảm 3 -> 2 ⇒ cược của họ là bluff bị bắt
    obs2 = {
        "player_id": 0,
        "my_dice": [1, 2, 6],
        "opponent_dice_count": 2,
        "current_bid": None,
        "history": [],
    }
    agent.act(obs2, [Bid(1, 2)])

    assert agent.resolved_opp_bids == 1
    assert agent.opp_bluffs == 1
    # Hậu nghiệm = (1 + 1) / (1 + 1 + 2) = 0.5
    assert abs(agent.bluff_rate() - 0.5) < 1e-9


def test_posterior_counts_truthful_resolution():
    """Đối thủ bị tố cáo nhưng nói THẬT (ta mất xúc xắc) ⇒ mẫu n tăng, k không tăng.

    Đây chính là ca trước đây bị bỏ sót gây thiên lệch ước lượng xuống thấp.
    """
    agent = BayesianAgent(name="Bayes")

    obs1 = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 3,
        "current_bid": None,
        "history": [],
    }
    agent.act(obs1, [Bid(1, 2)])

    agent.observe(Bid(2, 4), 1)     # đối thủ cược
    agent.observe(Challenge(), 0)   # ta tố cáo, bidder = đối thủ

    # Vòng mới: đối thủ GIỮ NGUYÊN 3 xúc xắc, còn TA giảm 3 -> 2 ⇒ đối thủ nói thật
    obs2 = {
        "player_id": 0,
        "my_dice": [1, 2],
        "opponent_dice_count": 3,
        "current_bid": None,
        "history": [],
    }
    agent.act(obs2, [Bid(1, 2)])

    assert agent.resolved_opp_bids == 1
    assert agent.opp_bluffs == 0
    # Hậu nghiệm = (0 + 1) / (1 + 1 + 2) = 0.25
    assert abs(agent.bluff_rate() - 0.25) < 1e-9


def test_my_own_bid_resolution_ignored():
    """Khi TA là người bị tố cáo, kết quả không nói gì về bluff của đối thủ ⇒ bỏ qua."""
    agent = BayesianAgent(name="Bayes")

    obs1 = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 3,
        "current_bid": None,
        "history": [],
    }
    agent.act(obs1, [Bid(1, 2)])

    # Ta cược, đối thủ Challenge ⇒ bidder = ta (id=0)
    agent.observe(Bid(2, 4), 0)
    agent.observe(Challenge(), 1)

    obs2 = {
        "player_id": 0,
        "my_dice": [1, 2],
        "opponent_dice_count": 3,
        "current_bid": None,
        "history": [],
    }
    agent.act(obs2, [Bid(1, 2)])

    # Không cập nhật gì về đối thủ
    assert agent.resolved_opp_bids == 0
    assert agent.opp_bluffs == 0


def test_threshold_increases_with_bluffing():
    """Đối thủ bluff nhiều ⇒ bluff_rate cao ⇒ ngưỡng động cao hơn (dễ Challenge)."""
    honest = BayesianAgent(name="Honest_view")
    bluffy = BayesianAgent(name="Bluffy_view")

    honest.resolved_opp_bids, honest.opp_bluffs = 10, 1
    bluffy.resolved_opp_bids, bluffy.opp_bluffs = 10, 9

    th_honest = max(0.4, min(0.6, 0.4 + honest.bluff_rate() * 0.2))
    th_bluffy = max(0.4, min(0.6, 0.4 + bluffy.bluff_rate() * 0.2))
    assert th_bluffy > th_honest
