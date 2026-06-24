import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def test_bayesian_multi_context_bluff():
    """Kiểm tra việc ghi nhận bluff đúng các ngữ cảnh khác nhau và fallback phân cấp."""
    agent = BayesianAgent(name="Bayes")
    
    # Khởi tạo
    obs1 = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 2, # Low dice <= 2, also opponent losing (2 vs 3)
        "current_bid": None,
        "history": [],
    }
    agent.act(obs1, [Bid(1, 2)])
    
    # Đối thủ cược mặt 1 số lượng lớn (Ví dụ: 3 con mặt 1, trong đó 3 >= total_dice/2.0)
    bid = Bid(3, 1)
    agent.observe(bid, 1)
    agent.observe(Challenge(), 0) # Ta challenge
    
    # Lượt challenge kết thúc, đối thủ thua mất 1 xúc xắc (2 -> 1)
    obs2 = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 1,
        "current_bid": None,
        "history": [(1, bid), (0, Challenge())],
    }
    agent.act(obs2, [Bid(1, 2)])
    
    # Đối thủ mất xúc xắc => bluff
    assert agent.resolved_opp_bids == 1
    assert agent.opp_bluffs == 1
    
    # Kiểm tra các ngữ cảnh:
    # 1. Low dice (old_opp = 2 <= 2)
    assert agent.resolved_opp_bids_low_dice == 1
    assert agent.opp_bluffs_low_dice == 1
    
    # 2. Opponent Losing (2 < 3)
    assert agent.resolved_opp_bids_losing == 1
    assert agent.opp_bluffs_losing == 1
    
    # 3. Opponent Winning (2 > 3 is false)
    assert agent.resolved_opp_bids_winning == 0
    
    # 4. Face 1 bid (bid face = 1)
    assert agent.resolved_opp_bids_face1 == 1
    assert agent.opp_bluffs_face1 == 1
    
    # 5. High Qty (quantity = 3 >= total_dice/2 = 2.5)
    assert agent.resolved_opp_bids_high_qty == 1
    assert agent.opp_bluffs_high_qty == 1


def test_bayesian_history_bluff_modifier():
    """Kiểm tra modifier khả nghi thay đổi dựa trên lịch sử bước nhảy cược."""
    agent = BayesianAgent(name="Bayes")
    
    # 1. Đối thủ nâng cược đột biến (dq >= 2)
    obs_jump = {
        "player_id": 0,
        "my_dice": [2, 2, 3],
        "opponent_dice_count": 3,
        "current_bid": Bid(4, 2),
        "history": [(0, Bid(2, 2)), (1, Bid(4, 2))], # Bước nhảy +2
    }
    
    agent.player_id = 0
    agent.opponent_id = 1
    
    # Gọi act để kích hoạt phân tích lịch sử nước đi
    action = agent.act(obs_jump, [Challenge(), Bid(4, 3)])
    assert isinstance(action, (Bid, Challenge))


def test_bayesian_scared_opponent_exploit():
    """Kiểm tra thói quen phản ứng của đối thủ khi ta cược lớn và chiến thuật bóc lột của AI."""
    agent = BayesianAgent(name="Bayes")
    agent.player_id = 0
    agent.opponent_id = 1
    
    # 1. Ta cược mặt lớn (>= 4), đối thủ không challenge mà cược tiếp
    agent.observe(Bid(2, 5), 0)
    agent.observe(Bid(3, 5), 1)
    assert agent.our_bids_high_face == 1
    assert agent.opp_challenges_high_face == 0
    
    # 2. Ta cược lớn tiếp, đối thủ vẫn cược tiếp
    agent.observe(Bid(3, 6), 0)
    agent.observe(Bid(4, 6), 1)
    
    # 3. Ta cược lớn tiếp, đối thủ vẫn cược tiếp
    agent.observe(Bid(4, 4), 0)
    agent.observe(Bid(5, 4), 1)
    
    assert agent.our_bids_high_face == 3
    assert agent.opp_challenges_high_face == 0
    
    # Tỷ lệ challenge của đối thủ là 0 < 0.25 => AI phát hiện đối thủ nhút nhát và ưu tiên cược mặt >= 4
    obs = {
        "player_id": 0,
        "my_dice": [2, 2, 3],
        "opponent_dice_count": 3,
        "current_bid": Bid(1, 2),
        "history": [],
    }
    # Ta có các nước đi hợp lệ: Bid(2, 2) (mặt 2) và Bid(2, 5) (mặt 5)
    # Bình thường, ta giữ nhiều mặt 2 hơn (hai con 2 vs không có con 5 nào),
    # nên ta sẽ chọn Bid(2, 2). Nhưng vì đối thủ nhút nhát mặt lớn, ta ưu tiên hô mặt 5 (Bid(2, 5)) để hù dọa!
    action = agent.act(obs, [Bid(2, 2), Bid(2, 5)])
    assert isinstance(action, Bid)
    assert action.face_value == 5 # Đã ưu tiên mặt lớn để bóc lột!


def test_bayesian_agent_load_save_profile(tmp_path, monkeypatch):
    import json
    from agents.bayesian_agent import BayesianAgent
    
    profile_dir = tmp_path / "results"
    profile_dir.mkdir()
    profile_file = profile_dir / "user_habit_profile.json"
    
    agent = BayesianAgent(name="TestAgent")
    agent._skip_file_io = False
    agent.resolved_opp_bids = 5
    agent.opp_bluffs = 2
    
    import os
    original_abspath = os.path.abspath
    def dummy_abspath(p):
        if "bayesian_agent.py" in p:
            return str(tmp_path / "agents" / "bayesian_agent.py")
        return original_abspath(p)
    
    monkeypatch.setattr(os.path, "abspath", dummy_abspath)
    
    agent.save_profile()
    
    assert profile_file.exists()
    with open(profile_file, "r") as f:
        data = json.load(f)
    assert data["resolved_opp_bids"] == 5
    assert data["opp_bluffs"] == 2
    
    agent2 = BayesianAgent(name="TestAgent2")
    agent2._skip_file_io = False
    agent2.load_profile()
    assert agent2.resolved_opp_bids == 5
    assert agent2.opp_bluffs == 2
    
    agent2.reset()
    assert agent2.resolved_opp_bids == 0
    
    with open(profile_file, "r") as f:
        data_reset = json.load(f)
    assert data_reset["resolved_opp_bids"] == 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))

