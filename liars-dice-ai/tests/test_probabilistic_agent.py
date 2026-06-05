import pytest
from game.actions import Bid, Challenge
from agents.probabilistic_agent import ProbabilisticAgent

def test_probabilistic_agent_first_bid():
    agent = ProbabilisticAgent(name="ProbAgent", threshold=0.5)
    
    # Giả lập observation cho lượt đi đầu tiên (current_bid = None)
    # Tay bài: [1, 2, 4, 4, 5] -> Mặt 4 xuất hiện nhiều nhất (hai viên 4 + một viên 1 = 3 viên)
    observation = {
        "player_id": 0,
        "my_dice": [1, 2, 4, 4, 5],
        "opponent_dice_count": 5,
        "current_bid": None,
        "history": []
    }
    
    # Một số hành động cược hợp lệ
    legal_actions = [
        Bid(1, 2), Bid(1, 4), Bid(2, 4), Bid(1, 5), Bid(3, 4)
    ]
    
    # Agent phải chọn Bid của mặt 4 có quantity nhỏ nhất
    action = agent.act(observation, legal_actions)
    assert isinstance(action, Bid)
    assert action.face_value == 4
    assert action.quantity == 1

def test_probabilistic_agent_challenge_on_low_probability():
    agent = ProbabilisticAgent(name="ProbAgent", threshold=0.5)
    
    # Giả lập observation: đối thủ cược quá cao
    # Tay ta: [2, 3, 5] -> Không có mặt 4 nào, không có mặt 1 wild
    # Đối thủ cược: 5 viên mặt 4. Đối thủ chỉ còn 3 xúc xắc ẩn
    # k = 5 - 0 = 5. n = 3. Do k > n -> P_truth = 0.0 < 0.5 -> Nên Challenge
    observation = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 3,
        "current_bid": Bid(5, 4),
        "history": [(1, Bid(5, 4))]
    }
    
    legal_actions = [
        Challenge(),
        Bid(5, 5), Bid(5, 6), Bid(6, 1)
    ]
    
    action = agent.act(observation, legal_actions)
    assert isinstance(action, Challenge)

def test_probabilistic_agent_raise_on_high_probability():
    agent = ProbabilisticAgent(name="ProbAgent", threshold=0.5)
    
    # Giả lập observation: đối thủ cược cực kỳ an toàn
    # Tay ta: [1, 4, 4] -> Có 3 viên mặt 4 (tính cả mặt 1 wild)
    # Đối thủ cược: 3 viên mặt 4. Đối thủ còn 3 xúc xắc ẩn
    # k = 3 - 3 = 0. n = 3. Do k <= 0 -> P_truth = 1.0 >= 0.5 -> Nên nâng cược chứ không Challenge
    observation = {
        "player_id": 0,
        "my_dice": [1, 4, 4],
        "opponent_dice_count": 3,
        "current_bid": Bid(3, 4),
        "history": [(1, Bid(3, 4))]
    }
    
    legal_actions = [
        Challenge(),
        Bid(3, 5), Bid(3, 6), Bid(4, 1), Bid(4, 4)
    ]
    
    action = agent.act(observation, legal_actions)
    assert isinstance(action, Bid)
    # Phải nâng cược. Do tay ta giữ nhiều mặt 4 nhất (3 viên), nên chọn cược mặt 4 có số lượng nhỏ nhất hợp lệ
    # Trong legal_actions, Bid(4, 4) là hợp lệ cho mặt 4
    assert action == Bid(4, 4)
