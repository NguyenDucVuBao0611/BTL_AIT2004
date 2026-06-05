import pytest
from game.actions import Bid, Challenge
from agents.bayesian_agent import BayesianAgent

def test_bayesian_agent_initialization():
    agent = BayesianAgent(name="Bayes")
    assert agent.bluff_opportunities == 0
    assert agent.actual_bluffs == 0
    
    # Giả lập lượt đi đầu tiên
    observation = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 5,
        "current_bid": None,
        "history": []
    }
    legal_actions = [Bid(1, 2), Bid(1, 3)]
    
    # Lần act đầu tiên sẽ xác định player_id và tính threshold mặc định
    action = agent.act(observation, legal_actions)
    assert agent.player_id == 0
    assert agent.opponent_id == 1
    # bluff_opportunities = 0, bluff_rate = 0.3, threshold = 0.4 + 0.3 * 0.2 = 0.46
    assert agent.bluff_opportunities == 0
    assert agent.actual_bluffs == 0

def test_bayesian_agent_bluff_opportunity():
    agent = BayesianAgent(name="Bayes")
    
    # 1. Đối thủ cược trước khi ta kịp act (Opponent đi trước)
    # Opponent (ID = 1) cược 3 viên mặt 4
    agent.observe(Bid(3, 4), 1)
    
    # 2. Ta chuẩn bị act
    observation = {
        "player_id": 0,
        "my_dice": [2, 3, 5],            # Ta có 0 viên mặt 4 (không có mặt 1 wild)
        "opponent_dice_count": 3,
        "current_bid": Bid(3, 4),
        "history": [(1, Bid(3, 4))]
    }
    legal_actions = [Challenge(), Bid(3, 5)]
    
    # Khi act() được gọi, agent sẽ xử lý cược trì hoãn của đối thủ
    agent.act(observation, legal_actions)
    
    # my_count = 0, opp_dice_count = 3
    # E = my_count + 1/3 * opp_dice_count = 0 + 1 = 1
    # Lượng cược Q = 3 > E = 1 -> Bluff Opportunity!
    assert agent.bluff_opportunities == 1
    assert agent.actual_bluffs == 0

def test_bayesian_agent_actual_bluff_detection():
    agent = BayesianAgent(name="Bayes")
    
    # Thiết lập trạng thái ban đầu của ván đấu
    observation_r1 = {
        "player_id": 0,
        "my_dice": [2, 3, 5],            # 0 viên mặt 4
        "opponent_dice_count": 3,
        "current_bid": None,
        "history": []
    }
    agent.act(observation_r1, [Bid(1, 2)])
    
    # Đối thủ cược 3 viên mặt 4 (Q = 3, E = 0 + 1 = 1 -> bluff_opportunity)
    agent.observe(Bid(3, 4), 1)
    
    # Ta gọi act() lần 2 để cập nhật cược của đối thủ
    observation_mid = {
        "player_id": 0,
        "my_dice": [2, 3, 5],
        "opponent_dice_count": 3,
        "current_bid": Bid(3, 4),
        "history": [(0, Bid(1, 2)), (1, Bid(3, 4))]
    }
    agent.act(observation_mid, [Challenge(), Bid(3, 5)])
    assert agent.bluff_opportunities == 1
    
    # Ta quyết định Challenge đối thủ
    agent.observe(Challenge(), 0)
    
    # Giả lập đối thủ bị lật tẩy nói dối (bị trừ 1 xúc xắc, từ 3 xuống 2) ở vòng mới
    observation_r2 = {
        "player_id": 0,
        "my_dice": [1, 2, 6],
        "opponent_dice_count": 2,        # Số xúc xắc đối thủ giảm từ 3 xuống 2
        "current_bid": None,
        "history": []
    }
    
    # Gọi act() cho vòng mới
    agent.act(observation_r2, [Bid(1, 2)])
    
    # Đối thủ là bidder trong lần challenge trước và bị mất xúc xắc -> actual_bluffs tăng 1
    assert agent.actual_bluffs == 1
    
    # bluff_rate = 1/1 = 1.0
    # dynamic_threshold = 0.4 + (1.0 * 0.2) = 0.6
    # Ta kiểm tra xem ngưỡng động có thay đổi chính xác không
    # Thử thách đối thủ cược tiếp theo
    # Đối thủ (chỉ còn 2 xúc xắc) cược 2 viên mặt 2 (Q=2)
    agent.observe(Bid(2, 2), 1)
    
    # Ta act
    observation_mid2 = {
        "player_id": 0,
        "my_dice": [1, 2, 6],            # Có 2 viên mặt 2 (1 là wild)
        "opponent_dice_count": 2,
        "current_bid": Bid(2, 2),
        "history": [(1, Bid(2, 2))]
    }
    
    # Ta kiểm tra hành động được chọn
    # my_count = 2, Q = 2, k = 0, n = 2. p = 1/3
    # P_truth = binomial_probability(0, 2, 1/3) = 1.0
    # Vì P_truth = 1.0 > 0.2 -> Agent sẽ nâng cược chứ không Challenge
    action = agent.act(observation_mid2, [Challenge(), Bid(2, 3), Bid(2, 6)])
    assert isinstance(action, Bid)
