import os
import tempfile
import pytest
from game.actions import Bid, Challenge
from agents.cfr_agent import CFRAgent

def test_cfr_agent_training_and_entries():
    agent = CFRAgent(name="CFR_Tester")
    
    # 1. Ban đầu bảng trống
    assert len(agent.regret_table) == 0
    assert len(agent.strategy_table) == 0
    
    # 2. Huấn luyện thử 20 lượt tự chơi
    agent.train(iterations=20)
    
    # Phải sinh ra các trạng thái thông tin (infosets) trong bảng
    assert len(agent.regret_table) > 0
    assert len(agent.strategy_table) > 0
    
    # Kiểm tra một số key có tồn tại giá trị cược hợp lệ
    first_key = list(agent.strategy_table.keys())[0]
    assert len(agent.strategy_table[first_key]) > 0
    assert any(val >= 0.0 for val in agent.strategy_table[first_key].values())

def test_cfr_agent_save_and_load():
    agent = CFRAgent(name="CFR_Saver")
    agent.train(iterations=10)
    
    # Tạo file tạm thời để lưu
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_filepath = tmp.name
        
    try:
        # Lưu weights
        agent.save_weights(tmp_filepath)
        assert os.path.exists(tmp_filepath)
        
        # Tạo agent mới tinh
        agent_loader = CFRAgent(name="CFR_Loader")
        assert len(agent_loader.strategy_table) == 0
        
        # Tải weights lên
        agent_loader.load_weights(tmp_filepath)
        
        # Kiểm tra tính khớp nhau của bảng trọng số
        assert len(agent_loader.strategy_table) == len(agent.strategy_table)
        assert set(agent_loader.strategy_table.keys()) == set(agent.strategy_table.keys())
        
        # Lấy một key ngẫu nhiên kiểm tra chi tiết
        random_key = list(agent.strategy_table.keys())[0]
        for action_str in agent.strategy_table[random_key]:
            assert abs(agent_loader.strategy_table[random_key][action_str] - agent.strategy_table[random_key][action_str]) < 1e-9
            
    finally:
        # Dọn dẹp file tạm
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)

def test_cfr_agent_fallback():
    agent = CFRAgent(name="CFR_With_Fallback")
    
    # Chưa train gì cả -> Bảng trống
    assert len(agent.strategy_table) == 0
    
    # Giả lập observation
    # Tay bài: [1, 2, 4, 4, 5] -> Mặt 4 nhiều nhất -> Bắt buộc phải chọn mặt 4 ở lượt cược đầu
    observation = {
        "player_id": 0,
        "my_dice": [1, 2, 4, 4, 5],
        "opponent_dice_count": 5,
        "current_bid": None,
        "history": []
    }
    
    legal_actions = [
        Bid(1, 2), Bid(1, 4), Bid(2, 4), Bid(1, 5)
    ]
    
    # Gọi act(). Sẽ kích hoạt fallback sang ProbabilisticAgent
    action = agent.act(observation, legal_actions)
    
    # ProbabilisticAgent sẽ chọn Bid có mặt cược 4 với quantity nhỏ nhất là Bid(1, 4)
    assert isinstance(action, Bid)
    assert action.face_value == 4
    assert action.quantity == 1
