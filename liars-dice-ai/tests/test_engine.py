import pytest
from game.state import GameState
from game.actions import Bid, Challenge
from game.engine import (
    count_matching_dice,
    validate_action,
    get_legal_actions,
    apply_action
)

def test_count_matching_dice():
    # Tay bài: P0 có [1, 2, 4], P1 có [1, 4, 5, 6]
    # Lưu ý: Mặt 1 là wild card
    hands = [[1, 2, 4], [1, 4, 5, 6]]
    
    # Cược mặt 4: Cần đếm số mặt 4 (hai viên) + mặt 1 (hai viên) = 4
    assert count_matching_dice(hands, 4) == 4
    
    # Cược mặt 2: Cần đếm số mặt 2 (một viên) + mặt 1 (hai viên) = 3
    assert count_matching_dice(hands, 2) == 3
    
    # Cược mặt 1: Chỉ đếm mặt 1 (hai viên) = 2 (mặt 1 không tự làm wild cho chính nó)
    assert count_matching_dice(hands, 1) == 2
    
    # Cược mặt 3: Cần đếm số mặt 3 (không viên nào) + mặt 1 (hai viên) = 2
    assert count_matching_dice(hands, 3) == 2

def test_validate_action_first_bid():
    state = GameState(start_dice=5)
    state.current_bid = None
    
    # Bất kỳ cược thông thường nào cũng hợp lệ ở lượt đầu
    assert validate_action(state, Bid(1, 2)) is True
    assert validate_action(state, Bid(5, 6)) is True
    assert validate_action(state, Bid(10, 1)) is True  # Tổng số xúc xắc trên bàn là 10
    
    # Cược vượt quá tổng số xúc xắc (10) là không hợp lệ
    assert validate_action(state, Bid(11, 4)) is False
    
    # Cược với giá trị mặt xúc xắc không hợp lệ
    assert validate_action(state, Bid(1, 7)) is False
    assert validate_action(state, Bid(1, 0)) is False
    
    # Không thể tố cáo ở lượt đầu tiên
    assert validate_action(state, Challenge()) is False

def test_validate_action_raise_bid():
    state = GameState(start_dice=5)
    state.current_bid = Bid(3, 4)
    
    # 1. Tăng số lượng cược, giữ nguyên mặt xúc xắc
    assert validate_action(state, Bid(4, 4)) is True
    
    # 2. Tăng số lượng cược, giảm mặt xúc xắc
    assert validate_action(state, Bid(4, 2)) is True
    
    # 3. Tăng số lượng cược, tăng mặt xúc xắc
    assert validate_action(state, Bid(4, 6)) is True
    
    # 4. Giữ nguyên số lượng, tăng mặt xúc xắc
    assert validate_action(state, Bid(3, 5)) is True
    assert validate_action(state, Bid(3, 6)) is True
    
    # 5. Giữ nguyên số lượng, giảm mặt xúc xắc (không hợp lệ)
    assert validate_action(state, Bid(3, 3)) is False
    
    # 6. Giữ nguyên cả số lượng và mặt xúc xắc (không hợp lệ)
    assert validate_action(state, Bid(3, 4)) is False
    
    # 7. Giảm số lượng cược (không hợp lệ)
    assert validate_action(state, Bid(2, 6)) is False
    
    # Hành động tố cáo là hợp lệ khi đã có một lượt cược trước đó
    assert validate_action(state, Challenge()) is True

def test_get_legal_actions():
    state = GameState(start_dice=2) # Tổng cộng có 4 xúc xắc
    state.current_bid = Bid(3, 5)
    
    actions = get_legal_actions(state)
    
    # Danh sách phải chứa hành động Tố cáo
    assert Challenge() in actions
    
    # Cược cùng số lượng, mặt lớn hơn
    assert Bid(3, 6) in actions
    assert Bid(3, 4) not in actions
    
    # Cược số lượng lớn hơn, mặt bất kỳ
    assert Bid(4, 1) in actions
    assert Bid(4, 6) in actions
    
    # Cược số lượng vượt quá tổng số xúc xắc
    assert Bid(5, 1) not in actions

def test_apply_bid():
    state = GameState(start_dice=5)
    state.current_player = 0
    state.current_bid = None
    
    bid = Bid(2, 3)
    outcome = apply_action(state, bid)
    
    assert state.current_bid == bid
    assert state.current_player == 1
    assert len(state.history) == 1
    assert state.history[-1] == (0, bid)
    assert outcome["type"] == "bid"

def test_apply_challenge_bidder_bluffs():
    state = GameState(start_dice=5)
    # Gán tay bài thủ công để tránh yếu tố ngẫu nhiên khi kiểm thử
    state.hands = [[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]] # Tổng số mặt 1 là 0
    state.current_player = 1
    state.current_bid = Bid(3, 1) # Người cược (Player 0) cược có 3 viên mặt 1
    
    # Player 1 gọi Tố cáo
    outcome = apply_action(state, Challenge())
    
    # Tổng số mặt 1 thực tế là 0, ít hơn 3 -> Người cược (Player 0) nói dối!
    # Player 0 thua và mất 1 xúc xắc.
    assert outcome["challenge_success"] is True
    assert outcome["loser"] == 0
    assert state.dice_counts[0] == 4
    # Reset vòng mới, người thua cuộc (Player 0) bắt đầu lượt đi đầu tiên
    assert state.current_bid is None
    assert state.current_player == 0

def test_apply_challenge_bidder_truth():
    state = GameState(start_dice=5)
    state.hands = [[1, 1, 4, 5, 6], [1, 2, 4, 5, 6]] # Tổng số mặt 1 thực tế là 3
    state.current_player = 1
    state.current_bid = Bid(3, 1) # Người cược (Player 0) cược có 3 viên mặt 1
    
    # Player 1 gọi Tố cáo
    outcome = apply_action(state, Challenge())
    
    # Tổng số mặt 1 thực tế là 3, bằng với số cược (>= 3) -> Người cược (Player 0) nói thật!
    # Người tố cáo (Player 1) thua và mất 1 xúc xắc.
    assert outcome["challenge_success"] is False
    assert outcome["loser"] == 1
    assert state.dice_counts[1] == 4
    # Reset vòng mới, người thua cuộc (Player 1) bắt đầu lượt đi đầu tiên
    assert state.current_bid is None
    assert state.current_player == 1
