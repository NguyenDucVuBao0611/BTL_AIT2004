from typing import List, Tuple, Dict, Any, Optional
from game.state import GameState
from game.actions import Action, Bid, Challenge

def count_matching_dice(hands: List[List[int]], face_value: int) -> int:
    """Đếm tổng số lượng xúc xắc có mặt `face_value` hoặc mặt 1 (wild).
    Nếu `face_value` là 1, chỉ đếm số xúc xắc mặt 1.
    """
    total = 0
    for hand in hands:
        for die in hand:
            if die == face_value:
                total += 1
            elif face_value != 1 and die == 1:
                # 1 là xúc xắc wild card, được tính như face_value trừ khi cược chính mặt 1
                total += 1
    return total

def validate_action(state: GameState, action: Action) -> bool:
    """Kiểm tra xem một hành động có hợp lệ trong trạng thái game hiện tại hay không."""
    total_dice = sum(state.dice_counts)
    
    if isinstance(action, Challenge):
        # Chỉ có thể tố cáo nếu đã có người cược trước đó trong vòng này
        return state.current_bid is not None

    elif isinstance(action, Bid):
        # Giá trị mặt xúc xắc phải nằm trong khoảng từ 1 đến 6
        if not (1 <= action.face_value <= 6):
            return False
        
        # Số lượng cược phải nằm trong khoảng từ 1 đến tổng số xúc xắc hiện có trên bàn
        if not (1 <= action.quantity <= total_dice):
            return False

        # Nếu đã có lượt cược trước đó, so sánh với lượt cược trước
        if state.current_bid is not None:
            prev = state.current_bid
            # Quy tắc nâng cược: tăng số lượng HOẶC (số lượng bằng và tăng giá trị mặt)
            if action.quantity > prev.quantity:
                return True
            elif action.quantity == prev.quantity:
                return action.face_value > prev.face_value
            else:
                return False
        else:
            # Lượt cược đầu tiên của vòng có thể là bất kỳ giá trị hợp lệ nào
            return True

    return False

def get_legal_actions(state: GameState) -> List[Action]:
    """Tạo ra tất cả các hành động hợp lệ cho người chơi hiện tại trong trạng thái hiện tại."""
    legal_actions: List[Action] = []
    total_dice = sum(state.dice_counts)

    # 1. Hành động tố cáo (nếu có cược trước đó)
    if state.current_bid is not None:
        legal_actions.append(Challenge())

    # 2. Các hành động cược hợp lệ
    if state.current_bid is None:
        # Đầu vòng chơi: có thể cược số lượng từ 1 đến total_dice, mặt từ 1 đến 6
        for q in range(1, total_dice + 1):
            for f in range(1, 7):
                legal_actions.append(Bid(q, f))
    else:
        prev = state.current_bid
        # Cược cùng số lượng, mặt lớn hơn
        for f in range(prev.face_value + 1, 7):
            legal_actions.append(Bid(prev.quantity, f))
        
        # Cược số lượng lớn hơn, mặt bất kỳ (từ 1 đến 6)
        for q in range(prev.quantity + 1, total_dice + 1):
            for f in range(1, 7):
                legal_actions.append(Bid(q, f))

    return legal_actions

def apply_action(state: GameState, action: Action) -> Dict[str, Any]:
    """Áp dụng hành động lên trạng thái game. Thay đổi trực tiếp trạng thái (in-place).
    Trả về một dictionary mô tả chi tiết kết quả hành động (đặc biệt hữu ích khi Challenge).
    """
    if not validate_action(state, action):
        raise ValueError(f"Hành động không hợp lệ: {action} trong trạng thái hiện tại.")

    outcome = {
        "action": action,
        "player_id": state.current_player,
        "type": "bid" if isinstance(action, Bid) else "challenge"
    }

    if isinstance(action, Bid):
        state.current_bid = action
        state.history.append((state.current_player, action))
        # Chuyển lượt sang người chơi tiếp theo
        state.current_player = 1 - state.current_player
    
    elif isinstance(action, Challenge):
        state.history.append((state.current_player, action))
        
        # Xử lý tố cáo
        challenger = state.current_player
        bidder = 1 - challenger
        bid = state.current_bid
        
        # Đếm số xúc xắc thực tế khớp với cược
        actual_count = count_matching_dice(state.hands, bid.face_value)
        
        # Xác định người thua
        if actual_count >= bid.quantity:
            # Người cược nói thật/thắng. Challenger (người tố) thua và mất 1 xúc xắc.
            loser = challenger
            winner = bidder
            success = False # Tố cáo thất bại
        else:
            # Người cược nói dối (bluff). Bidder (người cược) thua và mất 1 xúc xắc.
            loser = bidder
            winner = challenger
            success = True # Tố cáo thành công
            
        state.dice_counts[loser] -= 1
        
        outcome.update({
            "challenger": challenger,
            "bidder": bidder,
            "bid": bid,
            "actual_count": actual_count,
            "winner": winner,
            "loser": loser,
            "challenge_success": success,
            "hands": [list(hand) for hand in state.hands] # Lưu lại tay xúc xắc để đối chiếu
        })
        
        # Nếu trò chơi chưa kết thúc, reset vòng chơi mới
        # Người thua cuộc ở vòng trước sẽ đi trước ở vòng mới
        if not state.is_game_over():
            state.reset_round(starting_player=loser)
            
    return outcome
