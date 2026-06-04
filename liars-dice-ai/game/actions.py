class Action:
    """Lớp cơ sở cho tất cả các hành động trong trò chơi Liar's Dice."""
    pass

class Bid(Action):
    """Đại diện cho hành động cược (Bid), nơi người chơi tuyên bố có ít nhất
    `quantity` (số lượng) viên xúc xắc có mặt `face_value` (giá trị mặt) giữa tất cả người chơi.
    """
    def __init__(self, quantity: int, face_value: int):
        self.quantity = quantity
        self.face_value = face_value

    def __eq__(self, other):
        if not isinstance(other, Bid):
            return False
        return self.quantity == other.quantity and self.face_value == other.face_value

    def __hash__(self):
        return hash(("Bid", self.quantity, self.face_value))

    def __repr__(self):
        return f"Bid({self.quantity} x {self.face_value})"

    def __str__(self):
        return f"Cược {self.quantity} viên mặt {self.face_value}"

class Challenge(Action):
    """Đại diện cho hành động tố cáo (Challenge - 'Liar!'), nơi người chơi
    tố cáo người cược trước đó đã nói dối (bluff).
    """
    def __eq__(self, other):
        return isinstance(other, Challenge)

    def __hash__(self):
        return hash("Challenge")

    def __repr__(self):
        return "Challenge()"

    def __str__(self):
        return "Tố cáo (Challenge)"
