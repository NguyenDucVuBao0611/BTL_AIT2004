import random
from typing import List
from agents.base_agent import Agent
from game.actions import Action

class RandomAgent(Agent):
    """Một Agent chơi ngẫu nhiên, tự động chọn một hành động hợp lệ từ danh sách cho trước."""
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)

    def act(self, observation: dict, legal_actions: List[Action]) -> Action:
        return random.choice(legal_actions)
