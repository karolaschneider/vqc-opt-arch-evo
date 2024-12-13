"""Coin class for coin game, can be initialized or reset"""
import random


class Coin:
    """Coin class for coingame"""

    def __init__(self, nr_agents: int) -> None:
        self.agent_ids = list(range(nr_agents))
        self.agent_id = None
        self.position = None

    def reset(self, position: int):
        """Resets the Coin"""
        self.position = position
        self.agent_id = random.choice(self.agent_ids)
