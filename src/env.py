"""Class for CoinGame Environment"""

import random
from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from board import Board
from coin import Coin

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_WEST = 2
MOVE_EAST = 3


def create_env(width: int, height: int):
    """Defines the Environment"""
    env = CoinGame(width, height)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class CoinGame(AECEnv): # type: ignore
    """Creates the Environment"""

    metadata = {
        "render_modes": ["human"],
        "name": "coingame",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    # x for x Axis, y for y Axis and z for which Agent/Coin is on the given position
    # z = 0 is nothing on this position
    # z = 1 is player red
    # z = 2 is player blue
    # z = 3 is coin red
    # z = 4 is coin blue

    def __init__(self, width: int, height: int):
        super().__init__()
        # config for the game with the most important information we need to know
        self.width = width
        self.height = height
        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]
        self.action_spaces_dict = {i: spaces.Discrete(4) for i in self.agents}
        self.observation_spaces_dict = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(4, self.width, self.height), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
                }
            )
            for i in self.agents
        }
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}

        # gives the number of max steps and in this case the end condition: standard is 150
        self.steptime = 100

        # create the board with following information
        # where an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # where indexes are column wise order
        # 0 3 6
        # 1 4 7
        # 2 5 8
        self.board = Board()

        # Class coin can be used in the length of self.agents different colours
        self.coin = Coin(len(self.agents))

        # define rewards, done for checking when an agent is done and infos for the Game
        self.rewards = {i: 0 for i in self.agents}
        #self.dones = {i: False for i in self.agents}
        self.infos = {i: {"legal_moves": list(range(0, 4))} for i in self.agents}
        self.state_dict = {i: None for i in self.agents}

        # agent selection
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces_dict[agent]

    def action_space(self, agent: str):
        return self.action_spaces_dict[agent]

    # function observe gives you the observation and action_mask checking the legal_moves
    def observe(self, agent: str):
        board_vals = np.array(self.board.squares, dtype=int).reshape(3, 3)
        current_player = self.possible_agents.index(agent)
        opponent_player = (current_player + 1) % 2
        coin_player1 = 5
        coin_player2 = 5
        if self.coin.agent_id == 0:
            coin_player1 = 3
        elif self.coin.agent_id == 1:
            coin_player2 = 4

        current_p_board = np.equal(board_vals, current_player + 1)
        opponent_p_player = np.equal(board_vals, opponent_player + 1)
        coin_red = np.equal(board_vals, coin_player1)
        coin_blue = np.equal(board_vals, coin_player2)

        observation = np.stack(
            [current_p_board, opponent_p_player, coin_red, coin_blue], axis=2
        ).astype(np.float32)
        legal_moves = self.legal_moves() if agent == self.agent_selection else []

        action_mask = np.zeros(4, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def legal_moves(self):
        """checks if a move is legal or illegal"""
        # tracks the position of both agents in the board
        position_agent1 = self.board.squares.index(1)
        position_agent2 = self.board.squares.index(2)

        # creates empty list for legal moves
        legal_moves = []

        # gives the integer on the agent 0 == player_1 and 1 == player_2
        agent = self.agents.index(self.agent_selection)

        # for each of the 4 possible actions check if they are an legal_move for
        # the selected agent, if yes append on the legal_moves list
        for action in range(4):

            if agent == 0:

                if action == MOVE_NORTH:
                    if position_agent1 != 0 and position_agent1 != 3 and position_agent1 != 6:
                        if position_agent1 - 1 != position_agent2:
                            legal_moves.append(0)

                elif action == MOVE_SOUTH:
                    if position_agent1 != 2 and position_agent1 != 5 and position_agent1 != 8:
                        if position_agent1 + 1 != position_agent2:
                            legal_moves.append(1)

                elif action == MOVE_WEST:
                    if position_agent1 - 3 >= 0:
                        if position_agent1 - 3 != position_agent2:
                            legal_moves.append(2)

                elif action == MOVE_EAST:
                    if position_agent1 + 3 <= 8:
                        if position_agent1 + 3 != position_agent2:
                            legal_moves.append(3)

            elif agent == 1:

                if action == MOVE_NORTH:
                    if position_agent2 != 0 and position_agent2 != 3 and position_agent2 != 6:
                        if position_agent2 - 1 != position_agent1:
                            legal_moves.append(0)

                elif action == MOVE_SOUTH:
                    if position_agent2 != 2 and position_agent2 != 5 and position_agent2 != 8:
                        if position_agent2 + 1 != position_agent1:
                            legal_moves.append(1)

                elif action == MOVE_WEST:
                    if position_agent2 - 3 >= 0:
                        if position_agent2 - 3 != position_agent1:
                            legal_moves.append(2)

                elif action == MOVE_EAST:
                    if position_agent2 + 3 <= 8:
                        if position_agent2 + 3 != position_agent1:
                            legal_moves.append(3)
            action += 1

        return legal_moves

    
    def step(self, action: Any, render: bool = False):
        self.steptime -= 1

        if render:
            print(f"Coin position: {self.coin.position}")
            print(f"Board state: {self.board.squares}")

        act: int = action
        self.rewards = {i: 0 for i in self.agents}
        a = np.array(self.legal_moves())
        b = np.array(act)
        mask = np.isin(b, a).all()
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
            
        assert mask, "played illegal move"

        newpos_coin = random.randint(0, 8)
        self.board.play_turn(self.agents.index(self.agent_selection), act)
        position_a1 = self.board.squares.index(1)
        position_a2 = self.board.squares.index(2)
        position_c = self.coin.position
        next_agent = self._agent_selector.next()

        if position_c == position_a1:
            if self.coin.agent_id == 0:
                self.rewards[self.agents[0]] += 1
            elif self.coin.agent_id == 1:
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 2
            if position_c is None:
                raise Exception("Fehler: Position des Coins ist noch nicht gesetzt")
            self.coin_reset(newpos_coin)
        
        elif self.coin.position == position_a2:
            if self.coin.agent_id == 1:
                self.rewards[self.agents[1]] += 1
            elif self.coin.agent_id == 0:
                self.rewards[self.agents[1]] += 1
                self.rewards[self.agents[0]] -= 2
            if position_c is None:
                raise Exception("Fehler: Position des Coins ist noch nicht gesetzt")
            self.coin_reset(newpos_coin)

        if self.steptime == 0:
            self.terminations = {i: True for i in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self.state_dict[self.agent_selection] = action
        self.agent_selection = next_agent

        self._accumulate_rewards()

        if render:
            self.render()
            print(f"Coin position: {self.coin.position}")
            print(f"Board state: {self.board.squares}")
            print("--------------------")

    def agent_reset(self, agent_i: int, pos_a1: int, pos_a2: int):
        """resets the agents position"""
        if agent_i == 0:
            self.board.squares[pos_a1] = 1

        if agent_i == 1:
            if pos_a1 != pos_a2:
                self.board.squares[pos_a2] = 2

            elif pos_a1 == pos_a2:
                if pos_a2 - 1 >= 0:
                    self.board.squares[pos_a2 - 1] = 2

                else:
                    self.board.squares[pos_a2 + 1] = 2

        return

    def coin_check_pos(self):
        """checks if agent_id of the coin and if id is given and there is a coin set the correct coin on board"""
        if self.coin.agent_id is None:
            raise Exception("Fehler: Der Coin ist noch keinem Agenten zugewiesen")
        elif self.coin.position is None:
            raise Exception("Fehler: Der Coin ist noch keinem Agenten zugewiesen")
        elif self.coin.agent_id == 0:
            self.board.squares[self.coin.position] = 3
        elif self.coin.agent_id == 1:
            self.board.squares[self.coin.position] = 4

    def coin_reset(self, pos_coin: int):
        """Resets the Coin and checks that coin is not placed on an agent or out of the board"""
        # reset coin
        # get the position of agent_1 and agent_2 and create a random position for the coin
        position_a1 = self.board.squares.index(1)
        position_a2 = self.board.squares.index(2)
        pos_coin = random.randint(0, 8)

        # if the coin position is neither the position of agent_1 nor
        # the position of agent_2 place the coin on the random int
        if pos_coin != position_a1 and pos_coin != position_a2:
            self.coin.reset(pos_coin)
            self.coin_check_pos()

        # of the coin position is the same as the position of agent_1
        # try to give a new position by add or sub 1 on the position
        # there are some edgecases which need extra handeling,
        # for example +1 or -1 leeds to agent_2 position or ist out of Board
        elif pos_coin == position_a1 and pos_coin != position_a2:
            if pos_coin - 1 != position_a2 and pos_coin - 1 >= 0:
                self.coin.reset(pos_coin - 1)
                self.coin_check_pos()
            elif pos_coin - 1 == position_a2:
                if pos_coin + 1 <= 8:
                    self.coin.reset(pos_coin + 1)
                    self.coin_check_pos()
                else:
                    self.coin.reset(pos_coin - 2)
                    self.coin_check_pos()
            elif pos_coin - 1 < 0:
                if pos_coin + 1 != position_a2:
                    self.coin.reset(pos_coin + 1)
                    self.coin_check_pos()
                else:
                    self.coin.reset(pos_coin + 2)
                    self.coin_check_pos()

        # Same structure as for agent_1 but now agent_2 has same position as coin,
        # so coin position needs to be adopted
        elif pos_coin != position_a1 and pos_coin == position_a2:
            if pos_coin - 1 != position_a1 and pos_coin - 1 >= 0:
                self.coin.reset(pos_coin - 1)
                self.coin_check_pos()
            elif pos_coin - 1 == position_a1:
                if pos_coin + 1 <= 8:
                    self.coin.reset(pos_coin + 1)
                    self.coin_check_pos()
                else:
                    self.coin.reset(pos_coin - 2)
                    self.coin_check_pos()
            elif pos_coin - 1 < 0:
                if pos_coin + 1 != position_a1:
                    self.coin.reset(pos_coin + 1)
                    self.coin_check_pos()
                else:
                    self.coin.reset(pos_coin + 2)
                    self.coin_check_pos()

    def reset(self, seed: Optional[int] = None, options: Any = None):
        """This Methode resets the environment means all the edited varibales,
        the agents and the coin"""

        # reset environment variables
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        #self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.state_dict = {i: None for i in self.agents}

        # resets the board and steptimer
        self.board = Board()
        self.steptime = 50
        # resets agents position
        random.seed(seed)
        pos_a1 = random.randint(0, 8)
        pos_a2 = random.randint(0, 8)
        pos_coin = random.randint(0, 8)
        for i, _ in enumerate(self.agents):
            self.agent_reset(i, pos_a1, pos_a2)

        # reset the coin
        self.coin_reset(pos_coin)

        # selects the first agent
        # alt: self._agent_selector.reinit(self.agents)
        self._agent_selector = agent_selector(self.agents)
        # alt self._agent_selector.reset()
        # alt self.agent_selection = self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()

    def render(self, mode: str = "human"):
        """Renders the Board of the Game to visualize what happens"""
        if mode != "human":
            return

        def get_symbol(x: int):
            if x == 0:
                return "-"
            elif x == 1:
                return "1"
            elif x == 2:
                return "2"
            elif x == 3:
                return "R"
            elif x == 4:
                return "B"

        board = list(map(get_symbol, self.board.squares))

        print(" " * 5 + "|" + " " * 5 + "|" + " " * 5)
        print(f"  {board[0]}  " + "|" + f"  {board[3]}  " + "|" + f"  {board[6]}  ")
        print("_" * 5 + "|" + "_" * 5 + "|" + "_" * 5)

        print(" " * 5 + "|" + " " * 5 + "|" + " " * 5)
        print(f"  {board[1]}  " + "|" + f"  {board[4]}  " + "|" + f"  {board[7]}  ")
        print("_" * 5 + "|" + "_" * 5 + "|" + "_" * 5)

        print(" " * 5 + "|" + " " * 5 + "|" + " " * 5)
        print(f"  {board[2]}  " + "|" + f"  {board[5]}  " + "|" + f"  {board[8]}  ")
        print(" " * 5 + "|" + " " * 5 + "|" + " " * 5)

    def seed(self, seed: Optional[int] = None) -> None:
        pass

    def state(self) -> np.ndarray:  # type: ignore
        pass

    def close(self):
        pass
