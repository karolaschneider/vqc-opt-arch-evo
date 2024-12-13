"""Board Module"""

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_WEST = 2
MOVE_EAST = 3


class Board:
    """Board class for coingame"""

    def __init__(self):
        # internally self.board.squares holds a flat representation of tic tac toe board
        # where an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # where indexes are column wise order
        # 0 3 6
        # 1 4 7
        # 2 5 8

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        self.squares = [0] * 9

    def play_turn(self, agent: int, pos: int):
        """plays a turn"""
        old_position = self.squares.index(agent + 1)

        if pos == MOVE_NORTH:
            if old_position != 0 and old_position != 3 and old_position != 6:
                if agent == 0:
                    if self.squares[old_position - 1] != 2:
                        self.squares[old_position - 1] = 1
                        self.squares[old_position] = 0
                    else:
                        return
                elif agent == 1:
                    if self.squares[old_position - 1] != 1:
                        self.squares[old_position - 1] = 2
                        self.squares[old_position] = 0
                    else:
                        return
                else:
                    return
            else:
                return

        elif pos == MOVE_SOUTH:
            if old_position != 2 and old_position != 5 and old_position != 8:
                if agent == 0:
                    if self.squares[old_position + 1] != 2:
                        self.squares[old_position + 1] = 1
                        self.squares[old_position] = 0
                    else:
                        return
                elif agent == 1:
                    if self.squares[old_position + 1] != 1:
                        self.squares[old_position + 1] = 2
                        self.squares[old_position] = 0
                    else:
                        return
                else:
                    return
            else:
                return

        elif pos == MOVE_WEST:
            if old_position - 3 >= 0:
                if agent == 0:
                    if self.squares[old_position - 3] != 2:
                        self.squares[old_position - 3] = 1
                        self.squares[old_position] = 0
                    else:
                        return
                elif agent == 1:
                    if self.squares[old_position - 3] != 1:
                        self.squares[old_position - 3] = 2
                        self.squares[old_position] = 0
                    else:
                        return
                else:
                    return
            else:
                return

        elif pos == MOVE_EAST:
            if old_position + 3 <= 8:
                if agent == 0:
                    if self.squares[old_position + 3] != 2:
                        self.squares[old_position + 3] = 1
                        self.squares[old_position] = 0
                    else:
                        return
                elif agent == 1:
                    if self.squares[old_position + 3] != 1:
                        self.squares[old_position + 3] = 2
                        self.squares[old_position] = 0
                    else:
                        return
                else:
                    return
            else:
                return
        else:
            return
