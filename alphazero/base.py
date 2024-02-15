import os

import alphazero


class Board():
    """
    Base class to encode the logic of a game and the state of its board when playing.
    All subclasses should implement the methods of this class in order to be compatible with all Players.
    If you need to implement other methods, please use __method_name naming to make them private.
    """

    DEFAULT_DISPLAY_DIR = os.path.join(alphazero.__path__[0], "outputs/")

    def __init__(self, display_dir: str = None):
        self.display_dir = display_dir if display_dir is not None else Board.DEFAULT_DISPLAY_DIR
        self.n = None
        self.cells = None
        self.player = None
        self.pass_move = None # must remain None if the game doesn't allow to pass
    
    def reset(self) -> None:
        """ Resets the board to the initial state. """
        raise NotImplementedError
    
    def clone(self) -> "Board":
        """ Returns a deep copy of the board. """
        raise NotImplementedError
    
    def get_board_size(self) -> tuple[int, int]:
        """ Returns the size of the board. """
        raise NotImplementedError

    def get_action_size(self) -> int:
        """ Returns the number of possible moves in the game. """
        raise NotImplementedError
    
    def get_score(self) -> int | float:
        """ Returns the current score of the board from the viewpoint of self.player. """
        raise NotImplementedError
    
    def is_legal_move(self, move, player) -> bool:
        """ Returns True if the move is legal, False otherwise (considering that it is a move for player <player>). """
        raise NotImplementedError
    
    def get_moves(self, player) -> list[tuple[int, int]]:
        """ Returns the list of legal moves for the player <player> in the current state of the board. """
        raise NotImplementedError
    
    def get_random_move(self, player) -> tuple[int, int]:
        """ Returns a random legal move for the player <player> in the current state of the board. """
        raise NotImplementedError
    
    def play_move(self, move) -> None:
        """ Plays the move on the board (self.player is playing this move). """
        raise NotImplementedError
    
    def is_game_over(self) -> bool:
        """ Returns True if the game is over, False otherwise. """
        raise NotImplementedError
    
    def get_winner(self) -> int:
        """ Returns the id of the winner of the game (1 or -1) or 0 if the game is a draw."""
        raise NotImplementedError
    
    def display(self, *args, **kwargs):
        """ Display the current state of the board. """
        raise NotImplementedError


class Player():
    """
    Base class to encode the logic of a player in a game.
    All subclasses should implement the methods of this class in order to be used in arenas.
    If you need to implement other methods, please use __method_name naming to make them private.
    """

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def reset(self, verbose: bool = False) -> None:
        """ Resets the internal state of the player. """
        pass

    def apply_move(self, move, player: int = None) -> None:
        """ Updates the internal state of the player after a move is played. """
        pass

    def get_move(self, board: Board) -> tuple[int, int] | None:
        """ Returns the best move for the player given the current board state. """
        raise NotImplementedError


def main():
    
    _ = Board()
    print("Board created successfully!")

    _ = Player()
    print("Player created successfully!")


if __name__ == "__main__":
    main()