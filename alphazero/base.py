import os
from abc import abstractmethod
import numpy as np
import torch
from copy import deepcopy

import alphazero


class Action():
    """
    Used for type indications. Usually a tuple of integers representing a move on the board of a game.
    """
    pass


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
        self.game_name = None
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def reset(self) -> None:
        """ Resets the board to the initial state. """
        raise NotImplementedError
    
    @abstractmethod
    def clone(self) -> "Board":
        """ Returns a deep copy of the board. """
        raise NotImplementedError
    
    @abstractmethod
    def get_board_shape(self) -> tuple[int, int]:
        """ Returns the size of the board. """
        raise NotImplementedError
    
    @abstractmethod
    def get_n_cells(self) -> int:
        """ Returns the number of cells in the board. """
        return NotImplementedError

    @abstractmethod
    def get_action_size(self) -> int:
        """ Returns the number of possible moves in the game. """
        raise NotImplementedError
    
    @abstractmethod
    def get_score(self) -> int | float:
        """ Returns the current score of the board from the viewpoint of self.player. """
        raise NotImplementedError
    
    @abstractmethod
    def is_legal_move(self, move, player) -> bool:
        """ Returns True if the move is legal, False otherwise (considering that it is a move for player <player>). """
        raise NotImplementedError
    
    @abstractmethod
    def get_moves(self, player) -> list[Action]:
        """ Returns the list of legal moves for the player <player> in the current state of the board. """
        raise NotImplementedError
    
    @abstractmethod
    def get_random_move(self, player) -> Action:
        """ Returns a random legal move for the player <player> in the current state of the board. """
        raise NotImplementedError
    
    @abstractmethod
    def play_move(self, move) -> None:
        """ Plays the move on the board (self.player is playing this move). """
        raise NotImplementedError
    
    @abstractmethod
    def is_game_over(self) -> bool:
        """ Returns True if the game is over, False otherwise. """
        raise NotImplementedError
    
    @abstractmethod
    def get_winner(self) -> int:
        """ Returns the id of the winner of the game (1 or -1) or 0 if the game is a draw."""
        raise NotImplementedError
    
    @abstractmethod
    def display(self, *args, **kwargs):
        """ Display the current state of the board. """
        raise NotImplementedError


class Player():
    """
    Base class to encode the logic of a player in a game.
    All subclasses should implement the methods of this class in order to be used in arenas.
    If you need to implement other methods, please use __method_name naming to make them private.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def __str__(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def clone(self) -> "Player":
        """ Returns a deep copy of the player. """
        raise NotImplementedError

    def reset(self) -> None:
        """ Resets the internal state of the player. """
        pass

    def apply_move(self, move, player: int = None) -> None:
        """ Updates the internal state of the player after a move is played. """
        pass

    @abstractmethod
    def get_move(self, board: Board, temp: float = None) -> Action | None:
        """ Returns the best move for the player given the current board state. """
        raise NotImplementedError

    def get_stats_after_move(self) -> dict[str, int|float]:
        """ Returns the statistics of the player's last move. """
        return {}


class PolicyValueNetwork():
    """
    Base class to encode the logic of a policy-value network used in by AlphaZero type of players
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return self.__class__.__name__
    
    def clone(self) -> "PolicyValueNetwork":
        """ Returns a deep copy of the network. """
        return deepcopy(self)

    @staticmethod
    def get_torch_device(device: str) -> torch.device:
        """ Returns torch.device if the requested device is available. If device is None, returns "cpu" by default. """
        
        if device is None:
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS not available...")
        elif device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA not available...")

        return torch.device(device)

    @abstractmethod
    def forward(self, input: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """ Forward through the network and outputs (logits of probabilitites, value). """
        raise NotImplementedError

    @abstractmethod
    def predict(self, input: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """ Returns the policy and value of the input state. """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        """ 
        Evaluation of the state of the cloned board from the viewpoint of the player that needs to play. 
        A PolicyValueNetwork always evaluates the board from the viewpoint of player with id 1.
        Therefore, the board should be switched if necessary.
        """
        raise NotImplementedError

    @abstractmethod
    def get_normalized_probs(self, probs: np.ndarray, legal_moves: list[Action]) -> dict[Action, float]:
        """ Returns the normalized probabilities over the legal moves. """
        raise NotImplementedError


def main():
    
    _ = Board()
    print("Board created successfully!")

    _ = Player()
    print("Player created successfully!")

    _ = PolicyValueNetwork()
    print("PolicyValueNetwork created successfully!")


if __name__ == "__main__":
    main()