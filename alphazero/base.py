import os
from pathlib import Path
from abc import abstractmethod
from aenum import Enum, NoAlias
from copy import deepcopy
import json
import numpy as np
import torch
from torch import nn

import alphazero
from alphazero.utils import dotdict, remove_ext, DEFAULT_MODEL_PATH


class Action():
    """
    Used for type indications. Usually a tuple of integers representing a move on the board of a game.
    """
    pass


class TreeEval(Enum):
    """ Enum to indicate the type of evaluation used for MCTS. """
    ROLLOUT = "rollout"
    NEURAL = "neural"

    @classmethod
    def to_dict(cls):
        return {x.value: x for x in cls}


class Config(Enum):
    """ Base class to define configuration to train AlphaZero. """
    _settings_ = NoAlias # avoid grouping items with same values

    @classmethod
    def to_dict(cls) -> dotdict[str, float | int | str]:
        """ To access config parameters easily, for example: EPOCHS/Epochs/epochs -> config.epochs """
        return dotdict({x.name.lower(): x.value for x in cls})


class Board():
    """
    Base class to encode the logic of a game and the state of its board when playing.
    All subclasses should implement the methods of this class in order to be compatible with all Players.
    If you need to implement other methods, please use __method_name naming to make them private.
    """

    DEFAULT_DISPLAY_DIR = os.path.join(alphazero.__path__[0], "outputs/")

    def __init__(self, display_dir: str = None):
        self.display_dir = display_dir if display_dir is not None else Board.DEFAULT_DISPLAY_DIR
        self.game_name = None # str: name of the game
        self.grid = None # np.ndarray: the board representation (2D array filled with 0s, 1s and -1s)
        self.player = None # int: id of the player that needs to play (1 or -1)
        self.pass_move = None # Action: must remain None if the game never allows to pass
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def __init_from_config(self, config_dict: dotdict) -> None:
        """ Initialize the board from a config given in a dotdict. """
        raise NotImplementedError

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
        """ Returns the number of cells in self.grid. """
        return NotImplementedError

    @abstractmethod
    def get_action_size(self) -> int:
        """ Returns the number of possible moves in the game. """
        raise NotImplementedError
    
    @abstractmethod
    def get_score(self) -> int | float:
        """ 
        Returns the current score of the board from the viewpoint of self.player.
        Mainly used by the GreedyPlayer to choose an action.
        """
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


class PolicyValueNetwork(nn.Module):
    """
    Base class to encode the logic of a policy-value network used by AlphaZeroPlayer players.
    """
    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        return self.__class__.__name__
    
    def clone(self) -> "PolicyValueNetwork":
        """ Returns a deep copy of the network. """
        return deepcopy(self)
    
    def get_parameters_count(self) -> int:
        """ Returns the number of parameters of the network. """
        return sum(p.numel() for p in self.parameters())
    
    def save_model(self, model_name: str, model_path: str = None, verbose: bool = False) -> None:
        """ Saves the weights of the model to a '<model_name>.pt' file. """
        model_name = remove_ext(model_name)
        # folder containing the model and the config
        model_path = os.path.join(DEFAULT_MODEL_PATH, model_name) if model_path is None else model_path
        
        # save model weights
        Path(model_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_path, f"{model_name}.pt"))
        if verbose:
            print(f"{self} saved in: {model_path}'")

    @classmethod
    def from_pretrained(cls, model_name: str, models_path: str = None, verbose: bool = False) -> "PolicyValueNetwork":
        """ Loads the model from using a 'config.json' file and '<model_name>.pt' weights. """
        models_path = DEFAULT_MODEL_PATH if models_path is None else models_path
        model_name = remove_ext(model_name)
        model_dir = os.path.join(models_path, model_name) # folder containing the model and the config

        # check that config and model files exist
        config_path = os.path.join(model_dir, "config.json")
        pt_path = os.path.join(model_dir, f"{model_name}.pt")
        if not os.path.isfile(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        if not os.path.isfile(pt_path):
            raise ValueError(f"Model file not found: {pt_path}")

        # load and create model
        with open(config_path, "r") as f:
            model = cls(config_dict=json.load(f))
        model.load_state_dict(torch.load(pt_path))

        if verbose:
            print(f"{model} loaded from: {model_dir}'")
        
        return model

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
    def __init_from_config(self, config_dict: dotdict) -> None:
        """ Initialize the network from a config given in a dotdict. """
        raise NotImplementedError

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

    @abstractmethod
    def to_neural_array(self, move_probs: dict[Action: float]) -> np.ndarray:
        """ Returns the probabilitites of move_probs in the format given as output by the network. """
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