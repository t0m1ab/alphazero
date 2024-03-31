import os
from pathlib import Path
from abc import abstractmethod
from aenum import Enum, NoAlias
from dataclasses import dataclass, asdict
from copy import deepcopy
import json
import numpy as np
import torch
from torch import nn

# base.py is the first file to be imported so it cannot rely on any other file except utils.py
import alphazero
from alphazero.utils import dotdict, remove_ext, DEFAULT_MODELS_PATH, DEFAULT_OUTPUTS_PATH


class Action():
    """
    Used for type indications. Usually a tuple of integers representing a move on the board of a game.
    """
    pass


class MoveFormat(Enum):
    """ Enum to indicate the type of move format required to indicate which move a player wants to play. """
    ROW_COL = "row_col"
    ROW = "row"
    COL = "col"

    @classmethod
    def to_dict(cls):
        return {x.value: x for x in cls}


class TreeEval(Enum):
    """ Enum to indicate the type of evaluation used for MCTS. """
    ROLLOUT = "rollout"
    NEURAL = "neural"

    @classmethod
    def to_dict(cls):
        return {x.value: x for x in cls}


class DisplayMode(Enum):
    """ Enum to indicate the mode of display to use. """
    HUMAN = "human"
    PIXEL = "pixel"


@dataclass
class Config():
    """ Base class to define configuration to train AlphaZero. """
    # GAME settings
    game: str = None
    # PLAYER settings
    simulations: int = None
    compute_time: float = None
    # TRAINING settings
    iterations: int = None
    episodes: int = None
    epochs: int = None
    batch_size: int = None
    learning_rate: float = None
    device: str = None
    # SAVE settings
    save: bool = True
    push: bool = False
    save_checkpoints: bool = True
    push_checkpoints: bool = False

    def to_dict(self) -> dotdict[str, float | int | str]:
        return dotdict(deepcopy(asdict(self)))


class Board():
    """
    Base class to encode the logic of a game and the state of its board when playing.
    All subclasses should implement the methods of this class in order to be compatible with all Players.
    If you need to implement other methods, please use __method_name naming to make them private.
    """

    CONFIG = Config

    def __init__(self, display_dir: str = None, display_mode: DisplayMode = None):
        self.display_dir = display_dir if display_dir is not None else DEFAULT_OUTPUTS_PATH
        if display_mode is None:
            self.display_mode = DisplayMode.HUMAN
        elif display_mode in DisplayMode:
            self.display_mode = DisplayMode(display_mode)
        else:
            raise ValueError(f"Unknown display mode: {display_mode}")
        self.game = None # str: name of the game
        self.grid = None # np.ndarray: the board representation (2D array filled with 0s, 1s and -1s)
        self.player = None # int: id of the player that needs to play (1 or -1)
        self.pass_move = None # Action: must remain None if the game never allows to pass
        self.max_moves = None # int: max limit to the number of moves in the game
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def __init_from_config(self, config: Config) -> None:
        """ Initialize the board from a Config object. """
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
    def human_display(self, *args, **kwargs):
        """ Display the current state of the board in HUMAN mode. """
        raise NotImplementedError

    @abstractmethod
    def pixel_display(self, *args, **kwargs):
        """ Display the current state of the board in PIXEL mode. """
        raise NotImplementedError
    
    def display(self, indexes: bool = True, filename: str = None, mode: str = None) -> None:
        """ 
        Displays the board according to the specified mode.

        ARGUMENTS:
            - indexes: if True, the indexes of the rows and columns are displayed on the board.
            - filename: the name of the file in which the image of the board will be saved.
            - mode: the display mode (should be a value of DisplayMode).
        """

        # check and associate display mode
        if mode is None:
            mode = self.display_mode
        elif mode in DisplayMode:
            mode = DisplayMode(mode)
        else:
            raise ValueError(f"Unknown display mode: {mode}")

        if mode == DisplayMode.HUMAN:
            self.human_display(indexes, filename)
        elif mode == DisplayMode.PIXEL:
            self.pixel_display(indexes, filename)
        else:
            raise NotImplementedError(f"Display mode {mode} is not implemented yet.")


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

    CONFIG = Config

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
        model_path = os.path.join(DEFAULT_MODELS_PATH, model_name) if model_path is None else model_path
        
        # save model weights
        Path(model_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_path, f"{model_name}.pt"))
        if verbose:
            print(f"{self} saved in: {model_path}'")

    @classmethod
    def from_pretrained(cls, model_name: str, models_path: str = None, verbose: bool = False) -> "PolicyValueNetwork":
        """ Loads the model from using a 'config.json' file and '<model_name>.pt' weights. """
        models_path = DEFAULT_MODELS_PATH if models_path is None else models_path
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
            json_config = json.load(f)
        model_config = cls.CONFIG(**json_config) # fetch the right Config subclass and init with the json_dict
        model = cls(config=model_config) # init the model with the config
        model.load_state_dict(torch.load(pt_path)) # load the weights

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
    def __init_from_config(self, config: Config) -> None:
        """ Initialize the network from a Config object. """
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


class TemperatureScheduler():
    """
    Temperature scheduler used to control the exploration of the MCTS.
    """
    def __init__(self, temp_max_step: int, temp_min_step: int, max_steps: int) -> None:
        self.temp_max_step = temp_max_step
        self.temp_min_step = temp_min_step
        self.max_steps = max_steps
    
    @abstractmethod
    def compute_temperature(self, step: int) -> float:
        """ Compute the temperature at the given step. """
        raise NotImplementedError
    
    def __getitem__(self, step: int) -> float:
        """ Returns the temperature at the given step. """
        return self.compute_temperature(step)
    
    def __iter__(self):
        """ Returns an iterator over the temperatures. """
        for step in range(self.max_steps + 1):
            yield self.compute_temperature(step)


def main():
    
    _ = Board()
    print("Board created successfully!")

    _ = Player()
    print("Player created successfully!")

    _ = PolicyValueNetwork()
    print("PolicyValueNetwork created successfully!")


if __name__ == "__main__":
    main()