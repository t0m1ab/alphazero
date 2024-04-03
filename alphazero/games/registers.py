from alphazero.base import MoveFormat, DataTransf
from alphazero.utils import dotdict
from alphazero.schedulers import ConstantTemperatureScheduler, LinearTemperatureScheduler
from alphazero.games.othello import OthelloConfig, OthelloBoard, OthelloNet
from alphazero.games.tictactoe import TicTacToeConfig, TicTacToeBoard, TicTacToeNet
from alphazero.games.connect4 import Connect4Config, Connect4Board, Connect4Net
from alphazero.games.othello import main as othello_main
from alphazero.games.tictactoe import main as tictactoe_main
from alphazero.games.connect4 import main as connect4_main


# GAME REGISTERS

GAMES_SET = set(["othello", "tictactoe", "connect4"])

CONFIGS_REGISTER = {
    "othello": OthelloConfig,
    "tictactoe": TicTacToeConfig,
    "connect4": Connect4Config,
}

BOARDS_REGISTER = {
    "othello": OthelloBoard,
    "tictactoe": TicTacToeBoard,
    "connect4": Connect4Board,
}

NETWORKS_REGISTER = {
    "othello": OthelloNet,
    "tictactoe": TicTacToeNet,
    "connect4": Connect4Net,
}

MOVE_FORMATS_REGISTER = {
    "othello": MoveFormat.ROW_COL,
    "tictactoe": MoveFormat.ROW_COL,
    "connect4": MoveFormat.COL,
}

DATA_AUGMENT_STRATEGIES = {
    "othello": dotdict({
        "reflection": DataTransf.REFLECT_H,
        "rotations": [DataTransf.ROTATE_90, DataTransf.ROTATE_180, DataTransf.ROTATE_270],
    }),
    "connect4": dotdict({
        "reflection": DataTransf.REFLECT_H,
        "rotations": [],
    }),
    "tictactoe": dotdict({
        "reflection": DataTransf.REFLECT_H,
        "rotations": [DataTransf.ROTATE_90, DataTransf.ROTATE_180, DataTransf.ROTATE_270],
    })
}


# TRAINING REGISTER

TEMP_SCHEDULERS = {
    "constant": ConstantTemperatureScheduler,
    "linear": LinearTemperatureScheduler,
}


def main():

    if not GAMES_SET == set(CONFIGS_REGISTER.keys()):
        raise ValueError("Config register has different keys than GAME_SET.")
    
    if not GAMES_SET == set(BOARDS_REGISTER.keys()):
        raise ValueError("Boards register has different keys than GAME_SET.")
    
    if not GAMES_SET == set(NETWORKS_REGISTER.keys()):
        raise ValueError("Networks register has different keys than GAME_SET.")
    
    if not GAMES_SET == set(MOVE_FORMATS_REGISTER.keys()):
        raise ValueError("Move formats register has different keys than GAME_SET.")
    
    if not GAMES_SET == set(DATA_AUGMENT_STRATEGIES.keys()):
        raise ValueError("Data augmentation strategies register has different keys than GAME_SET.")

    othello_main()
    tictactoe_main()
    connect4_main()


if __name__ == "__main__":
    main()