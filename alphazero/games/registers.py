from alphazero.base import MoveFormat
from alphazero.schedulers import ConstantTemperatureScheduler, LinearTemperatureScheduler
from alphazero.games.othello import OthelloConfig, OthelloBoard, OthelloNet
from alphazero.games.tictactoe import TicTacToeConfig, TicTacToeBoard, TicTacToeNet
from alphazero.games.connect4 import Connect4Config, Connect4Board, Connect4Net
from alphazero.games.othello import main as othello_main
from alphazero.games.tictactoe import main as tictactoe_main
from alphazero.games.connect4 import main as connect4_main

# GAME REGISTERS

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

# TRAINING REGISTER

TEMP_SCHEDULERS_REGISTER = {
    "constant": ConstantTemperatureScheduler,
    "linear": LinearTemperatureScheduler,
}


def main():

    configs_keys = set(CONFIGS_REGISTER.keys())
    boards_keys = set(BOARDS_REGISTER.keys())
    networks_keys = set(NETWORKS_REGISTER.keys())
    move_formats_keys = set(MOVE_FORMATS_REGISTER.keys())

    assert configs_keys == boards_keys == networks_keys == move_formats_keys, "Game registers have different keys."

    othello_main()
    tictactoe_main()
    connect4_main()


if __name__ == "__main__":
    main()