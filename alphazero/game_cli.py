import argparse

from alphazero.base import Board, Player
from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer
from alphazero.arena import Arena
from alphazero.games.othello import OthelloBoard
from alphazero.games.tictactoe import TicTacToeBoard
from alphazero.games.connect4 import Connect4Board
from alphazero.games.registers import BOARDS_REGISTER


def othello():
    """ Play Othello with CLI. """
    player1 = HumanPlayer(game="othello")
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = OthelloBoard(6)
    arena = Arena(player1, player2, board)
    arena.play_game(
        player2_starts=False, 
        verbose=True, 
        display=True, 
        save_frames=False,
        show_indexes=True,
    )


def tictactoe():
    """ Play TicTacToe with CLI. """
    player1 = HumanPlayer(game="tictactoe")
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = TicTacToeBoard()
    arena = Arena(player1, player2, board)
    arena.play_game(
        player2_starts=False, 
        verbose=True, 
        display=True, 
        save_frames=False,
        show_indexes=True,
    )


def connect4():
    """ Play Connect4 with CLI. """
    player1 = HumanPlayer(game="connect4")
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = Connect4Board(width=7, height=6)
    arena = Arena(player1, player2, board)
    arena.play_game(
        player2_starts=False, 
        verbose=True, 
        display=True, 
        save_frames=False,
        show_indexes=True,
    )


def main():

    DEFAULT_GAME = "othello"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--game",
        dest="game",
        type=str,
        default=DEFAULT_GAME,
        help="name of the game to play (ex: 'othello', 'tictactoe', 'connect4'...).",
    )
    parser.add_argument(
        "-l",
        "--list",
        dest="list_commands",
        action="store_true",
        default=False,
        help="if set then list existing commands to play any available game.",
    )
    args = parser.parse_args()

    if args.list_commands:
        print("Available commands:")
        for game in BOARDS_REGISTER.keys():
            print(f"* python game_cli.py --game {game}")
        return

    if args.game == "othello":
        othello()
    elif args.game == "tictactoe":
        tictactoe()
    elif args.game == "connect4":
        connect4()
    else:
        raise ValueError(f"Unknown game: {args.game}")


if __name__ == "__main__":
    main()