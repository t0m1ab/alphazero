import argparse

from alphazero.base import Board, Player, PolicyValueNetwork
from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer, AlphaZeroPlayer
from alphazero.arena import Arena
from alphazero.games.othello import OthelloBoard, OthelloNet
from alphazero.games.tictactoe import TicTacToeBoard, TicTacToeNet
from alphazero.games.connect4 import Connect4Board, Connect4Net
from alphazero.games.registers import BOARDS_REGISTER


def othello(display_mode: str = None):
    """ Play Othello with CLI. """
    player1 = HumanPlayer(game="othello")
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = OthelloBoard(6, display_mode=display_mode)
    arena = Arena(player1, player2, board)
    arena.play_game(
        player2_starts=False, 
        verbose=True, 
        display=True, 
        save_frames=False,
        show_indexes=True,
    )


def tictactoe(display_mode: str = None):
    """ Play TicTacToe with CLI. """
    player1 = HumanPlayer(game="tictactoe")
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = TicTacToeBoard(display_mode=display_mode)
    arena = Arena(player1, player2, board)
    arena.play_game(
        player2_starts=False, 
        verbose=True, 
        display=True, 
        save_frames=False,
        show_indexes=True,
    )


def connect4(display_mode: str = None):
    """ Play Connect4 with CLI. """
    player1 = HumanPlayer(game="connect4")
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = Connect4Board(width=7, height=6, display_mode=display_mode)
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
        "-d",
        "--display",
        dest="display_mode",
        type=str,
        default="human",
        help="display mode for the game ('human' or 'pixel').",
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
        othello(display_mode=args.display_mode)
    elif args.game == "tictactoe":
        tictactoe(display_mode=args.display_mode)
    elif args.game == "connect4":
        connect4(display_mode=args.display_mode)
    else:
        raise ValueError(f"Unknown game: {args.game}")


if __name__ == "__main__":
    main()