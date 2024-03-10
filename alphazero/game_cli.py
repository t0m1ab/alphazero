from alphazero.base import Board, Player
from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer
from alphazero.arena import Arena
from alphazero.games.othello import OthelloBoard
from alphazero.games.tictactoe import TicTacToeBoard
from alphazero.games.connect4 import Connect4Board


def othello():
    """ Play Othello with CLI. """
    player1 = HumanPlayer(game="othello")
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = OthelloBoard(8)
    arena = Arena(player1, player2, board)
    arena.play_game(player2_starts=False, verbose=True, display=True)


def tictactoe():
    """ Play TicTacToe with CLI. """
    player1 = HumanPlayer(game="tictactoe")
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = TicTacToeBoard()
    arena = Arena(player1, player2, board)
    arena.play_game(player2_starts=False, verbose=True, display=True)


def connect4():
    """ Play Connect4 with CLI. """
    player1 = HumanPlayer(game="connect4")
    # player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    player2 = RandomPlayer()
    board = Connect4Board(width=7, height=6)
    arena = Arena(player1, player2, board)
    arena.play_game(player2_starts=False, verbose=True, display=True)


def main():
    # othello()
    # tictactoe()
    connect4()


if __name__ == "__main__":
    main()