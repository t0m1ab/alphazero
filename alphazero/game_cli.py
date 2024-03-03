from alphazero.base import Board, Player
from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer
from alphazero.arena import Arena
from alphazero.games.othello import OthelloBoard
from alphazero.games.tictactoe import TicTacToeBoard


def othello():
    """ Play Othello with CLI. """
    player1 = HumanPlayer()
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = OthelloBoard(8)
    arena = Arena(player1, player2, board)
    arena.play_game(player2_starts=False, verbose=True, display=True)


def tictactoe():
    """ Play TicTacToe with CLI. """
    player1 = HumanPlayer()
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = TicTacToeBoard()
    arena = Arena(player1, player2, board)
    arena.play_game(player2_starts=False, verbose=True, display=True)


def main():
    # othello()
    tictactoe()


if __name__ == "__main__":
    main()