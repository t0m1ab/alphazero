from alphazero.base import Board, Player
from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer
from alphazero.arena import Arena
from alphazero.games.othello import OthelloBoard


def main():

    player1 = HumanPlayer()
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)
    board = OthelloBoard(8)

    arena = Arena(player1, player2, board)

    arena.play_game(player2_starts=False, verbose=True, display=True)


if __name__ == "__main__":
    main()