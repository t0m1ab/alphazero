from alphazero.base import Board, Player
from alphazero.players import HumanPlayer, RandomPlayer, MCTSPlayer
from alphazero.arena import Arena


def main():

    player1 = HumanPlayer()
    player2 = MCTSPlayer(compute_time=2.0, verbose=True)

    arena = Arena(player1, player2, game="othello", n=8, display_dir=None)

    arena.play_game(player2_starts=False, verbose=True, display=True)


if __name__ == "__main__":
    main()