from tqdm import tqdm

from alphazero.players import HumanPlayer, RandomPlayer, MCTSPlayer
from alphazero.arena import Arena
from alphazero.games.othello import OthelloBoard


def contest_MCTS_Random(n_rounds: int = 2, n_process: int = None, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and RandomPlayer. """

    player1 = MCTSPlayer(compute_time=0.5, verbose=False)
    player2 = RandomPlayer()
    board = OthelloBoard(8)

    arena = Arena(player1, player2, board)

    # stats = arena.play_games(n_rounds, verbose=verbose, return_stats=True)
    stats = arena.play_games_in_parallel(n_rounds, verbose=verbose, return_stats=True, n_process=n_process)

    print(stats)


if __name__ == "__main__":

    contest_MCTS_Random(n_rounds=10, n_process=None)