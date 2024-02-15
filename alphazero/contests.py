from tqdm import tqdm

from alphazero.players import HumanPlayer, RandomPlayer, MCTSPlayer
from alphazero.arena import Arena
from alphazero.games.othello import OthelloBoard


def contest_MCTS_Random(n_rounds: int = 2, n_process: int = None, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and RandomPlayer. """

    player1 = MCTSPlayer(compute_time=0.2)
    player2 = RandomPlayer()
    board = OthelloBoard(6)

    arena = Arena(player1, player2, board)

    # stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose) # sequential
    stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True) # parallel

    print(stats)


if __name__ == "__main__":
    contest_MCTS_Random(n_rounds=10, n_process=10, verbose=False)