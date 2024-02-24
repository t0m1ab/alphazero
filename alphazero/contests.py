from tqdm import tqdm

from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer, AlphaZeroPlayer
from alphazero.arena import Arena
from alphazero.base import Player
from alphazero.games.othello import OthelloBoard, OthelloNet


def show_results(player1: Player, player2: Player, stats: dict, ):
    """ Print the results of a contest stored in <stats>. """
    n_rounds = len(stats["player1"]) + len(stats["player2"]) + stats["draw"]
    print(" ")
    print("\nRESULTS:")
    print(f"- {player1} wins = {len(stats['player1'])}/{n_rounds}")
    print(f"- {player2} wins = {len(stats['player2'])}/{n_rounds}")


def MCTS_vs_Random(n_rounds: int = 2, n_process: int = None, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and RandomPlayer. """

    player1 = MCTSPlayer(compute_time=0.2)
    player2 = RandomPlayer()
    board = OthelloBoard(6)

    arena = Arena(player1, player2, board)

    # stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose) # sequential
    stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True) # parallel

    show_results(player1, player2, stats)


def MCTS_vs_Greedy(n_rounds: int = 2, n_process: int = None, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and RandomPlayer. """

    player1 = MCTSPlayer(compute_time=0.2)
    player2 = GreedyPlayer()
    board = OthelloBoard(6)

    arena = Arena(player1, player2, board)

    # stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose) # sequential
    stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True) # parallel

    show_results(player1, player2, stats)


def AZ_vs_Greedy(n_rounds: int = 2, n_process: int = None, verbose: bool = False):
    """ Organize a contest between AlphaZeroPlayer and RandomPlayer. """

    board_dim = 6

    nn = OthelloNet(n=board_dim, device="cpu") # cuda or mps device will not work for parallel games
    player1 = AlphaZeroPlayer(compute_time=0.2, nn=nn, verbose=verbose)
    player2 = GreedyPlayer()
    board = OthelloBoard(n=board_dim)

    arena = Arena(player1, player2, board)

    # stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose) # sequential
    stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True) # parallel

    show_results(player1, player2, stats)


if __name__ == "__main__":
    # MCTS_vs_Random(n_rounds=50, n_process=10, verbose=False)
    # MCTS_vs_Greedy(n_rounds=50, n_process=10, verbose=False)
    AZ_vs_Greedy(n_rounds=50, n_process=10, verbose=False)

