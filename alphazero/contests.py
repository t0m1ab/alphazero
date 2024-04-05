from tqdm import tqdm

from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer, AlphaZeroPlayer
from alphazero.arena import Arena
from alphazero.base import Player
from alphazero.games.othello import OthelloBoard, OthelloNet
from alphazero.games.tictactoe import TicTacToeBoard, TicTacToeNet
from alphazero.games.connect4 import Connect4Board, Connect4Net


def Greedy_vs_Random_Othello(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between GreedyPlayer and RandomPlayer. """

    player1 = GreedyPlayer()
    player2 = RandomPlayer()
    board = OthelloBoard(6)

    arena = Arena(player1, player2, board)

    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)

def MCTS_vs_Random_Othello(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and RandomPlayer. """

    # player1 = MCTSPlayer(compute_time=0.2)
    player1 = MCTSPlayer(n_sim=100)
    player2 = RandomPlayer()
    board = OthelloBoard(8)

    arena = Arena(player1, player2, board)

    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)

def MCTS_vs_Greedy_Othello(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and GreedyPlayer. """

    player1 = MCTSPlayer(n_sim=100)
    player2 = GreedyPlayer()
    board = OthelloBoard(6)

    arena = Arena(player1, player2, board)

    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)

def AZ_vs_Greedy_Othello(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between AlphaZeroPlayer and GreedyPlayer. """

    n = 6

    net = OthelloNet(n=n, device="cpu") # cuda or mps device will not work for parallel games
    # net = OthelloNet.from_pretrained(model_name="alphazero-othello") # uncomment this line to use a pretrained model
    player1 = AlphaZeroPlayer(n_sim=100, nn=net, verbose=verbose)
    player2 = GreedyPlayer()
    board = OthelloBoard(n=n)

    arena = Arena(player1, player2, board)

    # set the following verbose to True to show stats for each move
    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=False)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=False, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)


def MCTS_vs_Random_TicTacToe(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and RandomPlayer. """

    player1 = MCTSPlayer(n_sim=100)
    player2 = RandomPlayer()
    board = TicTacToeBoard()

    arena = Arena(player1, player2, board)

    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)

def MCTS_vs_Greedy_TicTacToe(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and GreedyPlayer. """

    player1 = MCTSPlayer(n_sim=100)
    player2 = GreedyPlayer()
    board = TicTacToeBoard()

    arena = Arena(player1, player2, board)

    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)

def AZ_vs_Greedy_TicTacToe(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between AlphaZeroPlayer and GreedyPlayer. """

    net = TicTacToeNet(device="cpu") # cuda or mps device will not work for parallel games
    # net = TicTacToeNet.from_pretrained(model_name="alphazero-tictactoe") # uncomment this line to use a pretrained model
    player1 = AlphaZeroPlayer(n_sim=100, nn=net, verbose=verbose)
    player2 = MCTSPlayer(n_sim=100)
    board = TicTacToeBoard()

    arena = Arena(player1, player2, board)

    # set the following verbose to True to show stats for each move
    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=False)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=False, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)


def MCTS_vs_Random_Connect4(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and RandomPlayer. """

    player1 = MCTSPlayer(compute_time=0.2)
    player2 = RandomPlayer()
    board = Connect4Board(width=7, height=6)

    arena = Arena(player1, player2, board)

    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)

def MCTS_vs_Greedy_Connect4(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between MCTSPlayer and GreedyPlayer. """

    player1 = MCTSPlayer(compute_time=0.2)
    player2 = GreedyPlayer()
    board = Connect4Board(width=7, height=6)

    arena = Arena(player1, player2, board)

    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=verbose)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=verbose, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)

def AZ_vs_Greedy_Connect4(n_rounds: int = 2, n_process: int = 1, verbose: bool = False):
    """ Organize a contest between AlphaZeroPlayer and GreedyPlayer. """

    net = Connect4Net(board_width=7, board_height=6, device="cpu") # cuda or mps device will not work for parallel games
    # net = Connect4Net.from_pretrained(model_name="alphazero-connect4") # uncomment this line to use a pretrained model
    player1 = AlphaZeroPlayer(n_sim=100, nn=net, verbose=verbose)
    player2 = GreedyPlayer()
    board = Connect4Board(width=7, height=6)

    arena = Arena(player1, player2, board)

    # set the following verbose to True to show stats for each move
    if n_process == 1: # sequential
        stats = arena.play_games(n_rounds, return_stats=True, verbose=False)
    else: # parallel (use all available cores if n_process is None)
        stats = arena.play_games_in_parallel(n_rounds, n_process=n_process, verbose=False, return_stats=True)

    Arena.print_stats_results(player1, player2, stats)



if __name__ == "__main__":

    # Greedy_vs_Random_Othello(n_rounds=1000, n_process=10, verbose=False)
    # MCTS_vs_Random_Othello(n_rounds=100, n_process=10, verbose=False)
    # MCTS_vs_Greedy_Othello(n_rounds=100, n_process=10, verbose=False)
    # AZ_vs_Greedy_Othello(n_rounds=100, n_process=4, verbose=False)

    # MCTS_vs_Random_TicTacToe(n_rounds=50, n_process=10, verbose=False)
    # MCTS_vs_Greedy_TicTacToe(n_rounds=50, n_process=10, verbose=False)
    # AZ_vs_Greedy_TicTacToe(n_rounds=100, n_process=10, verbose=False)

    # MCTS_vs_Random_Connect4(n_rounds=50, n_process=10, verbose=False)
    # MCTS_vs_Greedy_Connect4(n_rounds=50, n_process=10, verbose=False)
    # AZ_vs_Greedy_Connect4(n_rounds=100, n_process=4, verbose=False)

    print("Uncomment one of the previous lines to run a contest!")

