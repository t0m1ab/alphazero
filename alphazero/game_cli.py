import argparse

from alphazero.base import Board, Player, PolicyValueNetwork
from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer, AlphaZeroPlayer
from alphazero.arena import Arena
from alphazero.games.othello import OthelloBoard, OthelloNet
from alphazero.games.tictactoe import TicTacToeBoard, TicTacToeNet
from alphazero.games.connect4 import Connect4Board, Connect4Net
from alphazero.games.registers import GAMES_SET, NETWORKS_REGISTER


def othello(bot_player: Player, bot_starts: bool = False, display_mode: str = None, show_probs: bool = False):
    """ Play Othello with CLI. """
    human_player = HumanPlayer(game="othello")
    board = OthelloBoard(6, display_mode=display_mode)
    arena = Arena(bot_player, human_player, board) if bot_starts else Arena(human_player, bot_player, board)
    arena.play_game(
        player2_starts=False, 
        verbose=True, 
        display=True, 
        save_frames=False,
        show_indexes=True,
        show_probs=show_probs,
    )


def tictactoe(bot_player: Player, bot_starts: bool = False, display_mode: str = None, show_probs: bool = False):
    """ Play TicTacToe with CLI. """
    human_player = HumanPlayer(game="tictactoe")
    board = TicTacToeBoard(display_mode=display_mode)
    arena = Arena(bot_player, human_player, board) if bot_starts else Arena(human_player, bot_player, board)
    arena.play_game(
        player2_starts=False, 
        verbose=True, 
        display=True, 
        save_frames=False,
        show_indexes=True,
        show_probs=show_probs,
    )


def connect4(bot_player: Player, bot_starts: bool = False, display_mode: str = None, show_probs: bool = False):
    """ Play Connect4 with CLI. """
    human_player = HumanPlayer(game="connect4")
    board = Connect4Board(width=7, height=6, display_mode=display_mode)
    arena = Arena(bot_player, human_player, board) if bot_starts else Arena(human_player, bot_player, board)
    arena.play_game(
        player2_starts=False, 
        verbose=True, 
        display=True, 
        save_frames=False,
        show_indexes=True,
        show_probs=show_probs,
    )


def main():

    GAME_LAUNCHERS = {
        "othello": othello,
        "tictactoe": tictactoe,
        "connect4": connect4,
    }

    if not set(GAME_LAUNCHERS.keys()) == GAMES_SET:
        raise ValueError("The game launchers keys don't match the games set.")

    # parser
    parser = argparse.ArgumentParser()
    for game in GAMES_SET:
        parser.add_argument(
            f"--{game}",
            dest=f"{game}",
            action="store_true",
            help=f"launch a game of {game} if specified.",
        )
    parser.add_argument(
        "--random",
        dest="random",
        action="store_true",
        help="load a RandomPlayer instance if specified.",
    )
    parser.add_argument(
        "--greedy",
        dest="greedy",
        action="store_true",
        help="load a GreedyPlayer instance if specified.",
    )
    parser.add_argument(
        "--mcts",
        dest="mcts",
        action="store_true",
        help="load a MCTSPlayer instance if specified.",
    )
    parser.add_argument(
        "-n",
        "--net",
        dest="net",
        type=str,
        default=None,
        help="name of the network to load into an AlphaZeroPlayer instance.",
    )
    parser.add_argument(
        "-b",
        "--bot-starts",
        dest="bot_starts",
        action="store_true",
        help="if set then the bot starts the game.",
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
        "-i",
        "--infos",
        dest="infos",
        action="store_true",
        help="if set then display probs and visit counts after each move.",
    )
    args = parser.parse_args()

    # game
    if args.othello:
        game = "othello"
    elif args.tictactoe:
        game = "tictactoe"
    elif args.connect4:
        game = "connect4"
    else:
        game = list(GAMES_SET)[0] # default game

    # bot player
    bot_player = None
    if args.random: # Random bot
        bot_player = RandomPlayer()
    elif args.greedy: # Greedy bot
        bot_player = GreedyPlayer()
    elif args.mcts or args.net is None: # MCTS bot = default bot
        bot_player = MCTSPlayer(n_sim=100, verbose=True)
    elif args.net is not None: # AlphaZero bot
        net = NETWORKS_REGISTER[game].from_pretrained(args.net)
        bot_player = AlphaZeroPlayer(n_sim=100, nn=net, verbose=True)

    # game launcher
    game_launcher = GAME_LAUNCHERS[game]

    # LAUNCH GAME
    game_launcher(bot_player, bot_starts=args.bot_starts, display_mode=args.display_mode, show_probs=args.infos)


if __name__ == "__main__":
    main()
    # EXAMPLE COMMANDS:
    # python game_cli.py --othello --random
    # python game_cli.py --tictactoe --greedy
    # python game_cli.py --connect4 --mcts
    # python game_cli.py --othello --net "alphazero-othello-6x6" --infos
    # python game_cli.py --tictactoe -n "alphazero-tictactoe" --display "pixel" --bot-starts