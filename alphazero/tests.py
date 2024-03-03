from time import time


def test_object_creation():
    """ Test the creation of objects defined in the package. """

    start_test = time()

    from alphazero.base import main as base_main
    from alphazero.mcts import main as mcts_main
    from alphazero.players import main as players_main
    from alphazero.games.othello import main as othello_main
    from alphazero.arena import main as arena_main
    from alphazero.trainer import tests as trainers_main
    from alphazero.timers import main as timer_main

    base_main()
    mcts_main()
    players_main()
    othello_main()
    arena_main()
    trainers_main()
    timer_main()

    print(f"> Object initialization successfully executed in {time() - start_test:.3f} seconds.")


def run_tests():
    """ Run all tests for the package. """

    from alphazero.utils import tests as utils_tests
    utils_tests() # test utils

    test_object_creation()

    from alphazero.games.registers import main as register_main
    register_main() # check registers consistency


if __name__ == "__main__":
    run_tests()
