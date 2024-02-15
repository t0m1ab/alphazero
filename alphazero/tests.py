from time import time


def test_object_creation():
    """ Test the creation of objects defined in the package. """

    start_test = time()

    from alphazero.base import main as base_main
    from alphazero.mcts import main as mcts_main
    from alphazero.players import main as players_main
    from alphazero.games.othello import main as othello_main
    from alphazero.arena import main as arena_main

    base_main()
    mcts_main()
    players_main()
    othello_main()
    arena_main()

    print(f"> Import test successfully executed in {time() - start_test:.3f} seconds.")


def run_tests():
    """ Run all tests for the package. """
    test_object_creation()


if __name__ == "__main__":
    run_tests()
