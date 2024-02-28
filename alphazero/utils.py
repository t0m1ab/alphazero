import numpy as np


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    def __getattr__(self, name):
        return self[name]


def fair_max(elements: list | np.ndarray, key = lambda x: x) -> int:
    """
    Returns the index of the maximum element in the collection <elements> by randomly resolving ties.
    """
    max_value = key(max(elements, key=key))
    max_elements = [x for x in elements if key(x) == max_value]
    return max_elements[np.random.choice(len(max_elements))]


def random_play(n: int = 8, n_turns: int = 100, display_dir: str = None) -> None:
    """ Generate a random game of Othello and display the board at the end. """

    # board = OthelloBoard(n=n, display_dir=display_dir)
    board = None

    for turn_i in range(n_turns):
        player1_random_move = board.get_random_move()
        board.play_move(player1_random_move)
        player2_random_move = board.get_random_move()
        board.play_move(player2_random_move)
        print(f"#{turn_i} - Player 1 played {player1_random_move} | Player 2 played {player2_random_move}")
        if player1_random_move == (n,n) and player2_random_move == (n,n):
            print("> Both players passed, game over!")
            score = board.get_score()
            if score == 0:
                print("> Draw!")
            else:
                print(f"> Player {board.get_winner()} won the game with score: {score}")
            break
    
    board.display(indexes=True)


def main():

    # test fair_max (at least it is a max operator)
    elements = [1, 2, 3, 3, 3, -4, -5]
    assert fair_max(elements) == 3


if __name__ == "__main__":
    main()