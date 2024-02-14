import os

import alphazero


class Board():

    DEFAULT_DISPLAY_DIR = os.path.join(alphazero.__path__[0], "outputs/")

    def __init__(self, display_dir: str = None):
        self.display_dir = display_dir if display_dir is not None else Board.DEFAULT_DISPLAY_DIR


def main():
    
    _ = Board()
    print("Board created successfully!")


if __name__ == "__main__":
    main()