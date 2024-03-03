from alphazero.games.othello import OthelloConfig, OthelloBoard, OthelloNet


CONFIGS_REGISTER = {
    "othello": OthelloConfig,
}

BOARDS_REGISTER = {
    "othello": OthelloBoard
}

NETWORKS_REGISTER = {
    "othello": OthelloNet
}


def main():

    configs_keys = set(CONFIGS_REGISTER.keys())
    boards_keys = set(BOARDS_REGISTER.keys())
    networks_keys = set(NETWORKS_REGISTER.keys())

    assert configs_keys == boards_keys == networks_keys, "Registers have different keys."


if __name__ == "__main__":
    main()