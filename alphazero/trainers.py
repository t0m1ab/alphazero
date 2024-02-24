import os
from pathlib import Path
import json

import alphazero
from alphazero.players import AlphaZeroPlayer
from alphazero.games.othello import OthelloBoard, OthelloNet, OthelloConfig


class AlphaZeroTrainer:

    DEFAULT_SAVE_DIR = os.path.join(alphazero.__path__[0], "outputs/")

    DEFAULT_EXP_NAME = "alphazero_test"

    CONFIGS = {
        "othello": OthelloConfig
    }

    def __init__(self, game: str, json_config_file: str = None, save_dir: str = None):
        
        # load/init the configuration
        if game in AlphaZeroTrainer.CONFIGS:
            self.game = game # store the name of the game
            if json_config_file is None: # load the default configuration
                self.config = AlphaZeroTrainer.CONFIGS[game].to_dict()
            else:
                self.config = self.load_config(json_config_file)
        else:
            raise ValueError(f"Game '{game}' is not supported by AlphaZeroTrainer.")
        for pname, value in self.config.items(): # set the configuration parameters as attributes
            setattr(self, f"_{pname}", value) # EPOCHS -> "epochs" -> self._epochs for example

        # main objects
        self.board = None
        self.nn = None
        self.az_player = None

        self.save_dir = save_dir if save_dir is not None else AlphaZeroTrainer.DEFAULT_SAVE_DIR
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.game.capitalize()}"
    
    def load_config(self, json_config_file: str):
        """ Load the configuration from a JSON file. """
        with open(json_config_file, "r") as f:
            json_config = json.load(f)

        # check if the configuration parameters are valid
        for config_param in json_config.keys():
            if not config_param in AlphaZeroTrainer.CONFIGS[self.game].to_dict():
                raise ValueError(f"Unknown configuration parameter '{config_param}' for game '{self.game}'")
        
        # check if the configuration parameters are complete
        for config_param in AlphaZeroTrainer.CONFIGS[self.game].to_dict():
            if not config_param in json_config:
                raise ValueError(f"Missing configuration parameter '{config_param}' for game '{self.game}'")

        return json_config

    def print_config(self):
        """ Print the configuration parameters. """
        print(f"# Configuration for {self}:")
        for pname, value in self.config.items():
            print(f"- {pname} = {value}")

    def train(self, experiment_name: str = None, save_dir: str = None, verbose: bool = False):
        """ Train the AlphaZero player for the specified game. """
        
        if save_dir is not None:
            self.save_dir = save_dir
        experiment_name = experiment_name if experiment_name is not None else AlphaZeroTrainer.DEFAULT_EXP_NAME

        self.print_config()

        # init the main objects
        print("\nInitializing the Board, PolicyValueNetwork and Player...")
        self.board = OthelloBoard(n=self._board_size)
        self.nn = OthelloNet(n=self._board_size, device=self._device)
        self.az_player = AlphaZeroPlayer(n_sim=self._simulations, nn=self.nn, verbose=verbose)

        print("\nTraining...")
        for epoch_idx in range(self._epochs):

            episodes_scores = []
            episodes_lengths = []
            for episode_idx in range(self._episodes):

                # initialize the board and the players
                self.board.reset()
                self.az_player.reset() # reset the MCT
                
                # self-play
                c = 0
                while not self.board.is_game_over():
                    
                    # get best move for the current player
                    move = self.az_player.get_move(self.board)

                    # play the move on the board
                    self.board.play_move(move)

                    # update internal state of the player
                    self.az_player.apply_move(move, player=-self.board.player)

                    c += 1
                
                score = abs(self.board.get_score())
                episodes_scores.append(score)
                episodes_lengths.append(c)
            
            print(f"Epoch {epoch_idx+1}/{self._epochs} scores: {episodes_scores}")
            print(f"Epoch {epoch_idx+1}/{self._epochs} lenghts: {episodes_lengths}")
        
        print("\nOVER.")


def main():

    for game in AlphaZeroTrainer.CONFIGS:
        trainer = AlphaZeroTrainer(game=game)
        print(f"{trainer} trainer created successfully!")


def fake_training():

    trainer = AlphaZeroTrainer(game="othello")
    trainer.train()


if __name__ == "__main__":
    # main()
    fake_training()