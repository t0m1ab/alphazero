import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

import alphazero
from alphazero.base import PolicyValueNetwork
from alphazero.utils import dotdict, push_model_to_hf_hub, DEFAULT_MODEL_PATH
from alphazero.players import AlphaZeroPlayer
from alphazero.games.registers import CONFIGS_REGISTER, BOARDS_REGISTER, NETWORKS_REGISTER
from alphazero.games.othello import OthelloBoard, OthelloNet, OthelloConfig
from alphazero.timer import SelfPlayTimer, NeuralTimer


class Sample():
    """
    Sample obtained when playing a move following the AlphaZero procedure and use to train PolicyValueNetwork.
    """

    def __init__(
            self, 
            state: np.ndarray, 
            pi: np.ndarray, 
            player: int, 
            outcome: int = None,
            episode_idx: int = None,
            move_idx: int = None,
        ) -> None:
        """
        ARGUMENTS:
            - state: board representation before the move.
            - pi: policy vector given by the AlphaZero MCTS procedure for the current state.
            - to_play: id of the player about to play.
            - outcome: outcome of the associated game (id of the winner (1 or -1) and 0 if draw).
        """
        self.state = state
        self.pi = pi
        self.player = player
        self.outcome = outcome
        self.episode_idx = episode_idx
        self.move_idx = move_idx
    
    def __str__(self) -> str:
        return f"Sample( \
            \n\tstate={self.state}\
            \n\tpi={self.pi} \
            \n\tplayer={self.player} \
            \n\toutcome={self.outcome} \
            \n\tepisode_idx={self.episode_idx} \
            \n\tplayer_idx={self.move_idx} \
        \n)"
    
    def normalize(self) -> None:
        """ Normalize the sample to self.player = 1. """
        self.state = self.state * self.player
        self.outcome = self.outcome * self.player


class AlphaZeroTrainer:

    DEFAULT_EXP_NAME = "alphazero-undefined"

    def __init__(self, game: str, verbose: bool = False):
        if game in CONFIGS_REGISTER:
            self.game = game # str: name of the game
        else:
            raise ValueError(f"Game '{game}' is not supported by AlphaZeroTrainer.")
        self.config = None # dotdict: configuration parameters
        self.board = None # Board object
        self.nn = None # PolicyValueNetwork object
        self.nn_twin = None # nn clone to use for specific training methods
        self.az_player = None # AlphaZeroPlayer object 
        self.memory = None # list storing normalize Sample objects to use for training the nn
        self.verbose = verbose
        self.loss_values = defaultdict(dict) # store loss values for each iteration and epoch
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.game.capitalize()}"
    
    def load_json_config(self, json_config_file: str):
        """ Load the configuration from a JSON file. """
        default_config = CONFIGS_REGISTER[self.game].to_dict()
        with open(json_config_file, "r") as f:
            json_config = json.load(f)
        
        # check if the configuration parameters are valid
        for config_param in json_config.keys():
            if not config_param in default_config:
                raise ValueError(f"Unknown configuration parameter '{config_param}' for game {self.game.capitalize()}")
        
        # check if the configuration parameters are complete
        for config_param in default_config:
            if not config_param in json_config:
                raise ValueError(f"Missing configuration parameter '{config_param}' for game {self.game.capitalize()}")

        return dotdict(json_config)
    
    def print(self, log: str):
        print(log) if self.verbose else None

    def print_config(self, verbose: bool = True):
        """ Print the configuration parameters. """
        if verbose:
            for pname, value in self.config.items():
                print(f"- {pname}: {value}")
    
    def get_training_time_estimation(self) -> float:
        """ 
        Return the estimated time for the training. 
        Use SelfPlayTimer to estimate duration of an episode of the game.
        Use NeuralTimer to estimate duration of neural network optimization over a batch.
        """

        self.print_config()

        # compute episode duration approximations
        spt = SelfPlayTimer(game=self.game, config=self.config)
        episode_duration_sec, episode_duration_steps = spt.timeit(n_episodes=10)

        # compute batch optimization duration approximation
        nt = NeuralTimer(game=self.game, config=self.config)
        batch_optim_duration = nt.timeit(n_batches=100)
        
        # compute duration estimations for all steps
        self_play_duration = self.config.episodes * episode_duration_sec
        n_samples = self.config.episodes * episode_duration_steps
        n_batches = n_samples // self.config.batch_size if self.config.batch_size < n_samples else 1
        epoch_optim_duration = batch_optim_duration * n_batches
        optim_duration = self.config.epochs * epoch_optim_duration
        iter_duration = self_play_duration + optim_duration
        training_duration = self.config.iterations * iter_duration

        # print the results
        sp_datetime = str(timedelta(seconds=round(self_play_duration)))
        opt_datetime = str(timedelta(seconds=round(optim_duration)))
        iter_datetime = str(timedelta(seconds=round(iter_duration)))
        train_datetime = str(timedelta(seconds=round(training_duration)))
        print(f"Self-play duration: {sp_datetime} (h:m:s)")
        print(f"Optimization duration: {opt_datetime} (h:m:s)")
        print(f"Iteration duration: {iter_datetime} (h:m:s)")
        print(f"Training duration: {train_datetime} (h:m:s)")

    def self_play(self, iter_idx: int):
        """ Simulate self-play games and add the generated data to the memory. """

        self.memory = [] # reset the memory

        self.print("")
        if self.verbose:
            pbar = tqdm(range(self.config.episodes), desc=f"Self-play ({self.config.episodes} episodes)")
        else:
            pbar = range(self.config.episodes)

        for episode_idx in pbar:

            # initialize the board and the players
            self.board.reset()
            self.az_player.reset() # reset the MCT
            
            # self-play
            memory_buffer = [] # temporary memory for the current episode
            while not self.board.is_game_over():
                
                # get best move for the current player
                move, move_probs = self.az_player.get_move(self.board, return_action_probs=True)

                # store the sample in the memory
                memory_buffer.append(Sample(
                    state=self.board.cells.copy(), 
                    pi=self.nn.to_neural_array(move_probs),
                    player=self.board.player,
                    episode_idx=episode_idx,
                    move_idx=len(memory_buffer),
                ))

                # play the move on the board
                self.board.play_move(move)

                # update internal state of the player
                self.az_player.apply_move(move, player=-self.board.player)
            
            # update samples in memory_buffer with the outcome of the game
            outcome = self.board.get_winner()
            for sample in memory_buffer:
                sample.outcome = outcome
                sample.normalize() # set the game in a position such that player with id=1 needs to play
            
            # add the samples to the memory
            self.memory += memory_buffer

            if self.verbose:
                pbar.set_postfix({"n_samples": len(self.memory)})
        
        del memory_buffer
    
    def _memory_generator(self, n_batches: int):
        """ Generator to iterate over the samples in the memory with the required batch size. """

        # shuffle the memory (by index)
        indexes = np.arange(len(self.memory))
        np.random.shuffle(indexes)
        indexes = indexes[:n_batches*self.config.batch_size].reshape(n_batches, self.config.batch_size)

        for batch_indexes in indexes:

            input_batch = np.zeros((self.config.batch_size, self.config.board_size, self.config.board_size))
            pi_batch = np.zeros((self.config.batch_size, self.board.get_action_size()))
            outcome_batch = np.zeros((self.config.batch_size, 1))

            for bidx, idx in enumerate(batch_indexes):
                input_batch[bidx] = self.memory[idx].state
                pi_batch[bidx] = self.memory[idx].pi
                outcome_batch[bidx] = self.memory[idx].outcome
            
            torch_input = torch.tensor(input_batch, dtype=torch.float32, device=self.config.device)
            torch_pi = torch.tensor(pi_batch, dtype=torch.float32, device=self.config.device)
            torch_outcome = torch.tensor(outcome_batch, dtype=torch.float32, device=self.config.device)

            yield torch_input, torch_pi, torch_outcome

    def optimize_network(self, iter_idx: int):
        """ Update the neural network using the collected data in memory. """

        self.nn_twin = self.nn.clone() # create a twin network to train

        self.nn_twin.train() # set the twin network in training mode

        optimizer = torch.optim.SGD(self.nn_twin.parameters(), lr=self.config.learning_rate)

        self.print(f"\nNetwork optimization ({self.config.epochs} epochs)")
        for epoch_idx in range(self.config.epochs):

            batch_losses = []

            n_batches = len(self.memory)//self.config.batch_size if self.config.batch_size < len(self.memory) else 1
            if self.verbose:
                pbar = tqdm(
                    self._memory_generator(n_batches), 
                    total=n_batches,
                    desc=f"Epoch {epoch_idx+1}/{self.config.epochs}",
                )
            else:
                pbar = self._memory_generator(n_batches)
            
            for input, pi, z in pbar:
                
                optimizer.zero_grad()
                
                log_probs, v = self.nn_twin(input)

                loss = torch.sum((v - z) ** 2) - torch.sum(pi * log_probs)
                
                loss.backward()

                optimizer.step()

                batch_losses.append(loss.cpu().item())

                if self.verbose:
                    pbar.set_postfix({"loss": batch_losses[-1]})
            
            self.loss_values[iter_idx][epoch_idx] = list(batch_losses)
    
    def update_network(self, iter_idx: int):
        """ Update the main network <self.nn> with the twin network <self.nn_twin>. """
        self.nn = self.nn_twin.clone()
        self.nn_twin = None
    
    def save_player(self, model_name: str, model_path: str = None):
        """ Save the trained neural network with the corresponding config file locally. """

        model_path = os.path.join(DEFAULT_MODEL_PATH, model_name) if model_path is None else model_path
        
        # save the neural network weights
        self.nn.save_model(model_name, model_path, verbose=False)

        # save the config file (path already exists because save_model created it just before)
        with open(os.path.join(model_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        
        # save the loss values during training
        with open(os.path.join(model_path, "loss.json"), "w") as f:
            json.dump(self.loss_values, f, indent=4)

    def train(
            self,
            json_config_file: str = None,
            experiment_name: str = None, 
            save: bool = True, 
            push_to_hub: bool = False, 
            save_checkpoints: bool = False,
            verbose: bool = None
        ):
        """ Train the AlphaZero player for the specified game. """

        # load/init the configuration
        if json_config_file is None: # load the default configuration
            self.config = CONFIGS_REGISTER[self.game].to_dict()
        else:
            self.config = self.load_json_config(json_config_file)
        if self.game != self.config.game:
            raise ValueError(f"Game '{self.game}' and game '{self.config.game}' in the configuration file do not match.")
        
        experiment_name = experiment_name if experiment_name is not None else AlphaZeroTrainer.DEFAULT_EXP_NAME
        save = save or push_to_hub # model must be saved locally before pushing to the hub
        self.verbose = verbose if verbose is not None else self.verbose

        # init the main objects
        self.print("[1] Initializing the Board, PolicyValueNetwork and Player...") 
        self.board = BOARDS_REGISTER[self.game](n=self.config.board_size)
        self.nn = NETWORKS_REGISTER[self.game](n=self.config.board_size, device=self.config.device)
        self.az_player = AlphaZeroPlayer(
            n_sim=self.config.simulations, 
            compute_time=self.config.compute_time, 
            nn=self.nn, 
            verbose=verbose
        )
        self.loss_values = defaultdict(dict) # reset the loss values
        self.print("")
        self.print_config(self.verbose)

        # training loop
        self.print("\n[2] Training...")
        for iter_idx in range(self.config.iterations):
            
            self.print(f"\n----- Iteration {iter_idx+1}/{self.config.iterations} -----")
            
            # simulate self-play games (generate game data and store it in self.memory)
            self.self_play(iter_idx)

            # optimize a twin neural network using the collected data in self.memory
            self.optimize_network(iter_idx)

            # update the main network with the twin network
            self.update_network(iter_idx)

            # save a checkpoint
            if save_checkpoints:
                chkpt_name = f"{experiment_name}-chkpt-{iter_idx+1}"
                self.save_player(
                    model_name=chkpt_name, 
                    model_path=os.path.join(DEFAULT_MODEL_PATH, experiment_name, chkpt_name)
                )
                self.print(f"\n{self} checkpoint {iter_idx+1}/{self.config.iterations} successfully saved.")
        
        if save:
            self.save_player(experiment_name)
            self.print(f"\n[3] {self} successfully saved!")
        
        if push_to_hub:
            self.print(f"\n[4] Pushing {self.az_player} to the Hugging Face Hub...\n")
            push_model_to_hf_hub(model_name=experiment_name, verbose=self.verbose)


def tests():

    for game in CONFIGS_REGISTER:
        trainer = AlphaZeroTrainer(game=game)
        print(f"{trainer} trainer created successfully!")


def main():

    trainer = AlphaZeroTrainer(game="othello", verbose=True)

    # trainer.get_training_time_estimation()

    trainer.train(
        json_config_file=None,
        experiment_name="alphazero-fake-custom", 
        save=True, 
        push_to_hub=False,
        save_checkpoints=True,
    )


if __name__ == "__main__":
    main()