import os
from pathlib import Path
from typing import Callable
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

import alphazero
from alphazero.base import Config, DataTransf
from alphazero.utils import dotdict, push_model_to_hf_hub, DEFAULT_CONFIGS_PATH, DEFAULT_MODELS_PATH
from alphazero.players import HumanPlayer, RandomPlayer, GreedyPlayer, MCTSPlayer, AlphaZeroPlayer, PLAYERS_SET, PLAYERS_REGISTER
from alphazero.games.othello import OthelloBoard, OthelloNet, OthelloConfig
from alphazero.timers import SelfPlayTimer, NeuralTimer
from alphazero.schedulers import TEMP_SCHEDULERS
from alphazero.arena import Arena
from alphazero.visualization import plot_loss, plot_eval_results
from alphazero.games.registers import (
    GAMES_SET,
    CONFIGS_REGISTER, 
    BOARDS_REGISTER, 
    NETWORKS_REGISTER, 
    DATA_AUGMENT_STRATEGIES,
)


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
            transformation: str = None,
        ) -> None:
        """
        ARGUMENTS:
            - state: board representation before the move.
            - pi: policy vector given by the AlphaZero MCTS procedure for the current state.
            - to_play: id of the player about to play.
            - outcome: outcome of the associated game (id of the winner (1 or -1) and 0 if draw).
            - episode_idx: index of the episode in which the sample was generated.
            - move_idx: index of the move in the episode.
            - transformation: nature of the transformation applied to create the sample (None if it is an original sample)
        """
        self.state = state
        self.pi = pi
        self.player = player
        self.outcome = outcome
        self.episode_idx = episode_idx
        self.move_idx = move_idx
        self.transformation = transformation
    
    def __str__(self) -> str:
        return f"Sample( \
            \n\tstate={self.state}\
            \n\tpi={self.pi} \
            \n\tplayer={self.player} \
            \n\toutcome={self.outcome} \
            \n\tepisode_idx={self.episode_idx} \
            \n\tmove_idx={self.move_idx} \
            \n\ttransformation={self.transformation} \
        \n)"
    
    def normalize(self) -> None:
        """ Normalize the sample to self.player = 1. """
        self.state = self.state * self.player
        self.outcome = self.outcome * self.player
        self.player = 1
    
    def create_reflection_twin(self, reflection: Callable, mode: DataTransf) -> "Sample":
        """ Return a reflected version of <self> using the reflection function <reflection> in mode <mode>. """

        if not mode in [DataTransf.REFLECT_H, DataTransf.REFLECT_V]:
            raise ValueError(f"Reflection mode must be either {DataTransf.REFLECT_H} or {DataTransf.REFLECT_V}.")
        
        axis = 1 if mode == DataTransf.REFLECT_H else 0

        return Sample(
            state=np.flip(self.state.copy(), axis=axis),
            pi=reflection(self.pi.copy(), axis=axis),
            player=self.player,
            outcome=self.outcome,
            episode_idx=self.episode_idx,
            move_idx=self.move_idx,
            transformation=mode.value if self.transformation is None else f"{self.transformation}+{mode.value}",
        )
    
    def create_rotation_twin(self, rotation: Callable, mode: DataTransf) -> "Sample":
        """ Return a rotated version of <self> using the rotation function <rotation> in mode <mode>. """

        if not mode in [DataTransf.ROTATE_90, DataTransf.ROTATE_180, DataTransf.ROTATE_270]:
            raise ValueError(f"Rotation mode must be either {DataTransf.ROTATE_90}, {DataTransf.ROTATE_180} or {DataTransf.ROTATE_270}.")

        angle = 90
        if mode == DataTransf.ROTATE_180:
            angle = 180
        elif mode == DataTransf.ROTATE_270:
            angle = 270

        return Sample(
            state=np.rot90(self.state.copy(), k=angle//90),
            pi=rotation(self.pi.copy(), angle=angle),
            player=self.player,
            outcome=self.outcome,
            episode_idx=self.episode_idx,
            move_idx=self.move_idx,
            transformation=mode.value if self.transformation is None else f"{self.transformation}+{mode.value}",
        )


class AlphaZeroTrainer:

    DEFAULT_EXP_NAME = "alphazero-undefined"

    def __init__(self, verbose: bool = False):
        self.game = None # str: name of the game (ex: "othello", "tictactoe"...)
        self.config = None # Config: configuration parameters
        self.board = None # Board
        self.nn = None # PolicyValueNetwork
        self.nn_twin = None # PolicyValueNetwork object: nn clone to use for specific training methods
        self.az_player = None # AlphaZeroPlayer 
        self.temp_scheduler = None # TemperatureScheduler: temperature scheduler for the game
        self.data_augment_strategy = None # dict: data augmentation strategy for the game
        self.memory = None # list[Sample]: list storing normalize samples to use nn training
        self.verbose = verbose # bool
        self.loss_values = None # dict: store loss values for each iteration and epoch
        self.eval_results = None # dict: store evaluation results for each iteration (if self.config.do_eval is True)
    
    def __str__(self) -> str:
        if self.game is not None:
            return f"{self.__class__.__name__}{self.game.capitalize()}"
        else:
            return f"{self.__class__.__name__}"
        
    def print(self, log: str):
        if self.verbose:
            print(log)

    @staticmethod
    def print_config(config: Config, verbose: bool = True):
        """ Print the configuration parameters. """
        if verbose:
            for pname, value in config.to_dict().items():
                print(f"- {pname}: {value}")
    
    @staticmethod
    def load_config_from_json(game: str, json_config_file: str):
        """ Load the configuration from a JSON file. """
        if json_config_file is None: # load the default configuration
            return CONFIGS_REGISTER[game]()
        else: # create a config from the JSON file
            with open(json_config_file, "r") as f:
                json_config = dotdict(json.load(f))
            return CONFIGS_REGISTER[json_config.game](**json_config)

    @staticmethod
    def estimate_training_duration(game: str, json_config_file: str = None):
        """ 
        Return the estimated time for the training. 
        Use SelfPlayTimer to estimate duration of an episode of the game.
        Use NeuralTimer to estimate duration of neural network optimization over a batch.
        
        ARGUMENTS:
            - json_config_file: path to the JSON configuration file. Use the default configuration if None.
        """

        # load/init the configuration
        config = AlphaZeroTrainer.load_config_from_json(game, json_config_file)
        game = game if game is not None else config.game
        if game is not None and game != config.game:
            raise ValueError(f"Game '{game}' and game '{config.game}' in the configuration file do not match.")
        AlphaZeroTrainer.print_config(config)

        # compute episode duration approximations
        spt = SelfPlayTimer(game, config)
        episode_duration_sec, episode_duration_steps = spt.timeit(n_episodes=10)

        # compute batch optimization duration approximation
        nt = NeuralTimer(game, config)
        batch_optim_duration = nt.timeit(n_batches=100)
        
        # compute duration estimations for all steps
        self_play_duration = config.episodes * episode_duration_sec
        n_samples = config.episodes * episode_duration_steps
        n_batches = n_samples // config.batch_size if config.batch_size < n_samples else 1
        epoch_optim_duration = batch_optim_duration * n_batches
        optim_duration = config.epochs * epoch_optim_duration
        # approximate opponent player with alphazero playing duration
        eval_duration = config.eval_episodes * episode_duration_sec if config.do_eval else 0
        iter_duration = self_play_duration + optim_duration + eval_duration
        total_duration = config.iterations * iter_duration

        # print the results
        sp_datetime = str(timedelta(seconds=round(self_play_duration)))
        opt_datetime = str(timedelta(seconds=round(optim_duration)))
        iter_datetime = str(timedelta(seconds=round(iter_duration)))
        eval_datetime = str(timedelta(seconds=round(eval_duration)))
        total_datetime = str(timedelta(seconds=round(total_duration)))
        print(f"Self-play duration: {sp_datetime} (h:m:s)")
        print(f"Optimization duration: {opt_datetime} (h:m:s)")
        print(f"Iteration duration: {iter_datetime} (h:m:s)")
        print(f"Evaluation duration: {eval_datetime} (h:m:s)")
        print(f"TOTAL training duration: {total_datetime} (h:m:s)")

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
            move_counter = 0
            while not self.board.is_game_over():

                # get the temperature for the current move
                temp = self.temp_scheduler[move_counter]
                
                # get best move for the current player
                move, move_probs, _, _ = self.az_player.get_move(self.board, temp=temp)

                # store the sample in the memory
                memory_buffer.append(Sample(
                    state=self.board.grid.copy(), 
                    pi=self.nn.to_neural_output(move_probs),
                    player=self.board.player,
                    episode_idx=episode_idx,
                    move_idx=move_counter,
                ))

                # play the move on the board
                self.board.play_move(move)

                # update internal state of the player
                self.az_player.apply_move(move, player=-self.board.player)

                # increment the move counter
                move_counter += 1
            
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

        if self.config.data_augmentation: # augment the memory with reflections and rotations
            augmented_samples = []
            for sample in tqdm(self.memory, desc=f"Data augmentation (from {len(self.memory)} samples)") if self.verbose else self.memory:
                if sample.move_idx >= 2: # symmetries are likely to produce copies for the first moves
                    reflected_sample = sample.create_reflection_twin(self.nn.reflect_neural_output, mode=self.data_augment_strategy.reflection)
                    augmented_samples.append(reflected_sample)
                    for rot_mode in self.data_augment_strategy.rotations: # 90°, 180° and 270° rotations
                        augmented_samples.append(sample.create_rotation_twin(self.nn.rotate_neural_output, mode=rot_mode))
                        augmented_samples.append(reflected_sample.create_rotation_twin(self.nn.rotate_neural_output, mode=rot_mode))
            self.memory += augmented_samples
        
        self.print(f"Total number of samples: {len(self.memory)}")
           
    def _batch_generator(self, n_batches: int):
        """ Generator to iterate over the samples in the memory with the required batch size. """

        # shuffle the memory (by index)
        indexes = np.arange(len(self.memory))
        np.random.shuffle(indexes)
        indexes = indexes[:n_batches*self.config.batch_size].reshape(n_batches, self.config.batch_size)

        for batch_indexes in indexes:
            
            if hasattr(self.config, "board_size"):
                input_batch = np.zeros((self.config.batch_size, self.config.board_size, self.config.board_size))
            elif hasattr(self.config, "board_width") and hasattr(self.config, "board_height"):
                input_batch = np.zeros((self.config.batch_size, self.config.board_height, self.config.board_width))
            else:
                raise AttributeError("Board size/width/height not found in the config...")
    
            # input_batch = np.zeros((self.config.batch_size, self.config.board_size, self.config.board_size))
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
        from alphazero.games.connect4 import Connect4Board
        self.nn_twin = self.nn.clone() # create a twin network to train

        self.nn_twin.train() # set the twin network in training mode

        optimizer = torch.optim.SGD(self.nn_twin.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        self.loss_values[iter_idx] = dict()
        self.print(f"\nNetwork optimization ({self.config.epochs} epochs)")
        for epoch_idx in range(self.config.epochs):

            batch_pi_losses = []
            batch_v_losses = []

            n_batches = len(self.memory) // self.config.batch_size
            if n_batches == 0:
                raise ValueError(f"Too few samples in the memory ({len(self.memory)}) to create a batch with batch_size = {self.config.batch_size}")
            
            if self.verbose:
                pbar = tqdm(
                    self._batch_generator(n_batches), 
                    total=n_batches,
                    desc=f"Epoch {epoch_idx+1}/{self.config.epochs}",
                )
            else:
                pbar = self._batch_generator(n_batches)
            
            for input, pi, z in pbar:
                
                optimizer.zero_grad()
                
                log_probs, v = self.nn_twin(input)

                loss_pi = -torch.sum(pi * log_probs) / self.config.batch_size
                loss_v = torch.sum((v - z) ** 2) / self.config.batch_size
                loss = loss_pi + loss_v
                
                loss.backward()

                optimizer.step()

                batch_pi_losses.append(loss_pi.cpu().item())
                batch_v_losses.append(loss_v.cpu().item())

                if self.verbose:
                    pbar.set_postfix({
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss_pi": loss_pi.cpu().item(),
                        "loss_v": loss_v.cpu().item(),
                        "loss": batch_pi_losses[-1] + batch_v_losses[-1],
                        "mean_loss": np.mean(batch_pi_losses) + np.mean(batch_v_losses),
                    })
            
            self.loss_values[iter_idx][epoch_idx] = {
                "pi": list(batch_pi_losses),
                "v": list(batch_v_losses),
            }

            lr_scheduler.step()
            
    def update_network(self, iter_idx: int):
        """ Update the main network <self.nn> with the twin network <self.nn_twin>. """
        self.nn = self.nn_twin.clone()
        self.nn_twin = None
        self.az_player.mct.nn = self.nn # update the nn used in the MCT during self-play with the new network
    
    def __init_evaluator(self):
        """ Check if the opponent player is valid if <self.config.do_eval> is true. """

        if not self.config.do_eval:
            return

        if self.config.eval_opponent not in PLAYERS_SET:
            raise ValueError(f"Opponent player '{self.config.eval_opponent}' not found in the players register.")
        if self.config.eval_opponent == "human":
            raise ValueError("Evaluation against a HumanPlayer during training is not allowed.")
        if self.config.eval_opponent == "alphazero":
            raise ValueError("Evaluation against another AlphaZeroPlayer during training is not yet implemented.")
        
        self.eval_results = {
            "eval_opponent": self.config.eval_opponent,
            "eval_episodes": self.config.eval_episodes,
            "results": dict(),
        }

    def evaluate(self, iter_idx: int):
        """
        Evaluate the trained neural network using opponent <self.config.eval_opponent> for <self.config.eval_episodes> games.
        """

        if not self.config.do_eval:
            return
        
        kwargs = {}
        if self.config.eval_opponent == "mcts":
            kwargs["n_sim"] = self.config.simulations

        # create AlphaZero player for evaluation (no dirichlet noise)
        eval_player = AlphaZeroPlayer(
            n_sim=self.config.simulations, 
            compute_time=self.config.compute_time, 
            nn=self.nn,
        )

        # create opponent player for evaluation
        opponent_player = PLAYERS_REGISTER[self.config.eval_opponent](**kwargs)

        arena = Arena(
            player1=eval_player, 
            player2=opponent_player, 
            board=BOARDS_REGISTER[self.game](config=self.config)
        )

        self.print(f"\nAlphaZeroPlayer evaluation against {opponent_player} ({self.config.eval_episodes} episodes)")
        stats = arena.play_games(n_rounds=self.config.eval_episodes, return_stats=True)

        if self.verbose:
            Arena.print_stats_results(eval_player, opponent_player, stats)

        for key in ["player1", "player2", "draw"]:
            stats.pop(key) if key in stats else None
        self.eval_results["results"][iter_idx] = stats
        self.eval_results["player1"] = f"{eval_player}"
        self.eval_results["player2"] = f"{opponent_player}"
    
    def save_player_pt(self, model_name: str, path: str = None):
        """ Save the trained neural network. """
        path = os.path.join(DEFAULT_MODELS_PATH, model_name) if path is None else path
        Path(path).mkdir(parents=True, exist_ok=True)
        self.nn.save_model(model_name, path, verbose=False)
    
    def save_player_config(self, model_name: str, path: str = None):
        """ Save the trained neural network. """
        path = os.path.join(DEFAULT_MODELS_PATH, model_name) if path is None else path
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)
    
    def save_training_stats(self, model_name: str, path: str = None):
        """ Save training informations like loss and evaluation results. """
        path = os.path.join(DEFAULT_MODELS_PATH, model_name) if path is None else path
        Path(path).mkdir(parents=True, exist_ok=True)

        # save loss values
        with open(os.path.join(path, "loss.json"), "w") as f:
            json.dump(self.loss_values, f, indent=4)

        # save evaluation results
        if self.config.do_eval:
            with open(os.path.join(path, "eval.json"), "w") as f:
                json.dump(self.eval_results, f, indent=4)

    def train(
            self,
            game: str = None,
            experiment_name: str = None, 
            json_config_file: str = None,
            plot: bool = False,
            verbose: bool = None
        ):
        """ Train the AlphaZero player for the specified game. """

        # load/init the configuration
        if game is None and json_config_file is None:
            raise ValueError("The name of the game or a JSON configuration file must be provided to launch a training.")
        self.config = AlphaZeroTrainer.load_config_from_json(game, json_config_file) # will use either self.game or the config file
        self.game = game if game is not None else self.config.game
        if self.game is not None and self.game != self.config.game:
            raise ValueError(f"Game '{game}' and game '{self.config.game}' in the configuration file do not match.")
        experiment_name = experiment_name if experiment_name is not None else AlphaZeroTrainer.DEFAULT_EXP_NAME
        self.verbose = verbose if verbose is not None else self.verbose
        
        # adjust flags for saving and pushing the results of the training
        self.config.save_checkpoints = self.config.save_checkpoints or self.config.push_checkpoints # checkpoints must be saved before pushing to the hub
        self.config.push = self.config.push or self.config.push_checkpoints # if checkpoints are pushed, the final model is pushed too
        self.config.save = self.config.save or self.config.push or self.config.save_checkpoints # model must be saved locally before pushing to the hub

        # init the main objects
        self.print("[1] Initializing the Board, PolicyValueNetwork and Player...") 
        self.board = BOARDS_REGISTER[self.game](config=self.config)
        self.nn = NETWORKS_REGISTER[self.game](config=self.config)
        self.az_player = AlphaZeroPlayer(
            n_sim=self.config.simulations, 
            compute_time=self.config.compute_time, 
            nn=self.nn,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            verbose=verbose
        )
        self.temp_scheduler = TEMP_SCHEDULERS[self.config.temp_scheduler_type](
            temp_max_step=self.config.temp_max_step,
            temp_min_step=self.config.temp_min_step,
            max_steps=self.board.max_moves,
        )
        self.data_augment_strategy = DATA_AUGMENT_STRATEGIES[self.game] if self.config.data_augmentation else None
        self.__init_evaluator() # check if the opponent player is valid and init self.eval_results
        self.loss_values = dict()

        # print the configuration and save it in the new model directory
        self.print("")
        AlphaZeroTrainer.print_config(self.config, self.verbose)
        self.save_player_config(experiment_name)

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

            # evaluate the trained neural network against an opponent
            self.evaluate(iter_idx)

            # save loss values (+ evaluation results)
            self.save_training_stats(experiment_name)

            # save checkpoint in experiment_name/checkpoint/ directory
            if self.config.save_checkpoints:
                self.save_player_pt(
                    model_name=f"{experiment_name}-chkpt-{iter_idx+1}", 
                    path=os.path.join(DEFAULT_MODELS_PATH, experiment_name, "checkpoints")
                )
                self.print(f"\n{self} checkpoint {iter_idx+1}/{self.config.iterations} successfully saved.")
                    
        if self.config.save:
            self.save_player_pt(experiment_name)
            self.print(f"\n[3] {self} successfully saved!")
        
        if self.config.push:
            self.print(f"\n[4] Pushing {self.az_player} to the Hugging Face Hub...\n")
            chkpts_files = {}
            if self.config.push_checkpoints:
                for iter_idx in range(self.config.iterations):
                    chkpt_name = f"{experiment_name}-chkpt-{iter_idx+1}"
                    chkpts_files[f"{chkpt_name}.pt"] = os.path.join(DEFAULT_MODELS_PATH, experiment_name, f"checkpoints/{chkpt_name}.pt")
            push_model_to_hf_hub(model_name=experiment_name, additional_files=chkpts_files, verbose=self.verbose)
        
        if plot:
            self.print("\n[5] Plotting training results...")
            plot_loss(model_name=experiment_name)
            if self.config.do_eval:
                plot_eval_results(model_name=experiment_name)


def tests():
    trainer = AlphaZeroTrainer()
    print(f"{trainer} trainer created successfully!")


def freeze_config(game: str = None):
    """ Freeze the configuration parameters for <game> or for all games if <game> is None. """
    games = [game] if game is not None else list(GAMES_SET)
    Path(DEFAULT_CONFIGS_PATH).mkdir(parents=True, exist_ok=True)
    for game in games:
        config = AlphaZeroTrainer.load_config_from_json(game, json_config_file=None)
        with open(os.path.join(DEFAULT_CONFIGS_PATH, f"{game}.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=4)


def main():

    DEFAULT_GAME = None
    DEFAULT_EXP_NAME = "alphazero-fake"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--game",
        dest="game",
        type=str,
        default=DEFAULT_GAME,
        help="name of the game to train on (ex: 'othello', 'tictactoe'...).",
    )
    parser.add_argument(
        "-e",
        "--experiment-name",
        dest="experiment_name",
        type=str,
        default=DEFAULT_EXP_NAME,
        help="name of the experiment/trained AlphaZero.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="json_config_file",
        type=str,
        default=None,
        help="path to a JSON config file for the training.",
    )
    parser.add_argument(
        "-p",
        "--no-plot",
        dest="no_plot",
        action="store_true",
        default=False,
        help="if set then don't plot the training loss and evaluation results.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        help="if set then verbose is set to False during download.",
    )
    parser.add_argument(
        "-x",
        "--estimate",
        dest="estimate",
        action="store_true",
        help="if set then only perform training time estimation.",
    )
    parser.add_argument(
        "-f",
        "--freeze",
        dest="freeze",
        action="store_true",
        help="if set then only save all game configurations in DEFAULT_CONFIGS_PATH.",
    )
    args = parser.parse_args()

    if args.estimate:
        AlphaZeroTrainer.estimate_training_duration(args.game, args.json_config_file)
    elif args.freeze:
        freeze_config(game=args.game)
    else:
        trainer = AlphaZeroTrainer(verbose=not(args.quiet))
        trainer.train(
            game=args.game,
            json_config_file=args.json_config_file,
            experiment_name=args.experiment_name,
            plot=not(args.no_plot),
        )


if __name__ == "__main__":
    main()