import numpy as np
from time import time
from tqdm import tqdm
import torch

from alphazero.base import Config
from alphazero.players import AlphaZeroPlayer
from alphazero.games.registers import CONFIGS_REGISTER, BOARDS_REGISTER, NETWORKS_REGISTER


class SelfPlayTimer():
    """
    Timer to measure the average time it takes to complete a self-play game.
    Neglect data augmentation time as it is often very fast compared to the game simulations.
    """

    def __init__(self, game: str, config: Config = None):

        # load default config
        self.config = CONFIGS_REGISTER[game]() if config is None else config

        # define Board, PolicyValueNetwork and Player
        self.board = BOARDS_REGISTER[game](config=self.config)
        self.nn = NETWORKS_REGISTER[game](config=self.config, device=self.config.device)
        self.az_player = AlphaZeroPlayer(
            n_sim=self.config.simulations, 
            compute_time=self.config.compute_time, 
            nn=self.nn, 
            verbose=False,
        )

    def self_play(self) -> tuple[float, int]:
        """ Simulate a self-play game and returns the time it took to complete and the number of steps. """

        # initialize the board and the players
        self.board.reset()
        self.az_player.reset() # reset the MCT
        
        # self-play
        n_moves = 0
        start_time = time()
        while not self.board.is_game_over():
            # get best move for the current player
            move, _ = self.az_player.get_move(self.board, return_action_probs=True)
            # play the move on the board
            self.board.play_move(move)
            # update internal state of the player
            self.az_player.apply_move(move, player=-self.board.player)
            n_moves += 1
        
        return time() - start_time, n_moves
    
    def timeit(self, n_episodes: int = None) -> tuple[float, float]:
        """
        Run multiple self-play games and return their average duration.
        """

        time_duration = []
        steps_duration = []
        pbar = tqdm(
            range(n_episodes if n_episodes is not None else self.config.n_episodes), 
            desc=f"Timing self-play {self.az_player}"
        )
        for _ in pbar:
            game_time, game_steps = self.self_play()
            time_duration.append(game_time)
            steps_duration.append(game_steps)
            pbar.set_postfix({"time": f"{game_time:.2f}s", "steps": game_steps})
        
        mean_time = np.mean(time_duration)
        std_time = np.std(time_duration)
        mean_steps = np.mean(steps_duration)
        std_steps = np.std(steps_duration)
        print(f"Average time to complete a self-play game: {mean_time:.2f} (+-{std_time:.2f}) seconds | {mean_steps:.2f} (+-{std_steps:.2f}) steps")

        return mean_time, mean_steps


class NeuralTimer():
    """
    Timer to measure the average time it takes to perform optimization of a neural network (backprop + optimizer).
    """

    def __init__(self, game: str, config: Config = None):

        # load default config
        self.config = CONFIGS_REGISTER[game]() if config is None else config

        # define Board, PolicyValueNetwork and Player
        self.board = BOARDS_REGISTER[game](config=self.config)
        self.nn = NETWORKS_REGISTER[game](config=self.config, device=self.config.device)
    
    def get_fake_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Create a fake batch of data for the neural network. """

        if hasattr(self.config, "board_size"):
            input = torch.randn(
                self.config.batch_size, 
                self.config.board_size, 
                self.config.board_size
            ).to(self.config.device)
        elif hasattr(self.config, "board_width") and hasattr(self.config, "board_height"):
            input = torch.randn(
                self.config.batch_size, 
                self.config.board_height,
                self.config.board_width
            ).to(self.config.device)
        else:
            raise AttributeError("Board size/width/height not found in the config...")

        pi = torch.randn(
            self.config.batch_size, 
            self.board.get_action_size()
        ).to(self.config.device)

        v = torch.randn(self.config.batch_size).to(self.config.device)

        return input, pi, v
    
    def timeit(self, n_batches: int = None) -> float:
        """ Run multiple optimization step for the neural network and return their average duration. """

        time_duration = []
        pbar = tqdm(
            range(n_batches if n_batches is not None else self.config.n_batches), 
            desc=f"Timing optimization {self.nn}"
        )

        optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.config.learning_rate)

        for _ in pbar:
            
            # create fake batch data
            input, pi, z = self.get_fake_batch()

            start_time = time()
            
            # optimization logic
            optimizer.zero_grad()
            log_probs, v = self.nn(input)
            loss = torch.sum((v - z) ** 2) - torch.sum(pi * log_probs)
            loss.backward()
            optimizer.step()

            time_duration.append(time() - start_time)
        
        mean_time = np.mean(time_duration)
        std_time = np.std(time_duration)
        print(f"Average time to optimize with a batch: {mean_time:.2f} (+-{std_time:.2f}) seconds")

        return mean_time


def demo():

    # create a SelfPlayTimer for Othello
    spt = SelfPlayTimer(game="othello")
    _ = spt.timeit(n_episodes=2)

    # create a NeuralTimer for Othello
    nt = NeuralTimer(game="othello")
    _ = nt.timeit(n_batches=10)


def main():
    
    _ = SelfPlayTimer(game="othello")
    print("SelfPlayTimer created successfully!")

    _ = NeuralTimer(game="othello")
    print("NeuralTimer created successfully!")


if __name__ == "__main__":
    demo()
