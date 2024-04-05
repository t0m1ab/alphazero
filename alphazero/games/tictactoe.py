import os
from pathlib import Path
from aenum import Enum, NoAlias
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.base import Action, Board, PolicyValueNetwork, Config
from alphazero.utils import get_rgb_code


@dataclass
class TicTacToeConfig(Config):
    """ Configuration for AlphaTicTacToeZero training. Any config file must define exactly all values listed below. """
    # GAME settings
    game: str = "tictactoe"
    board_size: int = 3
    # PLAYER settings
    simulations: int = 100 # None to use compute_time # (100)
    compute_time: float = None # None to use simulations # (None)
    dirichlet_alpha: float = 0.03 # (0.03)
    dirichlet_epsilon: float = 0.25 # (0.25)
    temp_scheduler_type: str = "linear" # linear | constant | exponential
    temp_max_step: int = 2 # temperature = 1 until step temp_step_max in every game
    temp_min_step: int = 2 # temperature = 0 from step temp_step_min until the end of the game
    # TRAINING settings
    iterations: int = 30 # (30)
    episodes: int = 100 # (100)
    epochs: int = 10 # (10)
    batch_size: int = 64 # (64)
    learning_rate: float = 0.01 # linear decay with gamma=0.9
    data_augmentation: bool = True
    device: str = "cpu"
    # EVALUATION settings
    eval_opponent: str = "mcts" # random | greedy | mcts
    eval_episodes: int = 100
    do_eval: bool = True
    # SAVE settings
    save: bool = True
    push: bool = False
    save_checkpoints: bool = True
    push_checkpoints: bool = False


class TicTacToeBoard(Board):
    """
    Class representing the TicTacToe board and implementing the logic of the game.
    The board is represented by a 2D numpy array of size 3 x 3 stored in self.grid.
    The cells of the grid are filled with:
        * 1 if there is a X
        * -1 if there is a O
        * 0 if the cell is empty.
    """

    CONFIG = TicTacToeConfig
    DIRECTIONS = [(1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1)]
    COLORS = {-1: "white", 0: "silver", 1: "black"}
    TEXT_COLOR = "red"

    def __init__(
            self, 
            grid: np.ndarray = None, 
            player: int = 1,
            display_dir: str = None,
            display_mode: str = None,
            config: Config = None,
        ):
        super().__init__(display_dir=display_dir, display_mode=display_mode)

        self.game = "tictactoe"

        if config is not None:
            self.__init_from_config(config)
        else:
            self.grid = grid if grid is not None else np.zeros((3,3))
            self.player = player
            self.max_moves = 9
    
    def reset(self) -> None:
        """ Resets the board to the initial state. """
        self.grid = np.zeros((3,3))
        self.player = 1
        self.max_moves = 9
    
    def __init_from_config(self, config: Config) -> None:
        """ Initialize the TicTacToe board from a configuration given in a Config object. """
        self.reset()
    
    def clone(self) -> "TicTacToeBoard":
        """ Returns a deep copy of the board. """
        return TicTacToeBoard(
            grid=self.grid.copy(), 
            player=self.player,
            display_dir=self.display_dir,
        )
    
    def get_board_shape(self) -> tuple[int, int]:
        return self.grid.shape
    
    def get_n_cells(self) -> int:
        return np.prod(self.grid.shape)
    
    def get_action_size(self) -> int:
        """ Returns the number of possible moves in the game = number of cells (pass is not allowed)."""
        return self.get_n_cells()
    
    def get_alignments_sums(self) -> np.ndarray:
        """ Returns the sum of pieces along all possible alignments at the current state of the game. """
        row_sums = np.sum(self.grid, axis=1)
        col_sums = np.sum(self.grid, axis=0)
        diag_sums = [np.sum(np.diag(self.grid)), np.sum(np.diag(np.fliplr(self.grid)))]
        return np.concatenate([row_sums, col_sums, diag_sums])

    def get_nb_free_cells_on_alignments(self, player: int = None) -> np.ndarray:
        """ Returns the number of free cells on each alignment at the current state of the game. """
        free_cells = (self.grid == 0).astype(int)
        row_sums = np.sum(free_cells, axis=1)
        col_sums = np.sum(free_cells, axis=0)
        diag_sums = [np.sum(np.diag(free_cells)), np.sum(np.diag(np.fliplr(free_cells)))]
        return np.concatenate([row_sums, col_sums, diag_sums])
    
    def get_score(self) -> int:
        """ Returns the current score of the board from the viewpoint of self.player. """
        player_alignments = self.player * self.get_alignments_sums()
        free_cells_sums = (self.get_nb_free_cells_on_alignments() > 0).astype(int)
        if 2 in player_alignments * free_cells_sums: # there exists a winning position for the player
            return float("inf")
        return 0
    
    def __is_a_cell(self, cell: tuple[int, int]) -> bool:
        """ Returns True if the cell is in the board, False otherwise. """
        return 0 <= cell[0] < 3 and 0 <= cell[1] < 3
    
    def is_legal_move(self, move: Action, player: int = None) -> bool:
        """ Returns True if the move is legal, False otherwise (considering that it is a move for player <player>). """
        if self.__is_a_cell(move) and self.grid[move[0]][move[1]] == 0: # check if the cell is real and empty
            return True
        return False
    
    def get_moves(self, player: int = None) -> list[Action]:
        """ 
        Returns all possible moves for player <player> which is self.player if <player> is None. 
        If no move is available, returns an empty list.
        """
        return [(cell[0],cell[1]) for cell in np.vstack(np.where(self.grid == 0)).T]
    
    def get_random_move(self, player: int = None) -> Action:
        """ Returns a random move for player <player>. If no move is available, returns (self.n, self.n) to pass."""
        available_moves = self.get_moves(player)
        return available_moves[np.random.choice(len(available_moves))]
      
    def play_move(self, move: Action) -> None:
        """ Plays the move on the board. """
        
        if not self.is_legal_move(move):
            raise ValueError(f"Illegal move {move} for player {self.player}")
        
        self.grid[move[0]][move[1]] = self.player # place the new piece on the board
        self.player = -self.player # switch player
    
    def is_game_over(self) -> bool:
        """ Returns True if the game is over, False otherwise. """
        if 3 in np.abs(self.get_alignments_sums()): # a player has won
            return True
        else: # draw or game is not over
            return np.all(self.grid != 0)

    def get_winner(self) -> int:
        """ Returns the id of the winner of the game (1 or -1) or 0 if the game is a draw."""
        if not self.is_game_over():
            raise ValueError("Game is not over yet...")
        
        sums = self.get_alignments_sums()
        if 3 in sums: # player 1 has won
            return 1
        elif -3 in sums: # player -1 has won
            return -1
        else: # draw
            return 0
        
    def human_display(self, show_indexes: bool = True, infos: dict[dict] = None, filename: str = None) -> None:
        """ 
        Displays the board in a grid, each cell being filled with a circle if there is a piece on it.
        """

        # create the image of the board
        fig, ax = plt.subplots(1, figsize=(3, 3))
        fig.set_size_inches(3, 3)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.add_patch(patches.Rectangle((0, 0), 3, 3, color=TicTacToeBoard.COLORS[0]))
        
        for row in range(3):
            for col in range(3):
                if self.grid[row][col] != 0:
                    ax.add_patch(patches.Circle(
                        xy=(col+0.5, 3 - row - 0.5), # reverse the row index to match numpy array indexing
                        radius=0.4, 
                        color=TicTacToeBoard.COLORS[self.grid[row][col]], 
                        zorder=1
                    ))

        ax.set_xticks([0,1,2,3])
        ax.set_yticks([0,1,2,3])
        plt.grid(True, color="black", linewidth=1)
        
        if show_indexes:
            eps = 0.15
            for row in range(3):
                ax.text(x=eps, y=3-row-eps, s=f"{row}", fontsize=10, ha='center', va='center', color="black")
            for col in range(1, 3):
                ax.text(x=col+eps, y=3-eps, s=f"{col}", fontsize=10, ha='center', va='center', color="black")
        
        if infos is not None:
            for info_idx, (info_name, info_dict) in enumerate(infos.items()):
                for (row, col), value in info_dict.items():
                    ax.text(
                        x = col + 0.5, 
                        y = 3 - row - 0.45 - (float(info_idx)*0.15), 
                        s = f"{info_name}={value:.3f}" if isinstance(value, float) else f"{info_name}={value}",
                        fontsize = 5, 
                        fontweight = "bold",
                        ha = "center", 
                        color = TicTacToeBoard.TEXT_COLOR, 
                        zorder = 2,
                    )

        # save the image
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) # remove the frame in the saved image
        save_dir = os.path.join(self.display_dir, f"{self.game}_human")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = filename if filename is not None else f"{self.game}.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches=extent, dpi=300)
        plt.close()
    
    def pixel_display(self, show_indexes: bool = True, filename: str = None) -> None:
        """ 
        Displays the board in a grid, each cell being a pixel.
        """

        # create the RGB board
        board_rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        for row in range(3):
            for col in range(3):
                board_rgb[row][col] = get_rgb_code(TicTacToeBoard.COLORS[self.grid[row][col]])

        # convert to pixel image
        board_image = Image.fromarray(board_rgb, "RGB")

        # save the image
        save_dir = os.path.join(self.display_dir, f"{self.game}_pixel")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = filename if filename is not None else f"{self.game}.png"
        board_image.save(os.path.join(save_dir, filename))


class TicTacToeNet(PolicyValueNetwork):
    """
    Policy-Value network to play TicTacToe using AlphaZero algorithm.
    The network evaluates the state of the board from the viewpoint of the player with id 1 and outputs a value v in [-1,1]
    representing the probability of player with id 1 to win the game from the current state.
    The network also outputs a policy p representing the probability distribution of the next move to play.
    """

    CONFIG = TicTacToeConfig

    def __init__(
            self, 
            device: str = None, 
            config: Config = None
        ):
        """ If <config> is provided, the value of <n> will be automatically overwritten. """
        super().__init__()

        # parametrized values
        if config is not None:
            self.__init_from_config(config)
        else:
            self.device = PolicyValueNetwork.get_torch_device(device)

        # self.dropout = 0.3
        self.action_size = 9

        self.fc1 = nn.Linear(9, 9, device=self.device)
        self.fc2 = nn.Linear(9, 9, device=self.device)
        self.fc_probs = nn.Linear(9, self.action_size, device=self.device)
        self.fc_value = nn.Linear(9, 1, device=self.device)
        self.flatten = nn.Flatten()

        self.bn1 = nn.BatchNorm1d(9, device=self.device)
        self.bn2 = nn.BatchNorm1d(9, device=self.device)

    def __init_from_config(self, config: Config) -> None:
        """ Initialize the network from a config given in a Config object. """
        self.device = PolicyValueNetwork.get_torch_device(config.device) 

    def forward(self, input: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """ Forward through the network and outputs (logits of probabilitites, value). """

        if input.ndim == 2: # need to add batch dim to a single input 
            input = input.unsqueeze(0)

        x = self.flatten(input)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        probs = self.fc_probs(x) # batch_size x action_size
        value = self.fc_value(x) # batch_size x 1

        return F.log_softmax(probs, dim=1), torch.tanh(value)
            
    def get_normalized_probs(self, probs: np.ndarray, legal_moves: list[Action]) -> dict[Action, float]:
        """ Returns the normalized probabilities over the legal moves. """

        sum_legal_probs = 0
        legal_probs = {}
        for move in legal_moves:
            legal_probs[move] = probs[3 * move[0] + move[1]]
            sum_legal_probs += legal_probs[move]

        if sum_legal_probs < 1e-6: # set uniform probabilities if the sum is too close to 0
            print(f"The sum of the probabilities of the {len(legal_moves)} legal moves is {sum_legal_probs}")
            return {move: 1/len(legal_moves) for move in legal_moves}

        # normalize the probabilities to sum to 1
        norm_probs = {move: prob/sum_legal_probs for move, prob in legal_probs.items()}

        return norm_probs
    
    def to_neural_output(self, move_probs: dict[Action: float]) -> np.ndarray:
        """ Returns the probabilitites of move_probs in the format given as output by the network. """
        pi = np.zeros(9)
        for move, prob in move_probs.items():
            pi[3 * move[0] + move[1]] = prob
        return pi

    def reflect_neural_output(self, neural_output: np.ndarray, axis: int) -> np.ndarray:
        """
        Take a neural output and reflect it along the specified axis. 
        * axis = 0: reflect vertically
        * axis = 1: reflect horizontally
        """

        if not neural_output.size == self.action_size:
            raise ValueError(f"Neural output should have size {self.action_size}, but has size {neural_output.size}")

        return np.flip(neural_output.reshape((3,3)), axis=axis).flatten()
    
    def rotate_neural_output(self, neural_output: np.ndarray, angle: int) -> np.ndarray:
        """ 
        Take a neural output and rotate it with <d90> successive 90° counterclockwise rotations. 
        * d90 = 1 -> 90° rotation
        * d90 = 2 -> 180° rotation
        * d90 = 3 -> 270° rotation
        * d90 = 4 -> 360° rotation (identity)
        """

        if not neural_output.size == self.action_size:
            raise ValueError(f"Neural output should have size {self.action_size}, but has size {neural_output.size}")

        return np.rot90(neural_output.reshape((3,3)), k=angle//90).flatten()


def main():
    
    _ = TicTacToeBoard()
    print("TicTacToeBoard created successfully!")

    _ = TicTacToeNet()
    print("TicTacToeNet created successfully!")
    

if __name__ == "__main__":
    main()