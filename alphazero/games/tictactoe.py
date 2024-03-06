import os
from pathlib import Path
from aenum import Enum, NoAlias
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.base import Action, Board, PolicyValueNetwork, Config


@dataclass
class TicTacToeConfig(Config):
    """ Configuration for AlphaTicTacToeZero training. Any config file must define exactly all values listed below. """
    # GAME settings
    game: str = "tictactoe"
    # PLAYER settings
    simulations: int = 1000 # None to use compute_time # (100)
    compute_time: float = None # None to use simulations # (None)
    # TRAINING settings
    iterations: int = 2 # (30)
    episodes: int = 10 # (100)
    epochs: int = 5 # (10)
    batch_size: int = 64 # (64)
    learning_rate: float = 0.001 # (0.001)
    device: str = "cpu"
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
    COLORS = {-1: "white", 0: "green", 1: "black"}

    def __init__(
            self, 
            grid: np.ndarray = None, 
            player: int = 1,
            display_dir: str = None,
            config: Config = None,
        ):
        super().__init__(display_dir)

        if config is not None:
            self.__init_from_config(config)
        else:
            self.grid = grid if grid is not None else np.zeros((3,3))
            self.player = player

        self.game_name = "tictactoe"
    
    def reset(self) -> None:
        """ Resets the board to the initial state. """
        self.grid = np.zeros((3,3))
        self.player = 1
    
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
        
    def display(self, indexes: bool = True, filename: str = None) -> None:
        """ 
        Displays the board in a grid, each cell being filled with a circle if there is a piece on it.

        ARGUMENTS:
            - indexes: if True, the indexes of the rows and columns are displayed on the board.
            - filename: the name of the file in which the image of the board will be saved.
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
                        zorder=10
                    ))
        
        if indexes:
            eps = 0.15
            for row in range(3):
                ax.text(x=eps, y=3-row-eps, s=f"{row}", fontsize=10, ha='center', va='center', color="black")
            for col in range(1, 3):
                ax.text(x=col+eps, y=3-eps, s=f"{col}", fontsize=10, ha='center', va='center', color="black")
        
        # plt.grid(True, color="black", linewidth=1)
        # plt.axis('off') # unnecessary if we remove the frame as below

        # save the image
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) # remove the frame in the saved image
        Path(self.display_dir).mkdir(parents=True, exist_ok=True)
        filename = filename if filename is not None else "tictactoe_board.png"
        plt.savefig(os.path.join(self.display_dir, filename), bbox_inches=extent, dpi=150)
        plt.close()


class TicTacToeNet(PolicyValueNetwork):
    """
    Policy-Value network to play TicTacToe using AlphaZero algorithm.
    The network evaluates the state of the board from the viewpoint of the player with id 1 and outputs a value v in [-1,1]
    representing the probability of player with id 1 to win the game from the current state.
    The network also outputs a policy p representing the probability distribution of the next move to play.
    """

    CONFIG = TicTacToeConfig

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("TicTacToeNet is not implemented yet.")


def main():
    
    _ = TicTacToeBoard()
    print("TicTacToeBoard created successfully!")

    _ = TicTacToeNet()
    print("TicTacToeNet created successfully!")
    

if __name__ == "__main__":
    main()