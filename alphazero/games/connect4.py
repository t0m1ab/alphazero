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
class Connect4Config(Config):
    """ Configuration for AlphaConnect4Zero training. Any config file must define exactly all values listed below. """
    # GAME settings
    game: str = "connect4"
    board_width: int = 8 # (8)
    board_height: int = 6 # (6)
    # PLAYER settings
    simulations: int = 50 # None to use compute_time # (100)
    compute_time: float = None # None to use simulations # (None)
    # TRAINING settings
    iterations: int = 2 # (30)
    episodes: int = 10 # (100)
    epochs: int = 5 # (10)
    batch_size: int = 64 # (64)
    learning_rate: float = 0.001 # (0.001)
    device: str = "cpu"
    # SAVE settings
    save: bool = False
    push: bool = False
    save_checkpoints: bool = True
    push_checkpoints: bool = False


class Connect4Board(Board):
    """
    Class representing the Connect4 board and implementing the logic of the game.
    The board is represented by a 2D numpy array of size n x n stored in self.grid (n is the size of the board).
    The cells of the grid are filled with:
        * 1 if there is a black piece
        * -1 if there is a white piece
        * 0 if the cell is empty.
    """

    CONFIG = Connect4Config
    DIRECTIONS = [(1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1)]
    COLORS = {-1: "white", 0: "green", 1: "black"}

    def __init__(
            self, 
            width: int,
            height: int,
            grid: np.ndarray = None, 
            player: int = 1,
            display_dir: str = None,
            config: Config = None,
        ):
        super().__init__(display_dir)

        self.game = "connect4"

        if config is not None:
            self.__init_from_config(config)
        else:
            self.width = width
            self.height = height
            self.grid = grid if grid is not None else self.__get_init_board()
            self.free_rows = self.__count_free_rows() # count the number of free cells in each column
            self.player = player

        if self.width < 4 or self.height < 4:
            raise ValueError(f"Borad size must be at least 4x4, got {self.width}x{self.height}")
    
    def reset(self) -> None:
        """ Resets the board to the initial state. """
        self.grid = self.__get_init_board()
        self.free_rows = self.__count_free_rows()
        self.player = 1

    def __init_from_config(self, config: Config) -> None:
        """ Initialize the Connect4 board from a configuration given in a Config object. """
        self.width = config.board_width
        self.height = config.board_height
        self.reset()
    
    def __get_init_board(self) -> np.ndarray:
        """ Returns the initial board state as a 2D np.ndarray representing the content of each cell. """  
        return np.zeros((self.height, self.width))
    
    def __count_free_rows(self) -> np.ndarray:
        """ Returns the number of free cells in each column. """
        return np.sum(self.grid == 0, axis=0)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.width}x{self.height}"
        
    def clone(self) -> "Connect4Board":
        """ Returns a deep copy of the board. """
        return Connect4Board(
            width=self.width,
            height=self.height, 
            grid=self.grid.copy(), 
            player=self.player,
            display_dir=self.display_dir,
        )
    
    def get_board_shape(self) -> tuple[int, int]:
        return self.grid.shape
    
    def get_n_cells(self) -> int:
        return np.prod(self.get_board_shape())
    
    def get_action_size(self) -> int:
        """ Returns the number of possible moves in the game = number of cells (pass is not allowed)."""
        return self.n * self.n
    
    def get_score(self) -> int:
        """ Returns the current score of the board from the viewpoint of self.player. """
        return np.sum(self.player * self.grid).astype(int)
        
    def __is_a_cell(self, cell: tuple[int, int]) -> bool:
        """ Returns True if the cell is in the board, False otherwise. """
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width
    
    def __is_a_playable_column(self, col: int) -> bool:
        """ Returns True if the column is not full, False otherwise. """
        if not 0 <= col < self.width:
            raise ValueError(f"Column index must be in [0, {self.width-1}], got {col}")
        return self.free_rows[col] > 0
    
    def is_legal_move(self, move: Action, player: int = None) -> bool:
        """ Returns True if the move is legal, False otherwise (considering that it is a move for player <player>). """
        if self.__is_a_playable_column(move):
            return True
        return False
    
    def get_moves(self, player: int = None) -> list[Action]:
        """ 
        Returns all possible moves for player <player> which is self.player if <player> is None. 
        If no move is available, returns an empty list.
        """
        return list(np.where(self.free_rows > 0)[0])
    
    def get_random_move(self, player: int = None) -> Action:
        """ Returns a random move for player <player>. """
        # print(self.grid, self.player, player)
        available_moves = self.get_moves(player)
        return available_moves[np.random.choice(len(available_moves))]
      
    def play_move(self, move: Action) -> None:
        """ Plays the move on the board. """
        
        if not self.is_legal_move(move):
            raise ValueError(f"Illegal move {move} for player {self.player}")
                
        row_idx = self.free_rows[move] - 1 # get the row index of the new piece if it is placed in the column
        if self.grid[row_idx][move] != 0:
            raise ValueError(f"Cell ({row_idx},{move}) is not empty...")
        self.grid[row_idx][move] = self.player # place the new piece on the board
        self.free_rows[move] -= 1 # update the number of free cells in the column
        self.player = -self.player # switch player
    
    def __check_alignment(self, cell: tuple[int, int], direction: tuple[int, int], grid: np.ndarray) -> int:
        """ 
        Check alignment from cell in the given direction in the grid.
        Returns 1 if there is a 4-alignment of 1, -1 if there is a 4-alignment of -1, 0 otherwise.
        """
        pos_sum = 0
        neg_sum = 0
        while self.__is_a_cell(cell):
            
            # update the sums for both players
            if grid[cell[0]][cell[1]] == 1:
                pos_sum += 1
                neg_sum = 0
            elif grid[cell[0]][cell[1]] == -1:
                pos_sum = 0
                neg_sum += 1
            else:
                pos_sum = 0
                neg_sum = 0

            # stop if there is a 4-alignment
            if pos_sum == 4:
                return 1
            elif neg_sum == 4:
                return -1
            
            # move to the next cell
            cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        return 0

    def __get_game_status(self) -> int | None:
        """ Returns the id of the winner of the game (1 or -1) or 0 if the game is a draw or None is the game is not over. """

        start_diags = [(i, 0) for i in range(1, self.height-3)] + [(0, i) for i in range(self.width-3)]
        start_rows = [(i, 0) for i in range(self.height)]
        start_cols = [(0, i) for i in range(self.width)]

        for start_cell in start_diags: # check top-left to bottom-right diagonals
            status = self.__check_alignment(cell=start_cell, direction=(1,1), grid=self.grid)
            if status != 0:
                return status
        
        for start_cell in start_diags: # check bottom-left to top-right diagonals
            status = self.__check_alignment(cell=start_cell, direction=(1,1), grid=np.fliplr(self.grid))
            if status != 0:
                return status
        
        for start_cell in start_rows: # check rows
            status = self.__check_alignment(cell=start_cell, direction=(0,1), grid=self.grid)
            if status != 0:
                return status
        
        for start_cell in start_cols: # check columns
            status = self.__check_alignment(cell=start_cell, direction=(1,0), grid=self.grid)
            if status != 0:
                return status

        if np.sum(self.free_rows) == 0: # draw
            return 0
        else: # game is not over because there is no 4-alignment for any player and there are still free cells
            return None

    def is_game_over(self) -> bool:
        """ Returns True if the game is over, False otherwise. """
        return self.__get_game_status() is not None

    def get_winner(self) -> int:
        """ Returns the id of the winner of the game (1 or -1) or 0 if the game is a draw. """
        game_status = self.__get_game_status()
        if game_status is None:
            raise ValueError("Game is not over yet...")
        elif game_status == 0: # draw
            return 0
        else: # return the id of the player who has the positive score (game_status is either 1 or -1 at this stage)
            return game_status
        
    def display(self, indexes: bool = True, filename: str = None) -> None:
        """ 
        Displays the board in a grid, each cell being filled with a circle if there is a piece on it.

        ARGUMENTS:
            - indexes: if True, the indexes of the rows and columns are displayed on the board.
            - filename: the name of the file in which the image of the board will be saved.
        """

        # create the image of the board
        fig, ax = plt.subplots(1, figsize=(self.width, self.height))
        fig.set_size_inches(self.width, self.height)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.add_patch(patches.Rectangle((0, 0), self.width, self.height, color=Connect4Board.COLORS[0]))
        
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row][col] != 0:
                    ax.add_patch(patches.Circle(
                        xy=(col+0.5, self.height - row - 0.5), # reverse the row index to match numpy array indexing
                        radius=0.4, 
                        color=Connect4Board.COLORS[self.grid[row][col]], 
                        zorder=10
                    ))
        
        if indexes:
            eps = 0.15
            # for row in range(self.height):
            #     ax.text(x=eps, y=self.height-row-eps, s=f"{row}", fontsize=10, ha='center', va='center', color="black")
            for col in range(self.width):
                ax.text(x=col+eps, y=self.height-eps, s=f"{col}", fontsize=10, ha='center', va='center', color="black")
        
        plt.grid(True, color="black", linewidth=1)
        # plt.axis('off') # unnecessary if we remove the frame as below

        # save the image
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) # remove the frame in the saved image
        Path(self.display_dir).mkdir(parents=True, exist_ok=True)
        filename = filename if filename is not None else f"{self.game}.png"
        plt.savefig(os.path.join(self.display_dir, filename), bbox_inches=extent, dpi=150)
        plt.close()


class Connect4Net(PolicyValueNetwork):
    """
    Policy-Value network to play Connect4 using AlphaZero algorithm.
    The network evaluates the state of the board from the viewpoint of the player with id 1 and outputs a value v in [-1,1]
    representing the probability of player with id 1 to win the game from the current state.
    The network also outputs a policy p representing the probability distribution of the next move to play.
    """

    CONFIG = Connect4Config

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Connect4Net is not implemented yet.")


def main():
    
    _ = Connect4Board(width=8, height=6)
    print("Connect4Board created successfully!")
    

if __name__ == "__main__":
    main()