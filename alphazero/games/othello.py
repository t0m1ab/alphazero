import os
from pathlib import Path
from aenum import Enum, NoAlias
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.base import Action, Board, PolicyValueNetwork, Config
from alphazero.utils import dotdict


class OthelloConfig(Config):
    """ Configuration for AlphaOthelloZero training. Any config file must define exactly all values listed below. """
    # GAME settings
    GAME: str = "othello"
    BOARD_SIZE: int = 6 # (6)
    # PLAYER settings
    SIMULATIONS: int = 100 # None to use compute_time # (100)
    COMPUTE_TIME: float = None # None to use n_sim # (None)
    # TRAINING settings
    ITERATIONS: int = 2 # (30)
    EPISODES: int = 10 # (100)
    EPOCHS: int = 5 # (10)
    BATCH_SIZE: int = 64 # (64)
    LEARNING_RATE: float = 0.001 # (0.001)
    DEVICE:str = "cpu"


class OthelloBoard(Board):
    """
    Class representing the Othello board and implementing the logic of the game.
    The board is represented by a 2D numpy array of size n x n stored in self.grid (n is the size of the board).
    The cells of the grid are filled with:
        * 1 if there is a black piece
        * -1 if there is a white piece
        * 0 if the cell is empty.
    """

    DIRECTIONS = [(1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1)]
    
    COLORS = {-1: "white", 0: "green", 1: "black"}

    def __init__(
            self, 
            n: int = None, 
            grid: np.ndarray = None, 
            player: int = 1,
            display_dir: str = None,
            config_dict: dotdict = None,
        ):
        super().__init__(display_dir)

        if config_dict is not None:
            self.__init_from_config(config_dict)
        else:
            self.n = n
            self.grid = grid if grid is not None else self.__get_init_board()
            self.player = player

        if self.n % 2 != 0:
            raise ValueError(f"Board size must be even but got n={self.n}")
        self.pass_move = self.get_board_shape() # pass is allowed in Othello only when a player has no legal move
        self.game_name = "othello"
    
    def __init_from_config(self, config_dict: dotdict) -> None:
        """ Initialize the Othello board from a configuration given in a dotdict. """
        self.n = config_dict.board_size
        self.grid = self.__get_init_board()
        self.player = 1
    
    def __get_init_board(self) -> np.ndarray:
        """ Returns the initial board state as a 2D np.ndarray representing the content of each cell. """  
        grid = np.zeros((self.n, self.n))
        grid[self.n//2-1][self.n//2-1] = 1
        grid[self.n//2][self.n//2] = 1
        grid[self.n//2-1][self.n//2] = -1
        grid[self.n//2][self.n//2-1] = -1
        return grid
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.n}"
    
    def reset(self) -> None:
        """ Resets the board to the initial state. """
        self.grid = self.__get_init_board()
        self.player = 1
    
    def clone(self) -> "OthelloBoard":
        """ Returns a deep copy of the board. """
        return OthelloBoard(
            n=self.n, 
            grid=self.grid.copy(), 
            player=self.player,
            display_dir=self.display_dir,
        )
    
    def get_board_shape(self) -> tuple[int, int]:
        return self.grid.shape
    
    def get_n_cells(self) -> int:
        return np.prod(self.get_board_shape())
    
    def get_action_size(self) -> int:
        """ Returns the number of possible moves in the game = number of cells + 1 (to pass)."""
        return self.n * self.n + 1
    
    def get_score(self) -> int:
        """ Returns the current score of the board from the viewpoint of self.player. """
        return np.sum(self.player * self.grid).astype(int)
    
    def __is_a_cell(self, cell: tuple[int, int]) -> bool:
        """ Returns True if the cell is in the board, False otherwise. """
        return 0 <= cell[0] < self.n and 0 <= cell[1] < self.n
    
    def __get_flips(self, cell: tuple[int,int], dir: tuple[int,int], player: int = None) -> list[tuple[int, int]]:
        """ Returns all the flips that would occur in the given direction <dir> if player <player> plays in <cell>. """
        player = player if player in [-1,1] else self.player
        flips = []
        c = (cell[0] + dir[0], cell[1] + dir[1])
        while self.__is_a_cell(c):
            if self.grid[c[0]][c[1]] == 0: # empty cell is breaking the sequence
                return []
            elif self.grid[c[0]][c[1]] == player: # sequence ends
                return flips
            flips.append(c) # sequence continues because the cell is occupied by the opponent
            c = (c[0] + dir[0], c[1] + dir[1])
        return []
    
    def is_legal_move(self, move: Action, player: int = None) -> bool:
        """ Returns True if the move is legal, False otherwise (considering that it is a move for player <player>). """
        
        if move == self.pass_move: # must check all free positions to see if the player can pass
            for row in range(self.n):
                for col in range(self.n):
                    if self.grid[row][col] == 0:
                        for dir in OthelloBoard.DIRECTIONS:
                            if len(self.__get_flips((row,col), dir, player)) > 0:
                                return False
            return True
        
        if not self.__is_a_cell(move) or self.grid[move[0]][move[1]] != 0: # check if the cell is real and empty
            return False
        
        for dir in OthelloBoard.DIRECTIONS: # check if the move would flip at least one piece
            if len(self.__get_flips(move, dir, player)) > 0:
                return True
            
        return False
    
    def get_moves(self, player: int = None) -> list[Action]:
        """ 
        Returns all possible moves for player <player> which is self.player if <player> is None. 
        If no move is available, returns [self.pass_move].
        Move self.pass_move is included in <moves> only when no other move is available to <player>.
        """
        moves = set()
        for row in range(self.n):
            for col in range(self.n):
                if self.is_legal_move((row,col), player):
                    moves.add((row, col))
        if len(moves) == 0: # player can only pass if no move is available
            moves.add(self.pass_move)
        return list(moves)
    
    def get_random_move(self, player: int = None) -> Action:
        """ Returns a random move for player <player>. """
        available_moves = self.get_moves(player)
        return available_moves[np.random.choice(len(available_moves))]
      
    def play_move(self, move: Action) -> None:
        """ Plays the move on the board. """
        
        if not self.is_legal_move(move):
            raise ValueError(f"Illegal move {move} for player {self.player}")
        
        if move == (self.n, self.n): # pass
            self.player = -self.player
            return
        
        for dir in OthelloBoard.DIRECTIONS: # flip the pieces in all directions if needed
            for cell in self.__get_flips(move, dir):
                self.grid[cell[0]][cell[1]] = self.player
        self.grid[move[0]][move[1]] = self.player # place the new piece on the board
        self.player = -self.player # switch player
    
    def is_game_over(self) -> bool:
        """ Returns True if the game is over, False otherwise. """
        player1_moves = self.get_moves(player=1)
        player2_moves = self.get_moves(player=-1)
        if (self.pass_move in player1_moves) and (self.pass_move in player2_moves):
            return True 
        else:
            return False

    def get_winner(self) -> int:
        """ Returns the id of the winner of the game (1 or -1) or 0 if the game is a draw."""
        if not self.is_game_over():
            raise ValueError("Game is not over yet...")
        score = self.get_score()
        if score == 0: # draw
            return 0
        else: # return the id of the player who has the positive score
            return self.player if score > 0 else -self.player
        
    def display(self, indexes: bool = True, filename: str = None) -> None:
        """ 
        Displays the board in a grid, each cell being filled with a circle if there is a piece on it.

        ARGUMENTS:
            - indexes: if True, the indexes of the rows and columns are displayed on the board.
            - filename: the name of the file in which the image of the board will be saved.
        """

        # create the image of the board
        fig, ax = plt.subplots(1, figsize=(self.n, self.n))
        fig.set_size_inches(self.n, self.n)
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, self.n)
        ax.add_patch(patches.Rectangle((0, 0), self.n, self.n, color=OthelloBoard.COLORS[0]))
        
        for row in range(self.n):
            for col in range(self.n):
                if self.grid[row][col] != 0:
                    ax.add_patch(patches.Circle(
                        xy=(col+0.5, self.n - row - 0.5), # reverse the row index to match numpy array indexing
                        radius=0.4, 
                        color=OthelloBoard.COLORS[self.grid[row][col]], 
                        zorder=10
                    ))
        
        if indexes:
            eps = 0.15
            for row in range(self.n):
                ax.text(x=eps, y=self.n-row-eps, s=f"{row}", fontsize=10, ha='center', va='center', color="black")
            for col in range(1, self.n):
                ax.text(x=col+eps, y=self.n-eps, s=f"{col}", fontsize=10, ha='center', va='center', color="black")
        
        plt.grid(True, color="black", linewidth=1)
        # plt.axis('off') # unnecessary if we remove the frame as below

        # save the image
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) # remove the frame in the saved image
        Path(self.display_dir).mkdir(parents=True, exist_ok=True)
        filename = filename if filename is not None else "othello_board.png"
        plt.savefig(os.path.join(self.display_dir, filename), bbox_inches=extent, dpi=150)
        plt.close()


class OthelloNet(PolicyValueNetwork):
    """
    Policy-Value network to play Othello using AlphaZero algorithm.
    The network evaluates the state of the board from the viewpoint of the player with id 1 and outputs a value v in [-1,1]
    representing the probability of player with id 1 to win the game from the current state.
    The network also outputs a policy p representing the probability distribution of the next move to play.
    """

    def __init__(
            self, 
            n: int = None, 
            device: str = None, 
            config_dict: dotdict = None
        ):
        """ If <config_dict> is provided, the value of <n> will be automatically overwritten. """
        super().__init__()

        # parametrized values
        if config_dict is not None:
            self.__init_from_config(config_dict)
        else:
            if n is None:
                raise ValueError("The board size must be a positive and even integer like 4, 6 or 8.")
            self.n = n
            self.device = PolicyValueNetwork.get_torch_device(device)
        
        # fixed values for the network's architecture
        self.action_size = self.n * self.n + 1
        self.n_channels = 32
        self.dropout = 0.3

        # layers
        self.conv1 = nn.Conv2d(1, self.n_channels, 3, stride=1, padding=1, device=self.device)
        self.conv2 = nn.Conv2d(self.n_channels, self.n_channels, 3, stride=1, padding=1, device=self.device)
        self.conv3 = nn.Conv2d(self.n_channels, self.n_channels, 3, stride=1, device=self.device)
        self.conv4 = nn.Conv2d(self.n_channels, self.n_channels, 3, stride=1, device=self.device)

        self.bn1 = nn.BatchNorm2d(self.n_channels, device=self.device)
        self.bn2 = nn.BatchNorm2d(self.n_channels, device=self.device)
        self.bn3 = nn.BatchNorm2d(self.n_channels, device=self.device)
        self.bn4 = nn.BatchNorm2d(self.n_channels, device=self.device)

        self.fc1_input_size = self.n_channels * (self.n-4) * (self.n-4)
        self.fc1 = nn.Linear(self.fc1_input_size, 1024, device=self.device)
        self.fc_bn1 = nn.BatchNorm1d(1024, device=self.device)

        self.fc2 = nn.Linear(1024, 512, device=self.device)
        self.fc_bn2 = nn.BatchNorm1d(512, device=self.device)

        self.fc_probs = nn.Linear(512, self.action_size, device=self.device)
        self.fc_value = nn.Linear(512, 1, device=self.device) 

    def __init_from_config(self, config_dict: dotdict) -> None:
        """ Initialize the network from a config given in a dotdict. """
        self.n = config_dict.board_size
        self.device = PolicyValueNetwork.get_torch_device(config_dict.device) 

    def forward(self, input: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """ Forward through the network and outputs (logits of probabilitites, value). """
        # x: batch_size x board_x x board_y
        x = input.view(-1, 1, self.n, self.n) # batch_size x 1 x n x n
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, self.fc1_input_size)

        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=self.dropout, training=self.training) # batch_size x 1024
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=self.dropout, training=self.training) # batch_size x 512

        probs = self.fc_probs(x) # batch_size x action_size
        value = self.fc_value(x) # batch_size x 1

        return F.log_softmax(probs, dim=1), torch.tanh(value)
    
    def predict(self, input: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """ Returns a policy and a value from the input state. """
        self.eval()
        with torch.no_grad():
            log_probs, v =  self.forward(input)
        return torch.exp(log_probs), v

    def evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        """ 
        Evaluation of the state of the cloned board from the viewpoint of the player that needs to play. 
        A PolicyValueNetwork always evaluates the board from the viewpoint of player with id 1.
        Therefore, the board should be switched if necessary.
        """
        input = torch.tensor(board.player * board.grid, dtype=torch.float, device=self.device)
        torch_probs, torch_v = self.predict(input)
        probs = torch_probs.cpu().numpy().reshape(-1)
        return probs, torch_v.cpu().item()
    
    def get_normalized_probs(self, probs: np.ndarray, legal_moves: list[Action]) -> dict[Action, float]:
        """ Returns the normalized probabilities over the legal moves. """

        sum_legal_probs = 0
        legal_probs = {}
        for move in legal_moves:
            # needs to make the difference between the pass move and the other moves
            prob = probs[move[0] * self.n + move[1]] if move != (self.n, self.n) else probs[-1]
            legal_probs[move] = prob
            sum_legal_probs += prob

        if sum_legal_probs < 1e-6: # set uniform probabilities if the sum is too close to 0
            print(f"The sum of the probabilities of the {len(legal_moves)} legal moves is {sum_legal_probs}")
            return {move: 1/len(legal_moves) for move in legal_moves}

        # normalize the probabilities to sum to 1
        norm_probs = {move: prob/sum_legal_probs for move, prob in legal_probs.items()}

        return norm_probs
    
    def to_neural_array(self, move_probs: dict[Action: float]) -> np.ndarray:
        """ Returns the probabilitites of move_probs in the format given as output by the network. """
        pi = np.zeros(self.action_size)
        for move, prob in move_probs.items():
            if move == (self.n, self.n): # pass move
                pi[-1] = prob
            else:
                pi[move[0] * self.n + move[1]] = prob
        return pi


def main():
    
    _ = OthelloBoard(n=8)
    print("OthelloBoard created successfully!")

    _ = OthelloNet(n=8)
    print("OthelloNet created successfully!")
    

if __name__ == "__main__":
    main()