import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from alphazero.base import Board


class OthelloBoard(Board):
    """
    Class representing the Othello board and implementing the logic of the game.
    The board is represented by a 2D numpy array of size n x n, where n is the size of the board.
    The cells are filled with 1 if there is a black piece, -1 if there is a white piece, and 0 if the cell is empty.
    """

    DIRECTIONS = [(1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1)]

    COLORS = {-1: "white", 0: "green", 1: "black"}

    def __init__(
            self, 
            n: int, 
            board: np.ndarray = None, 
            player: int = 1,
            display_dir: str = None,
        ):
        super().__init__(display_dir)

        if n % 2 != 0:
            raise ValueError("Board size must be even")
        self.n = n
        self.cells = board if board is not None else self.__get_init_board()
        self.player = player
        self.pass_move = (self.n, self.n) # pass is allowed in Othello only when a player has no legal move
        self.game_name = "othello"
    
    def __get_init_board(self) -> np.ndarray:
        """ Returns the initial board state as a 2D np.ndarray representing the content of each cell. """  
        cells = np.zeros((self.n, self.n))
        cells[self.n//2-1][self.n//2-1] = 1
        cells[self.n//2][self.n//2] = 1
        cells[self.n//2-1][self.n//2] = -1
        cells[self.n//2][self.n//2-1] = -1
        return cells
    
    def reset(self) -> None:
        """ Resets the board to the initial state. """
        self.cells = self.__get_init_board()
        self.player = 1
    
    def clone(self) -> "OthelloBoard":
        """ Returns a deep copy of the board. """
        return OthelloBoard(
            n=self.n, 
            board=self.cells.copy(), 
            player=self.player,
            display_dir=self.display_dir,
        )
    
    def get_board_size(self) -> tuple[int, int]:
        return (self.n, self.n)
    
    def get_action_size(self) -> int:
        """ Returns the number of possible moves in the game = number of cells + 1 (to pass)."""
        return self.n * self.n + 1
    
    def get_score(self) -> int:
        """ Returns the current score of the board from the viewpoint of self.player. """
        return np.sum(self.player * self.cells).astype(int)
    
    def __is_a_cell(self, cell: tuple[int, int]) -> bool:
        """ Returns True if the cell is in the board, False otherwise. """
        return 0 <= cell[0] < self.n and 0 <= cell[1] < self.n
    
    def __get_flips(self, cell: tuple[int,int], dir: tuple[int,int], player: int = None) -> list[tuple[int, int]]:
        """ Returns all the flips that would occur in the given direction <dir> if player <player> plays in <cell>. """
        player = player if player in [-1,1] else self.player
        flips = []
        c = (cell[0] + dir[0], cell[1] + dir[1])
        while self.__is_a_cell(c):
            if self.cells[c[0]][c[1]] == 0: # empty cell is breaking the sequence
                return []
            elif self.cells[c[0]][c[1]] == player: # sequence ends
                return flips
            flips.append(c) # sequence continues because the cell is occupied by the opponent
            c = (c[0] + dir[0], c[1] + dir[1])
        return []
    
    def is_legal_move(self, move: tuple[int, int], player: int = None) -> bool:
        """ Returns True if the move is legal, False otherwise (considering that it is a move for player <player>). """
        
        if move == self.pass_move: # must check all free positions to see if the player can pass
            for row in range(self.n):
                for col in range(self.n):
                    if self.cells[row][col] == 0:
                        for dir in OthelloBoard.DIRECTIONS:
                            if len(self.__get_flips((row,col), dir, player)) > 0:
                                return False
            return True
        
        if not self.__is_a_cell(move) or self.cells[move[0]][move[1]] != 0: # check if the cell is real and empty
            return False
        
        for dir in OthelloBoard.DIRECTIONS: # check if the move would flip at least one piece
            if len(self.__get_flips(move, dir, player)) > 0:
                return True
            
        return False
    
    def get_moves(self, player: int = None) -> list[tuple[int, int]]:
        """ 
        Returns all possible moves for player <player> which is self.player is <player> is None. 
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
    
    def get_random_move(self, player: int = None) -> tuple[int, int]:
        """ Returns a random move for player <player>. If no move is available, returns (self.n, self.n) to pass."""
        available_moves = self.get_moves(player)
        return available_moves[np.random.choice(len(available_moves))]
      
    def play_move(self, move: tuple[int, int]) -> None:
        """ Plays the move on the board. """
        
        if not self.is_legal_move(move):
            raise ValueError(f"Illegal move {move} for player {self.player}")
        
        if move == (self.n, self.n): # pass
            self.player = -self.player
            return
        
        for dir in OthelloBoard.DIRECTIONS: # flip the pieces in all directions if needed
            for cell in self.__get_flips(move, dir):
                self.cells[cell[0]][cell[1]] = self.player
        self.cells[move[0]][move[1]] = self.player # place the new piece on the board
        self.player = -self.player # switch player
    
    def is_game_over(self) -> bool:
        """ Returns True if the game is over, False otherwise. """
        player1_moves = self.get_moves(player=1)
        player2_moves = self.get_moves(player=-1)
        if (self.pass_move not in player1_moves) or (self.pass_move not in player2_moves):
            return False 
        else:
            return True

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
                if self.cells[row][col] != 0:
                    ax.add_patch(patches.Circle(
                        xy=(col+0.5, self.n - row - 0.5), # reverse the row index to match numpy array indexing
                        radius=0.4, 
                        color=OthelloBoard.COLORS[self.cells[row][col]], 
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


def random_play(n: int = 8, n_turns: int = 100, display_dir: str = None) -> None:
    """ Generate a random game of Othello and display the board at the end. """

    board = OthelloBoard(n=n, display_dir=display_dir)

    for turn_i in range(n_turns):
        player1_random_move = board.get_random_move()
        board.play_move(player1_random_move)
        player2_random_move = board.get_random_move()
        board.play_move(player2_random_move)
        print(f"#{turn_i} - Player 1 played {player1_random_move} | Player 2 played {player2_random_move}")
        if player1_random_move == (n,n) and player2_random_move == (n,n):
            print("> Both players passed, game over!")
            score = board.get_score()
            if score == 0:
                print("> Draw!")
            else:
                print(f"> Player {board.get_winner()} won the game with score: {score}")
            break
    
    board.display(indexes=True)


def main():
    
    _ = OthelloBoard(n=8)
    print("OthelloBoard created successfully!")
    # random_play()


if __name__ == "__main__":
    main()