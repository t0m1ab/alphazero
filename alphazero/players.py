import numpy as np
from time import sleep

from alphazero.base import Board
from alphazero.mcts import MCT


class Player():

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def reset(self, verbose: bool = False) -> None:
        """ Resets the internal state of the player. """
        pass

    def get_move(self, board: Board) -> int:
        """ Returns the best move for the player given the current board state. """
        pass

    def apply_move(self, move: int, player: int = None) -> None:
        """ Updates the internal state of the player after a move is played. """
        pass


class HumanPlayer(Player):

    def __init__(self) -> None:
        super().__init__()
    
    def reset(self, verbose: bool = False) -> None:
        if verbose:
            print("Hint: HumanPlayer moves should be entered in the format 'row col' or 'row,col' or 'row-col' (e.g. '3 4').")
    
    def __parse_input(self, input: str) -> tuple[int,int]:
        """ Parse the input string asked in get_move() to extract the move. """
        if len(input) != 3: # error
            return None
        else:
            row = input[0]
            sep = input[1]
            col = input[2]
            if sep not in [" ", ",", "-"]: # allowed separators
                return None
            if not col.isdigit() or not row.isdigit():
                return None
            return (int(row), int(col))
    
    def get_move(self, board: Board) -> int:

        move = None
        while move is None:
            
            move_input = input("Enter your move: ")

            if len(move_input) == 0: # pass
                move = board.pass_move
                if board.pass_move not in board.get_moves(): # can't pass when you can play
                    print("You cannot pass when you can play!")
                    move = None
            else:
                move = self.__parse_input(move_input)
                if move is not None and not board.is_legal_move(move): # illegal move
                    print("Illegal move!")
                    move = None
        
        return move


class RandomPlayer(Player):

    def __init__(self, lock_time: float = None) -> None:
        super().__init__()
        self.lock_time = lock_time # in seconds
    
    def get_move(self, board: Board) -> int:
        if self.lock_time is not None: # for display purposes for example
            sleep(self.lock_time)
        return board.get_random_move()


class MCTSPlayer(Player):
    """
    Player using Monte Carlo Tree Search to select the best move.
    MCTSPlayer doesn't have an internal representation of the board as an attribute.
    But it needs to always have its root node corresponding to the current state of the board.
    MCTSPlayer always assumes that it plays as player with id 1 but the board might be reversed.
    """

    def __init__(self, n_sim: int = None, compute_time: float = None) -> None:
        super().__init__()
        self.n_sim = n_sim
        self.compute_time = compute_time
        if self.n_sim is None and self.compute_time is None:
            raise ValueError("MCTSPlayer needs to have either n_sim or compute_time specified.")
        if self.n_sim is not None and self.compute_time is not None:
            raise ValueError("MCTSPlayer can't have both n_sim and compute_time specified.")
        self.mct = MCT()

    def reset(self, verbose: bool = False) -> None:
        self.mct = MCT()
    
    def apply_move(self, move: int, player: int = None) -> None:
        """ Maintain the root node synchronized with the current state of the board. """
        self.mct.change_root(move)
    
    def get_move(self, board: Board) -> int:
        """ 
        MCTSPlayer always assumes that it plays as player with id 1 but the board might be reversed. 
        So, it is required to adapt the board to the player's perspective using self.player to check the id of MCTSPlayer.
        """

        # perform MCTS
        self.mct.search(
            board=board, 
            n_sim=self.n_sim,
            compute_time=self.compute_time
        )

        verbose = True
        if verbose:
            n_rollouts, simulation_time = self.mct.get_stats()
            print(f"MCTSPlayer current score = {board.get_score()} | Number of rollouts = {n_rollouts} | time = {simulation_time:6f}")

        best_action = self.mct.get_best_action(board)

        return best_action # best_action = (row, col)


def main():

    _ = Player()
    print("Player created successfully!")

    _ = HumanPlayer()
    print("HumanPlayer created successfully!")

    _ = RandomPlayer()
    print("RandomPlayer created successfully!")

    _ = MCTSPlayer()
    print("MCTSPlayer created successfully!")


if __name__ == "__main__":
    main()