import numpy as np
from time import sleep

from alphazero.base import Board, Player
from alphazero.mcts import MCT


class HumanPlayer(Player):
    """
    Player asking the user to choose the move to play.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
    
    def __parse_input(self, input: str) -> tuple[int,int]:
        """ 
        Parse the input string asked in get_move() to extract the move. 
        HumanPlayer moves should be entered in the format 'row col' or 'row,col' or 'row-col' (e.g. '3 4').
        """
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
    
    def get_move(self, board: Board) -> tuple[int,int]:

        move = None
        while move is None: # UI loop
            
            move_input = input("\nEnter your move: ")

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
    """
    Player selecting a random legal move at each turn.
    """

    def __init__(self, lock_time: float = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.lock_time = lock_time # in seconds
    
    def clone(self) -> "RandomPlayer":
        """ Returns a deep copy of the player. """
        return RandomPlayer(
            lock_time=self.lock_time,
            verbose=self.verbose,
        )
    
    def get_move(self, board: Board) -> tuple[int,int]:
        if self.lock_time is not None: # for display purposes for example
            sleep(self.lock_time)
        return board.get_random_move()


class MCTSPlayer(Player):
    """
    Player using Monte Carlo Tree Search to select the best move.
    MCTSPlayer doesn't have an internal representation of the board as an attribute.
    But it needs to always have its root node synchronized to the current state of the board.
    """

    def __init__(self, n_sim: int = None, compute_time: float = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.n_sim = n_sim
        self.compute_time = compute_time
        if self.n_sim is None and self.compute_time is None:
            raise ValueError("MCTSPlayer needs to have either n_sim or compute_time specified.")
        if self.n_sim is not None and self.compute_time is not None:
            raise ValueError("MCTSPlayer can't have both n_sim and compute_time specified.")
        self.mct = MCT()
    
    def clone(self) -> "MCTSPlayer":
        """ Returns a deep copy of the player. """
        return MCTSPlayer(
            n_sim=self.n_sim, 
            compute_time=self.compute_time, 
            verbose=self.verbose,
        )

    def reset(self) -> None:
        self.mct = MCT()
    
    def apply_move(self, move, player: int = None) -> None:
        """ Maintain the root node synchronized with the current state of the board. """
        self.mct.change_root(move)
    
    def get_move(self, board: Board) -> tuple[int,int]:
        """ 
        Perform MCTS as long as the constraint (n_sim or compute_time) is not reached then select and return the best action.
        """

        if board.is_game_over():
            raise ValueError("MCTSPlayer.get_move was called with a board in game over state...")

        # perform MCTS
        self.mct.search(
            board=board, 
            n_sim=self.n_sim,
            compute_time=self.compute_time
        )

        # select best action
        best_action = self.mct.get_best_action(board) # (row, col) for Othello for example

        return best_action
    
    def get_stats_after_move(self) -> dict[str, int|float]:
        n_rollouts, simulation_time = self.mct.get_stats()
        return {"n_rollouts": n_rollouts, "time": simulation_time}


def main():

    _ = HumanPlayer()
    print("HumanPlayer created successfully!")

    _ = RandomPlayer()
    print("RandomPlayer created successfully!")

    _ = MCTSPlayer(n_sim=42)
    print("MCTSPlayer created successfully!")


if __name__ == "__main__":
    main()