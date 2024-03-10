import numpy as np
from time import sleep

from alphazero.base import Action, Board, Player, PolicyValueNetwork, MoveFormat
from alphazero.utils import fair_max
from alphazero.mcts import MCT
from alphazero.games.registers import MOVE_FORMATS_REGISTER


class HumanPlayer(Player):
    """
    Player asking the user to choose the move to play.
    """

    def __init__(self, game: str = None, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        if not game in MOVE_FORMATS_REGISTER: # self.move_format set to MoveFormat.ROW_COL by default
            self.move_format = MoveFormat.ROW_COL
        else:
            self.move_format = MOVE_FORMATS_REGISTER[game]
    
    def __parse_input(self, input: str) -> tuple[int,int]:
        """ 
        Parse the input string asked in get_move() to extract the move. 
        HumanPlayer moves should be entered in the format 'row col' or 'row,col' or 'row-col' (e.g. '3 4').
        """
        if self.move_format == MoveFormat.ROW_COL: # coordinates of the cell must be given
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
            
        elif self.move_format == MoveFormat.COL: # only the coordinate of the column must be given
            if not input.isdigit(): # error
                return None
            else:
                return int(input)
            
        elif self.move_format == MoveFormat.ROW: # only the coordinate of the row must be given
            if not input.isdigit(): # error
                return None
            else:
                return int(input)
            
        else:
            raise ValueError(f"Unknown move format {self.move_format}...")
    
    def get_move(self, board: Board, temp: float = None) -> Action:

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
    
    def get_move(self, board: Board, temp: float = None) -> Action:
        if self.lock_time is not None: # for display purposes for example
            sleep(self.lock_time)
        return board.get_random_move()
    

class GreedyPlayer(Player):
    """
    Player selecting the move that maximizes its score at each turn.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
    
    def clone(self) -> "RandomPlayer":
        """ Returns a deep copy of the player. """
        return GreedyPlayer(
            verbose=self.verbose,
        )
    
    def get_move(self, board: Board, temp: float = None) -> Action:
        """ Returns the move that maximizes the score of the board for the current player. """

        moves = board.get_moves()

        move2score = {}
        for move in moves:
            cloned_board = board.clone()
            cloned_board.play_move(move)
            move2score[move] = -cloned_board.get_score() # the score returned is from the viewpoint of the other player
        
        return fair_max(move2score.items(), key=lambda x: x[1])[0]


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
    
    def apply_move(self, move: Action, player: int = None) -> None:
        """ Maintain the root node synchronized with the current state of the board. """
        self.mct.change_root(move)
    
    def get_move(self, board: Board, temp: float = 0, return_action_probs: bool = False) -> Action:
        """ 
        Perform MCTS as long as the constraint (n_sim or compute_time) is not reached then select and return the best action.
        temp is the temperature parameter controlling the level of exploration of the MCTS (acts over the action probs)
        """

        if board.is_game_over():
            raise ValueError(f"{self}.get_move was called with a board in game over state...")

        # perform MCTS
        self.mct.search(
            board=board, 
            n_sim=self.n_sim,
            compute_time=self.compute_time
        )

        # select best action
        action_probs = self.mct.get_action_probs(board, temp)
        items = list(action_probs.items())
        if len(items) == 1: # only one action (if temp=0 for example)
            best_action = items[0][0]
        else: # need to sample an action according to the action probs
            best_action = items[np.random.choice(len(items), p=[prob for (_, prob) in items])][0]
        
        if return_action_probs:
            return best_action, action_probs
        else:
            return best_action
    
    def get_stats_after_move(self) -> dict[str, int|float]:
        n_rollouts, simulation_time = self.mct.get_stats()
        return {"n_rollouts": n_rollouts, "time": simulation_time}


class AlphaZeroPlayer(MCTSPlayer):
    """
    Player using MCTS with state evaluation done by a neural network to select the best move.
    """

    def __init__(
            self, 
            n_sim: int = None, 
            compute_time: float = None, 
            nn: PolicyValueNetwork = None,
            verbose: bool = False,
        ) -> None:
        super().__init__(n_sim=n_sim, compute_time=compute_time, verbose=verbose)
        self.mct = MCT(eval_method="neural", nn=nn) # nn maybe init to None but loaded/assigned later
    
    def clone(self) -> "AlphaZeroPlayer":
        """ Returns a deep copy of the player. """
        return AlphaZeroPlayer(
            n_sim=self.n_sim, 
            compute_time=self.compute_time, 
            nn=self.mct.nn.clone() if self.mct.nn is not None else None,
            verbose=self.verbose,
        )
    
    def reset(self) -> None:
        self.mct = MCT(eval_method="neural", nn=self.mct.nn)


def main():

    _ = HumanPlayer()
    print("HumanPlayer created successfully!")

    _ = RandomPlayer()
    print("RandomPlayer created successfully!")

    _ = MCTSPlayer(n_sim=42)
    print("MCTSPlayer created successfully!")

    _ = AlphaZeroPlayer(compute_time=0.2)
    print("AlphaZeroPlayer created successfully!")


if __name__ == "__main__":
    main()