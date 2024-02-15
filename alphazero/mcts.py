import numpy as np
from time import time

from alphazero.utils import fair_max
from alphazero.base import Board


class Node:
    """
    Object which represents a state of the game (= board configuration)
    Each node can be associated with a state of the game (board representation).
    Nodes don't directly contain this information as an attribute. 
    That's why the state of the board must be maintained in parallel during select and rollout.
    """

    def __init__(self, move: tuple[int,int], parent) -> None:
        self.move = move # move that led to this node
        self.parent = parent # parent node
        self.N = 0 # number of visits of the node from the creation of the tree
        self.Q = 0 # expected reward of the node (meaning of the action that led to the node)
        self.children = {} # {move1: NodeChild1, move2: NodeChild2, ...}
    
    def __str__(self) -> str:
        n_siblings = len(self.parent.children) if self.parent is not None else 0
        return f"Node: N={self.N} | Q={self.Q} | n_children={len(self.children)} | move={self.move} | n_siblings={n_siblings}"
    
    def add_child(self, move: tuple[int,int]) -> None:
        self.children[move] = Node(move=move, parent=self)

    def UCB(self) -> float:
        coeff_exploration = np.sqrt(2)
        if self.N == 0:
            return float("inf") # if explore == 0 else GameMeta.INF
        else:
            return self.Q + coeff_exploration * np.sqrt(np.log(self.parent.N) / self.N)


class MCT():

    EVAL_METHODS = ["rollout", "nn"]

    def __init__(self, eval_method: str = None) -> None:
        self.root = Node(move=None, parent=None)
        self.eval_method = MCT.EVAL_METHODS[0] if eval_method is None else eval_method
        self.n_rollouts = 0
        self.simulation_time = 0
    
    def get_stats(self) -> tuple[int, float]:
        return self.n_rollouts, self.simulation_time
    
    def get_best_action(self, board: Board) -> tuple[int,int] | None:
        """ 
        Returns the best action to play according to the current state of the MCT.
        Returns board.pass_move if the root node has no children (no legal move allowed) but the game allows to pass.
        Returns None if the root node has no children (no legal move allowed) and the game doesn't allow to pass.
        """
        
        if len(self.root.children) == 0:
            return board.pass_move # board.pass_move is None if the game doesn't allow to pass (see Board.__init__)

        (best_move, _) = fair_max(self.root.children.items(), key=lambda x: x[1].N)

        return best_move
    
    def change_root(self, move: tuple[int,int]) -> None:
        """ Change the root of the tree to the node corresponding to the new state of the board. """

        if move in self.root.children: # the new state of the board was already in the tree
            self.root = self.root.children[move]
            self.root.parent = None
        else: # the new state of the board was not in the tree so we init the tree with a new root node
            self.root = Node(move=None, parent=None)
      
    def select_node(self, cloned_board: Board) -> tuple:
        """ Perform SELECT and EXPAND steps as described in most of diagrams of the general MCTS approach. """
        
        node = self.root # use other variable to avoid modifying self.root directly
        
        while len(node.children) != 0: # while node is not a terminal node
            
            (move, node) = fair_max(node.children.items(), key=lambda x: x[1].UCB())

            cloned_board.play_move(move)

            if node.N == 0: # the selected child node has not been explored yet therefore
                return node, cloned_board

        if cloned_board.is_game_over(): # nothing to expand
            return node, cloned_board
        
        for move in cloned_board.get_moves(): # add all legal children to the node
            node.add_child(move)
        
        child_moves = list(node.children.keys())
        move = child_moves[np.random.choice(len(child_moves))] # select a child randomly
        cloned_board.play_move(move)

        return node.children[move], cloned_board

    def rollout(self, cloned_board: Board) -> int:
        """ Perform a rollout from the current state of the board and returns the id of the winner. """

        while not cloned_board.is_game_over():
            move = cloned_board.get_random_move()
            cloned_board.play_move(move)

        return cloned_board.get_winner()
    
    def back_propagate(self, node: Node, player_id: int, outcome: int) -> None:
        """ Update the statistics of the nodes on the path from the selected node to the root. """

        if outcome == 0: # draw
            reward = 0
        else:
            # node.Q is going to be updated with the reward value at the beginning of the while loop below
            # node.Q is the value of the action that led to node
            # so if the player that needs to play from node is the winner, one must not try to get to node position again
            # therefore reward must not improve the value of the action that led to node => reward = 0
            reward = 0 if outcome == player_id else 1
        
        while node is not None:
            node.Q += (node.N * node.Q + reward) / (node.N + 1) # update Q (expected reward of the node)
            node.N += 1 # update N (number of visits of the node)
            node = node.parent
            if outcome == 0: # draw => no reward
                reward = 0
            else: # the rollout ended with a win or a loss (alterate reward as we go up in the tree)
                reward = 1 - reward
    
    def __search_iter(self, cloned_board: Board) -> None:
        """ A single iteration of the search loop. """

        node, cloned_board = self.select_node(cloned_board) # cloned board is modified there to reflect the selected node
        player_to_play = cloned_board.player # if of the player that needs to play from the selected node

        if self.eval_method == "rollout":
            outcome = self.rollout(cloned_board) # player id of the winner (0 if it's a draw)
        else:
            raise ValueError(f"Evaluation method not implemented: {self.eval_method}")

        self.back_propagate(node, player_to_play, outcome)

        self.n_rollouts += 1

    def search(self, board: Board, n_sim: int = None, compute_time: float = None) -> None:
        """ Perform the MCTS algorithm to improve the knowledge of the tree. """
    
        self.n_rollouts = 0
        start_time = time()

        if n_sim is not None:
            for _ in range(n_sim):
                self.__search_iter(board.clone())
        else: # compute_time is not None
            while time() - start_time < compute_time:
                self.__search_iter(board.clone())

        self.simulation_time = time() - start_time


def main():

    _ = Node(None, None)
    print("Node created successfully!")

    _ = MCT()
    print("MCT created successfully!")


if __name__ == "__main__":
    main()