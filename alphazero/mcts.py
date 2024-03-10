import numpy as np
from time import time

from alphazero.utils import fair_max
from alphazero.base import TreeEval, Action, Board, PolicyValueNetwork


class Node:
    """
    Represents a state of the game (= board configuration)
    Each node can be associated with a state of the game (board representation).
    Nodes don't directly contain this information as an attribute. 
    That's why the state of the board must be maintained in parallel during select and rollout.
    """

    def __init__(self, move: Action, parent, prob: float = None) -> None:
        self.move = move # move that led to this node
        self.parent = parent # parent node
        self.N = 0 # number of visits of the node from the creation of the tree
        self.Q = 0 # avergage win rate/expected reward of the node (meaning of the action that led to the node)
        self.children = {} # {move1: NodeChild1, move2: NodeChild2, ...}
        # self.P and self._children_probs are only used when the evaluation method is set to TreeEval.NEURAL (AlphaZero)
        self.P = prob # prior probability of taking the action that led to the node from the parent node (see why prior in PUCT formula)
        self._children_probs = None # array storing the raw probs (obtained from a nn) of all existing actions in the game
    
    def __str__(self) -> str:
        n_siblings = len(self.parent.children) if self.parent is not None else 0
        return f"Node: N={self.N} | Q={self.Q} | UCT={self.UCT()} | n_children={len(self.children)} | move={self.move} | n_siblings={n_siblings}"
    
    def add_child(self, move: Action, prob: float = None) -> None:
        self.children[move] = Node(move=move, parent=self, prob=prob)

    def UCT(self, c_exploration: float = np.sqrt(2)) -> float:
        """ Classic UCB formula for trees. """
        if self.N == 0: # always give the priority to a node that has not been explored yet during selection
            return float("inf")
        return self.Q + c_exploration * np.sqrt(np.log(self.parent.N) / self.N)
    
    def PUCT(self, c_puct: float = 1.0) -> float:
        """ PUCT formula for MCTS when used with AlphaZero."""
        return self.Q + c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)


class MCT():
    """
    Monte Carlo Tree of reachable states of the game and the associated statistics of the transitions.
    The tree is composed of <Node> and expanded following the MCTS algorithm.
    Two evaluation methods are available: ROLLOUT (classic simulation) and NEURAL (nn evaluation).
    """
    
    def __init__(self, eval_method: str = None, nn: PolicyValueNetwork = None) -> None:
        self.root = Node(move=None, parent=None)
        self.n_rollouts = 0
        self.simulation_time = 0
        self.eval_method = TreeEval.to_dict()["rollout" if eval_method is None else eval_method]
        self._nn = nn
        if self._nn is not None and self.eval_method != TreeEval.NEURAL:
            raise ValueError(f"A neural network has been set for the MCT but the evaluation method is {self.eval_method}")

    @property
    def nn(self):
        """ Getter: self._nn """
        return self._nn
    
    @nn.setter
    def nn(self, nn: PolicyValueNetwork) -> None:
        """ Setter: self._nn """
        if self.eval_method != TreeEval.NEURAL:
            raise ValueError(f"Trying to set a neural network for the MCT but the evaluation method is {self.eval_method}")
        self._nn = nn
    
    def get_stats(self) -> tuple[int, float]:
        return self.n_rollouts, self.simulation_time
    
    def get_action_probs(self, board: Board, temp: float = 0) -> dict[Action, float]:
        """ 
        Returns a probs distribution on the legal actions according to the current state of the MCT.
        Returns None if the root node has no children (no legal move meaning the game doesn't allow to pass).
        """
        
        if len(self.root.children) == 0:
            return {board.pass_move: 1} # board.pass_move is None if the game doesn't allow to pass (see Board.__init__)

        if temp == 0: # return the move with the highest visit count
            # for move, node in self.root.children.items():
            #     print(f"{move} -> {node}")
            best_move, _ = fair_max(self.root.children.items(), key=lambda x: x[1].N)
            return {best_move: 1}
        else: # return a distribution of the moves according to their visit count
            move_values = {move: node.N ** (1. / temp) for move, node in self.root.children.items()}
            sum_values = sum(move_values.values())
            return {move: value / sum_values for move, value in move_values.items()}
    
    def change_root(self, move: Action) -> None:
        """ Change the root of the tree to the node corresponding to the new state of the board. """

        if move in self.root.children: # the new state of the board was already in the tree
            self.root = self.root.children[move]
            self.root.parent = None
        else: # the new state of the board was not in the tree so we init the tree with a new root node
            self.root = Node(move=None, parent=None)
      
    def select_node(self, cloned_board: Board) -> tuple:
        """ Perform SELECT and EXPAND steps: find a leaf node, expand it if possible and returns the expanded node. """
        
        node = self.root # start from the root node
        
        while len(node.children) != 0: # don't modify attributes of node in this loop: only read them
            
            if self.eval_method ==TreeEval.ROLLOUT:
                move, node = fair_max(node.children.items(), key=lambda x: x[1].UCT())
            elif self.eval_method == TreeEval.NEURAL:
                move, node = fair_max(node.children.items(), key=lambda x: x[1].PUCT())
            else:
                raise ValueError(f"Unsuported evaluation method: {self.eval_method}")

            cloned_board.play_move(move)

            if node.N == 0: # that means that node is an expanded node (the selected node was its parent)
                return node, cloned_board

        if cloned_board.is_game_over(): # terminal node => nothing to expand, the node evaluation will simply be the outcome of the game
            return node, cloned_board
        
        ### At this point of the code, <node> is the selected node that needs to be expanded
        
        # add (ghost) children to the node (their visit count N is 0 so they are not really part of the tree yet)
        if self.eval_method == TreeEval.ROLLOUT:
            for move in cloned_board.get_moves():
                node.add_child(move=move)
        elif self.eval_method == TreeEval.NEURAL:
            legal_moves = cloned_board.get_moves()
            # at this level node._children_probs is not None because <node> didn't trigger the condition node.N == 0 before
            norm_children_probs = self._nn.get_normalized_probs(node._children_probs, legal_moves)     
            for move, prob in norm_children_probs.items():
                node.add_child(move=move, prob=prob)
        
        # expand the selected node by selecting a move
        if self.eval_method == TreeEval.ROLLOUT: # select a child randomly
            child_moves = list(node.children.keys())
            move = child_moves[np.random.choice(len(child_moves))]
        elif self.eval_method == TreeEval.NEURAL: # use PUCT
            move, _ = fair_max(node.children.items(), key=lambda x: x[1].PUCT())
        
        cloned_board.play_move(move)

        return node.children[move], cloned_board

    def rollout(self, cloned_board: Board) -> int:
        """ Perform the SIMULATION/ROLLOUT/PLAYOUT step from the current state of the board and returns the id of the winner. """

        while not cloned_board.is_game_over():
            move = cloned_board.get_random_move()
            cloned_board.play_move(move)

        return cloned_board.get_winner()
    
    def nn_evaluation(self, cloned_board: Board, node: Node) -> tuple[dict[Action, float], int]:
        """ Perform ROLLOUT through nn evaluation from the current state of the board and returns the id of the winner. """

        if cloned_board.is_game_over():
            return cloned_board.get_winner()
        
        # evaluation of the state of the cloned board from the viewpoint of the player that needs to play
        probs, outcome = self._nn.evaluate(cloned_board) # v in [-1,1]
        if node._children_probs is None: # store the raw prior probs of the children of the node
            node._children_probs = probs
        else:
            raise ValueError("The children probs of the node are not empty... This should not happen.")
        
        return outcome
    
    def back_propagate(self, node: Node, player_id: int, outcome: int) -> None:
        """ Perform the BACKPROPAGATION step: update the statistics of the nodes on the path from the selected node to the root. """

        draw = (outcome == 0)

        if draw:
            reward = 0
        elif self.eval_method == TreeEval.ROLLOUT:
            # node.Q is going to be updated with the reward value at the beginning of the while loop below
            # node.Q is the value of the action that led to node
            # so if the player that needs to play from node is the winner, one must not try to get to node position again
            # that means that node.Q must be penalized (-1) when player_id is the winner
            reward = -1 if player_id == outcome else 1
            # reward = 0 if player_id == outcome else 1 ### ALTERNATIVE UPDATE BUT WEAKER
        elif self.eval_method == TreeEval.NEURAL:
            reward = -1 if player_id == outcome else 1
        
        while node is not None:
            node.Q = (node.N * node.Q + reward) / (node.N + 1) # update Q (expected reward of the transition that led to the node)
            node.N += 1 # update N (number of visits of the node <=> number of times the transition that led to the node was selected)
            node = node.parent
            if draw:
                reward = 0
            elif self.eval_method == TreeEval.ROLLOUT:
                reward = -reward # alterate the increment of the win count as we go up in the tree
                # reward = 1 - reward ### ALTERNATIVE UPDATE BUT WEAKER
            elif self.eval_method == TreeEval.NEURAL:
                reward = -reward
    
    def __search_iter(self, cloned_board: Board) -> None:
        """ A single iteration of the search loop. """

        # init the root node with the nn probs if the tree was restarted from an unexplored state
        if self.eval_method == TreeEval.NEURAL and self.root._children_probs is None:
            probs, _ = self._nn.evaluate(cloned_board)
            self.root._children_probs = probs
        
        node, cloned_board = self.select_node(cloned_board) # cloned board is modified there to reflect the selected node
        player_to_play = cloned_board.player # id of the player that needs to play from the selected node

        if self.eval_method == TreeEval.ROLLOUT:
            outcome = self.rollout(cloned_board) # player id of the winner (0 if it's a draw)
        elif self.eval_method == TreeEval.NEURAL:
            outcome = self.nn_evaluation(cloned_board, node) # player id of the winner (0 if it's a draw)

        self.back_propagate(node, player_to_play, outcome)

        self.n_rollouts += 1

    def search(self, board: Board, n_sim: int = None, compute_time: float = None) -> None:
        """ Perform the MCTS algorithm to improve the knowledge of the tree. """
    
        self.n_rollouts = 0
        start_time = time()

        if n_sim is not None:
            for _ in range(n_sim):
                self.__search_iter(board.clone())
        elif compute_time is not None:
            while time() - start_time < compute_time:
                self.__search_iter(board.clone())
        else:
            raise ValueError("MCT.search needs to have either n_sim or compute_time specified.")

        self.simulation_time = time() - start_time


def main():

    _ = Node(None, None)
    print("Node created successfully!")

    _ = MCT()
    print("MCT created successfully!")


if __name__ == "__main__":
    main()