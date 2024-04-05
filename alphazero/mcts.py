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
        self.has_dirichlet_noise = False # flag to know if dirichlet noise has been added to the P attributes of the children
    
    def __str__(self) -> str:
        n_siblings = len(self.parent.children) if self.parent is not None else 0
        if self.P is not None:
            action_value = f"P={self.P} | PUCT={self.PUCT()}"
        else:
            action_value = f"UCT={self.UCT()}"
        return f"Node: N={self.N} | Q={self.Q} | {action_value} | n_children={len(self.children)} | move={self.move} | n_siblings={n_siblings}"
    
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
    
    def __init__(
            self, 
            eval_method: str = None, 
            nn: PolicyValueNetwork = None,
            dirichlet_alpha: float = None,
            dirichlet_epsilon: float = None,
        ) -> None:
        self.root = Node(move=None, parent=None)
        self.n_rollouts = 0
        self.simulation_time = 0
        self.eval_method = TreeEval.to_dict()["rollout" if eval_method is None else eval_method]
        self._nn = nn
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
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
    
    def get_prior_probs(self) -> dict[Action, float]:
        """ 
        Return the prior probs of the children of the root node.
        This method is only useful when the evaluation method is set to TreeEval.NEURAL otherwise it returns None for each move.
        """
        return {move: node.P for move, node in self.root.children.items()}

    def get_action_probs(self, board: Board, temp: float = 0) -> tuple[dict[Action, float], dict[Action, int]]:
        """ 
        Returns a probs distribution on the legal actions according to the current state of the MCT.
        Returns None if the root node has no children (no legal move meaning the game doesn't allow to pass).
        """
        
        if len(self.root.children) == 0:
            return {board.pass_move: 1.} # board.pass_move is None if the game doesn't allow to pass (see Board.__init__)
        
        # print("\nRoot children:")
        # for move, node in self.root.children.items():
        #     print(f"{move} -> {node}")
        
        visit_counts = {move: int(node.N) for move, node in self.root.children.items()}

        if temp == 0: # return the move with the highest visit count
            best_move, _ = fair_max(self.root.children.items(), key=lambda x: x[1].N)
            return {best_move: 1}, visit_counts
        else: # return a distribution of the moves according to their visit count
            move_values = {move: node.N ** (1. / temp) for move, node in self.root.children.items()}
            sum_values = sum(move_values.values())
            return {move: value / sum_values for move, value in move_values.items()}, visit_counts
    
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
            for move, prob in norm_children_probs.items(): # OPTIONAL UPDATE: ADD DIRICHLET NOISE HERE IF NODE==ROOT 
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
    
    def nn_evaluation(self, cloned_board: Board, node: Node) -> int:
        """ Perform ROLLOUT through nn evaluation from the current state of the board and returns the id of the winner. """

        if cloned_board.is_game_over():
            return cloned_board.get_winner()
        
        # evaluation of the state of the cloned board in [-1,1]: >0 value => 1 likely to win | <0 value => -1 likely to win
        probs, outcome = self._nn.evaluate(cloned_board)
        if node._children_probs is None: # store the raw prior probs of the children of the node
            node._children_probs = probs
        else:
            raise ValueError("The children probs of the node are not empty... This should not happen.")
        
        return outcome
    
    def back_propagate(self, node: Node, player_id: int, outcome: float) -> None:
        """ 
        Perform the BACKPROPAGATION step: update the statistics of the nodes on the path from the selected node to the root.
        
        ARGUMENTS:
            - node: the selected node from which the simulation/evaluation was performed
            - player_id: the id of the player that needs to play from the selected node
            - outcome: value in [-1,1] which estimates/evaluates the winner of the game (0 if it's a draw)
        """

        draw = (abs(outcome) < 1e-4) # useful if neural evaluation is too close to 0

        if draw:
            reward = 0
        else:
            # node.Q is going to be updated with the reward value at the beginning of the while loop below
            # node.Q is the value of the action that led to node
            # so if the player that needs to play from node is the winner, one must try to avoid this position in the future
            # that means that node.Q must be penalized with negative reward
            # we use the condition (player_id * outcome > 0) to check safely if both values in [-1,1] have the same sign (i.e. penalization required)
            reward = -abs(outcome) if player_id * outcome > 0 else abs(outcome)
                
        while node is not None:
            node.Q = (node.N * node.Q + reward) / (node.N + 1) # update Q (expected reward of the transition that led to the node)
            node.N += 1 # update N (number of visits of the node <=> number of times the transition that led to the node was selected)
            node = node.parent
            reward = -reward # alterate reward (0 stays 0 if draw)

    
    def __search_iter(self, cloned_board: Board) -> None:
        """ A single iteration of the search loop. """

        if self.eval_method == TreeEval.NEURAL:
            # init the root node with the nn probs if the tree was restarted from an unexplored state
            if self.root._children_probs is None: 
                probs, _ = self._nn.evaluate(cloned_board)
                self.root._children_probs = probs
            # add dirichlet noise to the transition probabilities from the root node if necessary
            if self.dirichlet_alpha is not None and self.dirichlet_epsilon is not None:
                if self.root.has_dirichlet_noise is False and len(self.root.children) > 0:
                    self.root.has_dirichlet_noise = True
                    dirichlet_noise = np.random.dirichlet([self.dirichlet_alpha for _ in range(len(self.root.children))])
                    for idx, move in enumerate(self.root.children):
                        self.root.children[move].P = (1 - self.dirichlet_epsilon) * self.root.children[move].P + self.dirichlet_epsilon * dirichlet_noise[idx]
        
        node, cloned_board = self.select_node(cloned_board) # cloned board is modified there to reflect the selected node
        player_to_play = cloned_board.player # id of the player that needs to play from the selected node

        if self.eval_method == TreeEval.ROLLOUT:
            outcome = float(self.rollout(cloned_board)) # player id of the winner (0 if it's a draw)
        elif self.eval_method == TreeEval.NEURAL:
            outcome = self.nn_evaluation(cloned_board, node) # player id estimation of the winner (0 if it's a draw)

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