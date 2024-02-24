import os
from tqdm import tqdm
from multiprocessing.pool import Pool
from time import time

from alphazero.base import Board, Player
from alphazero.players import HumanPlayer, RandomPlayer, MCTSPlayer
from alphazero.games.othello import OthelloBoard


class Arena():
    """
    Base class to play games between two players and return the statistics of the games.
    Games can be played in parallel using multiple processes to speed up the evaluation.
    """

    def __init__(self, player1: Player, player2: Player, board: Board) -> None:   
        self.board = board
        self.game = self.board.game_name
        self.player1 = player1
        self.player2 = player2

    def __check_inputs(self, n_rounds: int, n_process: int = None, start_player: int = None) -> None:
        """ Check if the inputs are valid. """
        if start_player is None and n_rounds % 2 != 0:
            raise ValueError("n_rounds must be an even number or the evaluation will be biased!")
        if n_process is not None: # parallel mode
            if n_process % 2 != 0:
                raise ValueError("n_process must be an even number when evaluating games in parallel or the evaluation will be biased!")
            if n_rounds % n_process != 0:
                raise ValueError("n_rounds must be divisible by the number of processes to play games in parallel!")
        if start_player is not None and start_player not in [1, 2]:
            raise ValueError("start_player must be either 1 or 2 or None (to alternate starts)")

    def play_game(
            self, 
            player2_starts: bool = False, 
            display: bool = False,
            return_results: bool = False,
            verbose: bool = False,
        ) -> int:

        player_idx = 1 if player2_starts else 0

        # logs and/or displays
        if verbose:
            print(f"# Game {self.game.capitalize()} is starting with player {player_idx+1}!")
            print(f"> Player 1 = {self.player1.__class__.__name__}")
            print(f"> Player 2 = {self.player2.__class__.__name__}")
        if display:
            self.board.display(indexes=True)

        # initialize the board and the players
        self.board.reset()
        self.player1.reset()
        self.player2.reset()
        players = (self.player1, self.player2)
                  
        while not self.board.is_game_over():
            
            # get best move for the current player
            move = players[player_idx].get_move(self.board)

            # play the move on the board
            self.board.play_move(move)

            # logs and/or displays
            if verbose:
                dict_stats = players[player_idx].get_stats_after_move()
                msg = f"{players[player_idx]} played {move} | score = {-self.board.get_score()} | "
                for k, v in dict_stats.items():
                    msg += f"{k} = {v:.3f} | " if type(v) == float else f"{k} = {v} | "
                print(msg[:-3])
            if display:
                self.board.display(indexes=True)

            # update internal state of players
            players[player_idx].apply_move(move, player=-self.board.player)
            players[1 - player_idx].apply_move(move, player=self.board.player)

            # switch to the other player
            player_idx = 1 - player_idx
        
        score = abs(self.board.get_score())

        if score == 0:
            print("Draw!") if verbose else None
            return {"winner": 0, "score": score} if return_results else None
        else:
            winner = self.board.get_winner() # 1 or -1
            if (winner == 1 and not player2_starts) or (winner == -1 and player2_starts):
                player_winner = 1 # player1
            else:
                player_winner = 2 # player2
            print(f"Player {player_winner} wins with score = {score}") if verbose else None
            return {"winner": player_winner, "score": score} if return_results else None
    
    def play_games(
        self, 
        n_rounds: int,
        start_player: int = None,
        return_stats: bool = False,
        verbose: bool = False,
        call_id: int = None,
    ) -> int:
        """
        Play n_rounds games between player1 and player2 and return the statistics of the games.

        ARGUMENTS:
            - n_rounds: number of games to play.
            - start_player: if None, the starting player will alternate between player1 and player2.
                            if 1, player1 will start all games and if 2, player2 will start all games.
            - return_stats: if True, the statistics of the games will be returned.
            - call_id: must always be None unless the function is called in parallel mode.
        """
        
        self.__check_inputs(n_rounds, start_player=start_player)
        
        # store final score of each game for the winner and the number of draws
        stats = {"player1": [], "player2": [], "draw": 0}

        start_time = time()
        iterator = tqdm(range(n_rounds), desc=f"Playing {n_rounds} games", position=call_id if call_id is not None else 0)
        start_time = time()
        for round_idx in iterator:
            
            # set the starting player
            if start_player == 1:
                p2s = False
            elif start_player == 2:
                p2s = True
            else:
                p2s = bool(round_idx%2)

            # play the game
            results = self.play_game(player2_starts=p2s, return_results=True, verbose=verbose)

            # store the results
            if results["winner"] == 0:
                stats["draw"] += 1
            else:
                stats[f"player{results['winner']}"].append(results["score"])
        
            if verbose:
                print(f"{self.player1} wins: {len(stats['player1'])}")
                print(f"{self.player2} wins: {len(stats['player2'])}")
                print(f"Draws: {stats['draw']}\n")
                print(f"{self.player1} scores = {stats['player1']}")
                print(f"{self.player2} scores = {stats['player2']}")
        
        if call_id is None: # to avoid printing the total time in parallel mode
            print(f"Total time = {time() - start_time:.2f} seconds")

        if return_stats:
            return stats
    
    def play_games_in_parallel(
        self, 
        n_rounds: int,
        n_process: int = None,
        verbose: bool = False,
        return_stats: bool = False,
    ) -> int:
        """
        Play n_rounds games between player1 and player2 and return the statistics of the games.
        The games are played in parallel using n_process process.
        For each process, an arena is created and n_rounds/n_process games are played with it.
        Half arenas start games with player1 and the other half with player2 to avoid biased evaluations.
        """

        n_process = n_process if n_process is not None else os.cpu_count()
        self.__check_inputs(n_rounds, n_process)
        
        # clone players to avoid conflicts between processes and deactivate logs
        p1 = self.player1.clone()
        p2 = self.player2.clone()
        p1.verbose = False
        p2.verbose = False

        start_time = time()
        with Pool(n_process) as pool:

            # create an arena for each process (n_rounds/n_process games will be played in each arena by each process)
            process_arenas_callable = [
                Arena(
                    player1=p1.clone(),
                    player2=p2.clone(), 
                    board=self.board.clone(),
                ).play_games for _ in range(n_process)
            ]

            # set the parameters for each arena
            parameters = [{
                "n_rounds": n_rounds//n_process,
                "start_player": 1 if bool(i%2) else 2, # half arenas will start games with player2 and the other half with player1
                "return_stats": True, 
                "verbose": verbose,
                "call_id": i,
            } for i in range(n_process)]

            # launch the games in arenas in parallel
            tasks = zip(process_arenas_callable, parameters)
            futures = [pool.apply_async(func=arena_call, kwds=kwds) for arena_call, kwds in tasks]
            arenas_stats = [fut.get() for fut in futures]
        
        print(f"Total time = {time() - start_time:.2f} seconds")
        print("PROOOOOOOOOUT")

        # merge stats from all arenas
        stats = {"player1": [], "player2": [], "draw": 0}
        for s in arenas_stats:
            stats["player1"].extend(s["player1"])
            stats["player2"].extend(s["player2"])
            stats["draw"] += s["draw"]

        if return_stats:
            return stats


def main():

    _ = Arena(player1=HumanPlayer(), player2=RandomPlayer(), board=OthelloBoard(8))
    print("Arena created successfully!")


if __name__ == "__main__":
    main()
