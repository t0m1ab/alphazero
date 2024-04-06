import os
from collections import defaultdict
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
        self.game = self.board.game
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
            save_frames: bool = False,
            return_results: bool = False,
            show_indexes: bool = True,
            show_probs: bool = False,
            verbose: bool = False,
        ) -> dict:

        player_idx = 1 if player2_starts else 0

        # logs and/or displays
        if verbose:
            print(f"# Game {self.game.capitalize()} is starting with player {player_idx+1}!")
            print(f"> Player 1 = {self.player1}")
            print(f"> Player 2 = {self.player2}")
        if display:
            self.board.display(
                show_indexes=show_indexes,
                filename=f"{self.board.game}_0.png" if save_frames else None,
            )

        # initialize the board and the players
        self.board.reset()
        self.player1.reset()
        self.player2.reset()
        players = (self.player1, self.player2)
        
        round_idx = 1
        while not self.board.is_game_over():
            
            # get best move for the current player
            move, action_probs, visit_counts, prior_probs = players[player_idx].get_move(self.board)

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
                infos = {} 
                if show_probs:                   
                    if action_probs is not None:
                        infos["P[mcts]"] = action_probs
                    if prior_probs is not None:
                        infos["P[nn]"] = prior_probs
                    if visit_counts is not None:
                        infos["N"] = visit_counts
                self.board.display(
                    show_indexes=show_indexes,
                    infos=None if len(infos) == 0 else infos,
                    filename=f"{self.board.game}_{round_idx}.png" if save_frames else None,
                )

            # update internal state of players
            players[player_idx].apply_move(move, player=-self.board.player)
            players[1 - player_idx].apply_move(move, player=self.board.player)

            # switch to the other player and increment the round index
            player_idx = 1 - player_idx
            round_idx += 1
        
        score = abs(self.board.get_score())

        winner = self.board.get_winner() # 1 or -1 (or 0 for draw)
        if winner == 0: # draw
            print("Draw!") if verbose else None
            return {"winner": 0, "score": score} if return_results else None
        else:
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
    ) -> dict:
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
        stats = {
            "player1": [], # all scores for player1 victories
            "player2": [], # all scores for player2 victories
            "draw": 0, # total number of draws
            "player1_starts": defaultdict(int), # store win/loss/draw counts when player1 starts
            "player2_starts": defaultdict(int), # store win/loss/draw counts when player2 starts
        }

        pbar = tqdm(range(n_rounds), desc=f"Playing {n_rounds} games", position=call_id if call_id is not None else 0)
        start_time = time()
        for round_idx in pbar:
            
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
                stats[f"player{2 if p2s else 1}_starts"]["draw"] += 1
            else:
                stats[f"player{results['winner']}"].append(results["score"])
                stats[f"player{2 if p2s else 1}_starts"]["win" if results["winner"] == (2 if p2s else 1) else "loss"] += 1
        
            if verbose:
                print(f"{self.player1} wins: {len(stats['player1'])}")
                print(f"{self.player2} wins: {len(stats['player2'])}")
                print(f"Draws: {stats['draw']}\n")
                print(f"{self.player1} scores = {stats['player1']}")
                print(f"{self.player2} scores = {stats['player2']}")
            
            pbar.set_postfix({"p1": len(stats["player1"]), "p2": len(stats["player2"]), "draw": stats["draw"]})
        
        if verbose and call_id is None: # to avoid printing the total time in parallel mode
            print(f"Total time = {time() - start_time:.2f} seconds")

        if return_stats:
            return stats
    
    def play_games_in_parallel(
        self, 
        n_rounds: int,
        n_process: int = None,
        verbose: bool = False,
        return_stats: bool = False,
    ) -> dict:
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
        
        if verbose:
            print(f"Total time = {time() - start_time:.2f} seconds")

        # merge stats from all arenas
        stats = {
            "player1": [], # all scores for player1 victories
            "player2": [], # all scores for player2 victories
            "draw": 0, # total number of draws
            "player1_starts": defaultdict(int), # store win/loss/draw counts when player1 starts
            "player2_starts": defaultdict(int), # store win/loss/draw counts when player2 starts
        }
        for s, params in zip(arenas_stats, parameters):
            stats["player1"].extend(s["player1"])
            stats["player2"].extend(s["player2"])
            stats["draw"] += s["draw"]
            start_player = f"player{params['start_player']}"
            stats[f"{start_player}_starts"]["win"] += s[f'{start_player}_starts']["win"]
            stats[f"{start_player}_starts"]["loss"] += s[f'{start_player}_starts']["loss"]
            stats[f"{start_player}_starts"]["draw"] += s[f'{start_player}_starts']["draw"]

        if return_stats:
            return stats
    
    @staticmethod
    def print_stats_results(player1: Player, player2: Player, stats: dict):
        """
        Print the results of a contest between <player1> and <player2> stored in <stats>.
        """

        n_rounds = len(stats["player1"]) + len(stats["player2"]) + stats["draw"]
        print("\nRESULTS:")

        # global stats
        print(f"- {player1} wins = {len(stats['player1'])}/{n_rounds}")
        print(f"- {player2} wins = {len(stats['player2'])}/{n_rounds}")
        print(f"- Draws = {stats['draw']}/{n_rounds}")

        # stats by starting player
        p1_starts = stats["player1_starts"]
        p2_starts = stats["player2_starts"]
        print(f"- {player1} results when starting: win={p1_starts['win']} | loose={p1_starts['loss']} | draw={p1_starts['draw']}")
        print(f"- {player2} results when starting: win={p2_starts['win']} | loose={p2_starts['loss']} | draw={p2_starts['draw']}")


def main():

    _ = Arena(player1=HumanPlayer(), player2=RandomPlayer(), board=OthelloBoard(8))
    print("Arena created successfully!")


if __name__ == "__main__":
    main()
