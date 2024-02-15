import os
from tqdm import tqdm
from multiprocessing.pool import Pool
from time import time

from alphazero.base import Board, Player
from alphazero.players import HumanPlayer, RandomPlayer, MCTSPlayer
from alphazero.games.othello import OthelloBoard


class Arena():

    def __init__(self, player1: Player, player2: Player, board: Board) -> None:   
        self.board = board
        self.game = self.board.game_name
        self.player1 = player1
        self.player2 = player2

    def play_game(
            self, 
            player2_starts: bool = False, 
            verbose: bool = False,
            display: bool = False,
            return_results: bool = False,
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
        self.player1.reset(verbose)
        self.player2.reset(verbose)
        players = (self.player1, self.player2)
                  
        while not self.board.is_game_over():
            
            # get best move for the current player
            move = players[player_idx].get_move(self.board)

            # play the move on the board
            self.board.play_move(move)

            # logs and/or displays
            # if verbose:
            #     print(f"Player {player_idx+1} plays move {move}")
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
        verbose: bool = False,
        return_stats: bool = False,
    ) -> int:
        """
        Play n_rounds games between player1 and player2 and return the statistics of the games.
        """
        
        if n_rounds % 2 != 0:
            raise ValueError("n_rounds must be an even number or the evaluation will be biased!")
        
        # store final score of each game for the winner and the number of draws
        stats = {"player1": [], "player2": [], "draw": 0}

        start_time = time()
        for round_idx in tqdm(range(n_rounds), desc=f"Playing {n_rounds} games"):

            p2s = bool(round_idx%2)
            results = self.play_game(player2_starts=p2s, return_results=True)

            if results["winner"] == 0:
                stats["draw"] += 1
            else:
                stats[f"player{results['winner']}"].append(results["score"])
        
            if verbose:
                print(f"{self.player1} wins: {len(stats['player1'])}")
                print(f"{self.player2} wins: {len(stats['player2'])}")
                print(f"Draws: {stats['draw']}")
                print()
                print(f"{self.player1} scores = {stats['player1']}")
                print(f"{self.player2} scores = {stats['player2']}")
        
        print(f"Total time = {time() - start_time:.2f} seconds")

        if return_stats:
            return stats
    
    def play_games_in_parallel(
        self, 
        n_rounds: int,
        verbose: bool = False,
        return_stats: bool = False,
        n_process: int = None,
    ) -> int:
        """
        Play n_rounds games between player1 and player2 and return the statistics of the games.
        The games are played in parallel using n_process process.
        """

        n_process = n_process if n_process is not None else os.cpu_count()

        if n_rounds % 2 != 0:
            raise ValueError("n_rounds must be an even number or the evaluation will be biased!")

        if n_process < 2:
            raise ValueError("n_process must be at least 2 to play games in parallel or you should call play_games instead")
        
        if n_rounds % n_process != 0:
            raise ValueError("n_rounds must be divisible by the number of processes to play games in parallel!")
        
        arenas_play_game_callable = [
            Arena(
                player1=self.player1.clone(verbose=False), # deactivate logs because multiple process in parallel
                player2=self.player2.clone(verbose=False), 
                board=self.board.clone(),
            ).play_game for _ in range(n_rounds)
        ]

        parameters = [{"player2_starts": bool(i%2), "return_results": True, "verbose": verbose} for i in range(n_rounds)]

        start_time = time()
        with Pool(n_process) as pool:
            tasks = zip(arenas_play_game_callable, parameters)
            futures = [pool.apply_async(func=play_game_func, kwds=kwds) for play_game_func, kwds in tasks]
            pool_results = [
                fut.get() for fut in tqdm(futures, desc=f"Playing {n_rounds} games with {n_process} process")
            ]
        
        print(f"Total time = {time() - start_time:.2f} seconds")

        # store final score of each game for the winner and the number of draws
        stats = {"player1": [], "player2": [], "draw": 0}
        for res in pool_results:
            if res["winner"] == 0:
                stats["draw"] += 1
            else:
                stats[f"player{res['winner']}"].append(res["score"])

        if return_stats:
            return stats


def main():

    _ = Arena(player1=HumanPlayer(), player2=RandomPlayer(), board=OthelloBoard(8))
    print("Arena created successfully!")


if __name__ == "__main__":
    main()
