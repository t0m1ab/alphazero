import numpy as np

from alphazero.players import Player, HumanPlayer, RandomPlayer, MCTSPlayer
from alphazero.games.othello import OthelloBoard


class Arena():

    GAMES = {
        "othello": OthelloBoard,
    }

    def __init__(self, player1: Player, player2: Player, game: str, **game_kwargs) -> None:

        if game not in Arena.GAMES:
            raise ValueError(f"Game {game} not available")
        
        self.game = game
        self.board = Arena.GAMES[game](**game_kwargs)
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
            if verbose:
                print(f"Player {player_idx+1} plays move {move}")
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


def main():

    _ = Arena(HumanPlayer(), RandomPlayer(), game="othello", n=8)
    print("Arena created successfully!")


if __name__ == "__main__":
    main()
