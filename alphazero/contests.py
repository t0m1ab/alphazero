from tqdm import tqdm

from alphazero.players import Player, HumanPlayer, RandomPlayer, MCTSPlayer
from alphazero.arena import Arena


def contest_1(n_rounds: int = 2):

    # store final score of each game for the winner and the number of draws
    stats = {
        "player1": [],
        "player2": [],
        "draw": 0,
    }

    for round_idx in tqdm(range(n_rounds)):

        player1 = MCTSPlayer(compute_time=0.5)
        player2 = RandomPlayer()

        arena = Arena(player1, player2, game="othello", n=8, display_dir=None)

        p2s = bool(round_idx%2)
        results = arena.play_game(player2_starts=p2s, return_results=True)

        if results["winner"] == 0:
            stats["draw"] += 1
        else:
            stats[f"player{results['winner']}"].append(results["score"])
    
        print(f"{player1} wins: {len(stats['player1'])}")
        print(f"{player2} wins: {len(stats['player2'])}")
        print(f"Draws: {stats['draw']}")
        print()
        print(f"{player1} scores = {stats['player1']}")
        print(f"{player2} scores = {stats['player2']}")


if __name__ == "__main__":
    contest_1(10)