# AlphaZero applied to board games

**Authors:** Tom LABIAUSSE - Amine CHERIF HAOUAT - Sami JALLOULI

**Date:** Feb/Mar 2024

## 0 - Setup

* Clone the repository:
```bash
git clone git@github.com:t0m1ab/alphazero.git
```

* Install `alphazero` as a package in edit mode (see config in `pyproject.toml`): 
```bash
cd alphazero/
pip install -e .
``` 

* You should be able to run tests on the package or print the docs with the following commands in the terminal: 
```bash
alphazero --test
alphazero --help
```

* Download the alphazero networks available in our [Huggingface Hub](https://huggingface.co/t0m1ab) using one of the following equivalent commands: 
```bash
alphazero --download
```
```python
python alphazero/utils.py
``` 
All models and configuration files will be stored in a `models/` folder by default when loading or training a player.

## 1 - Files

* `base`: implement parent classes such as *Board*, *Player*, *PolicyValueNetwork*...
* `players.py`: implement different game strategies
* `mcts.py`: implement Monte Carlo Tree Search
* `trainers.py`: implement a trainer for AlphaZero
* `arena.py`: organize several games between players and compare results
* `game_ui.py`: interface between user and algorithm to play a game (TODO using gradio)
* `contests.py`: define specific contests between players
* `utils.py`: utility functions
* `tests.py`: contains various tests that can be run to check the implementation

### games/
* `configs.py`: define default configurations mapping
* `othello.py`: implementation of the Othello environment, game config and neural network for AlphaZero
* `connect4.py`: TODO
* `tictactoe.py`: TODO

### docs/
* `help.txt`: general informations

### figures/
* `othello_board_8x8_init.png`: example of Othello 8x8 board display

<img src='./alphazero/figures/othello_board_8x8_init.png' width='300'>

## 2 - Demo

### 2.1 - Play against MCTSPlayer

```bash
python game_ui.py
```

Change code in `game_ui.py` to modify the machine player and/or the game settings. The state of the board will be automatically saved as a PNG file in `outputs/` and overwrite itself after each move.

### 2.2 - Compare machine players

```bash
python contests.py
```

Change code in `contests.py` to modify the machine players and/or the game settings.

### 2.3 - Train an AlphaZero player

```bash
python trainers.py
```

Change the game in the main function of `trainers.py` and the training configuration in the associated `games/<game_name>.py` file.
