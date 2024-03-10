#!/bin/bash

# OPTIONS:
# --game : name of the game (e.g. "othello")
# --experiment-name : name of the experiment (e.g. "alphaothellozero-test")
# --config : path to the json config file (e.g. "configs/othello.json")
# -x : use this flag to perform training time estimation instead of training

# LAUNCH COMMAND
# bash run.sh &

nohup python trainer.py \
    --game "othello" \
    --experiment-name "alphazero-othello" \
    # --config="configs/othello.json" \
