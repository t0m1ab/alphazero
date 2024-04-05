import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json

from alphazero.utils import DEFAULT_MODELS_PATH


def plot_loss(
        model_name: str,
        max_iterations: int = None,
        x_ticks: int = None,
        path: str = None,
    ):
    """
    Plot training loss from a given loss history JSON file and limit to the first <max_iterations> if specified.
    """

    path = DEFAULT_MODELS_PATH if path is None else path
    model_dir = os.path.join(path, model_name)
    filepath = os.path.join(model_dir, "loss.json")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found...")
    
    with open(filepath, "r") as f:
        loss = json.load(f)

    iterations = len(loss)
    if iterations == 0:
        raise ValueError("No data found in loss file...")
    if max_iterations is not None and max_iterations > 0 and iterations > max_iterations:
        iterations = max_iterations
    epochs = len(loss["0"])
    
    # concatenate losses for all iterations
    all_pi_loss = []
    all_v_loss = []
    for iter_idx in range(iterations):
        for epoch_idx in range(epochs):
            all_pi_loss.extend(loss[str(iter_idx)][str(epoch_idx)]["pi"])
            all_v_loss.extend(loss[str(iter_idx)][str(epoch_idx)]["v"])
    all_pi_loss = np.array(all_pi_loss).reshape(-1)
    all_v_loss = np.array(all_v_loss).reshape(-1)
    
    if not len(all_pi_loss) == len(all_v_loss):
        raise ValueError("Length of pi and v losses do not match...")
    
    # number of optimization steps per epoch for each iteration
    steps_per_epoch = {iter_idx: len(loss[str(iter_idx)]["0"]["pi"]) for iter_idx in range(iterations)}

    # optimization step index at the end of each iteration
    step_after_iteration = np.zeros(1 + iterations)
    for iter_idx in range(iterations):
        step_after_iteration[iter_idx+1] = step_after_iteration[iter_idx] + epochs * steps_per_epoch[iter_idx]

    # compute mean loss for each epoch in each iteration
    pi_loss_epochs = np.zeros((iterations, epochs))
    v_loss_epochs = np.zeros((iterations, epochs))
    step_after_epoch = np.zeros((iterations, epochs)) # optimization step index at the end of each epoch
    step_counter = 0 
    for iter_idx in range(iterations):
        for epoch_idx in range(epochs):
            pi_loss_epochs[iter_idx, epoch_idx] = np.mean(loss[str(iter_idx)][str(epoch_idx)]["pi"])
            v_loss_epochs[iter_idx, epoch_idx] = np.mean(loss[str(iter_idx)][str(epoch_idx)]["v"])
            step_after_epoch[iter_idx, epoch_idx] = step_counter + (epoch_idx + 1) * steps_per_epoch[iter_idx]
        step_counter = step_after_epoch[iter_idx, -1]
    
    # compute total loss for each optimization step and each epoch average
    total_loss = all_pi_loss + all_v_loss
    total_loss_epochs = pi_loss_epochs + v_loss_epochs

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    x_ticks = x_ticks if x_ticks is not None else 5

    ax[0].plot(all_pi_loss, label="pi loss", alpha=0.5, color="tab:blue")
    ax[0].plot(all_v_loss, label="v loss", alpha=0.5, color="tab:orange")
    ax[0].plot(step_after_epoch.reshape(-1), pi_loss_epochs.reshape(-1), label="pi loss (mean over epoch)", marker="o", color="tab:blue")
    ax[0].plot(step_after_epoch.reshape(-1), v_loss_epochs.reshape(-1), label="v loss (mean over epoch)", marker="o", color="tab:orange")
    ax[0].set_xticks(step_after_iteration[::x_ticks], labels=[f"{i}" for i in range(0, iterations+1, x_ticks)])
    ax[0].set_ylabel("loss")
    ax[0].grid()
    ax[0].legend(loc="upper right")

    ax[1].plot(total_loss, label="loss", alpha=0.5, color="tab:green")
    ax[1].plot(step_after_epoch.reshape(-1), total_loss_epochs.reshape(-1), label="loss (mean over epoch)", marker="o", color="tab:green")
    ax[1].set_xticks(step_after_iteration[::x_ticks], labels=[f"{i}" for i in range(0, iterations+1, x_ticks)])
    ax[1].set_xlabel(f"policy iteration (total optimization steps = {len(total_loss)})")
    ax[1].set_ylabel("loss")
    ax[1].grid()
    ax[1].legend(loc="upper right")

    fig.suptitle(f"Training loss for {model_name} over {iterations} iterations with {epochs} epochs each", fontweight="bold")
    fig.tight_layout()

    plt.savefig(os.path.join(model_dir, "loss.png"), dpi=300)


def plot_eval_results(
        model_name: str,
        max_iterations: int = None,
        x_ticks: int = None,
        path: str = None,
    ):
    """
    Plot evaluation results from a given evaluation history JSON file and limit to the first <max_iterations> if specified.
    """

    path = DEFAULT_MODELS_PATH if path is None else path
    model_dir = os.path.join(path, model_name)
    filepath = os.path.join(model_dir, "eval.json")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found...")
    
    with open(filepath, "r") as f:
        eval_results = json.load(f)

    iterations = len(eval_results["results"])
    if iterations == 0:
        raise ValueError("No data found in results file...")
    if max_iterations is not None and max_iterations > 0 and iterations > max_iterations:
        iterations = max_iterations
    iteration_indexes = np.arange(1, iterations+1)
    
    n_episodes = eval_results["eval_episodes"]

    az_wins = np.zeros(iterations)
    draws = np.zeros(iterations)
    for iter_idx in range(iterations):
        az_wins[iter_idx] += eval_results["results"][str(iter_idx)]["player1_starts"]["win"]
        az_wins[iter_idx] += eval_results["results"][str(iter_idx)]["player2_starts"]["loss"]
        draws[iter_idx] += eval_results["results"][str(iter_idx)]["player1_starts"]["draw"]
        draws[iter_idx] += eval_results["results"][str(iter_idx)]["player2_starts"]["draw"]
    opponent_wins = n_episodes - az_wins - draws
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(iteration_indexes, az_wins, label=f"{model_name} wins", marker="o", color="tab:green")
    plt.plot(iteration_indexes, opponent_wins, label=f"{eval_results['eval_opponent']} wins", marker="o", color="tab:orange")
    plt.plot(iteration_indexes, draws, label="draws", marker="o", color="tab:blue")
    plt.xticks(range(0, iterations+1, x_ticks if x_ticks is not None else 5))
    plt.xlabel("evaluation iteration")
    plt.yticks(range(0, n_episodes+1, 20))
    plt.ylabel("number of games")
    plt.grid()
    plt.legend(loc="center right")

    fig.suptitle(f"Evaluation of {model_name} against {eval_results['eval_opponent'].upper()} player during training over {n_episodes} games", fontweight="bold")
    fig.tight_layout()

    plt.savefig(os.path.join(model_dir, "eval.png"), dpi=300)
    

def main():

    parser = argparse.ArgumentParser("Plot training loss and evaluation results for AlphaZero models.")
    parser.add_argument(
        "--model", 
        "-m", 
        dest="model_name",
        type=str, 
        default="alphazero-othello", 
        help="name of the model to create plots for."
    )
    parser.add_argument(
        "--max-iter", 
        "-i", 
        dest="max_iterations",
        type=int, 
        default=None, 
        help="maximum number of iterations to plot."
    )
    parser.add_argument(
        "--x-ticks", 
        "-x", 
        dest="x_ticks",
        type=int, 
        default=None, 
        help="x-ticks step for plotting."
    )

    args = parser.parse_args()

    # create loss plot
    plot_loss(
        model_name=args.model_name, 
        max_iterations=args.max_iterations,
        x_ticks=args.x_ticks,
    )

    # create evaluation results plot
    plot_eval_results(
        model_name=args.model_name, 
        max_iterations=args.max_iterations,
        x_ticks=args.x_ticks,
    )


if __name__ == "__main__":
    main()