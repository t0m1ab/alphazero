import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

from alphazero.utils import DEFAULT_MODELS_PATH


def plot_loss(model_name: str, path: str = None):

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

    # store loss values in np.arrays for each iteration to simplify mean calculation
    iter2loss = {iter_idx: np.zeros((2, epochs, steps_per_epoch[iter_idx])) for iter_idx in range(iterations)}
    for iter_idx in range(iterations):
        for epoch_idx in range(epochs):
            iter2loss[iter_idx][0, epoch_idx] = np.array(loss[str(iter_idx)][str(epoch_idx)]["pi"])
            iter2loss[iter_idx][1, epoch_idx] = np.array(loss[str(iter_idx)][str(epoch_idx)]["v"])
    
    # calculate mean loss for each epoch in each iteration
    pi_loss_epochs = np.zeros((iterations, epochs))
    v_loss_epochs = np.zeros((iterations, epochs))
    step_after_epoch = np.zeros((iterations, epochs)) # optimization step index at the end of each epoch
    step_counter = 0 # optimization step index at the beginning of each iteration
    for iter_idx in range(iterations):
        for epoch_idx in range(epochs):
            pi_loss_epochs[iter_idx, epoch_idx] = np.mean(iter2loss[iter_idx][0, epoch_idx])
            v_loss_epochs[iter_idx, epoch_idx] = np.mean(iter2loss[iter_idx][1, epoch_idx])
            step_after_epoch[iter_idx, epoch_idx] = step_counter + (epoch_idx + 1) * steps_per_epoch[iter_idx]
        step_counter = step_after_epoch[iter_idx, -1]
    
    # calculate total loss for each optimization step and each epoch average
    total_loss = all_pi_loss + all_v_loss
    total_loss_epochs = pi_loss_epochs + v_loss_epochs
    
    # print(len(iter2loss[0][0][0]))
    # print(len(all_pi_loss))
    # print(len(all_v_loss))
    # print(pi_loss_epochs.shape)
    # print(v_loss_epochs.shape)
    # print(step_after_epoch.shape)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    ax[0].plot(all_pi_loss, label="pi loss", alpha=0.5, color="tab:blue")
    ax[0].plot(all_v_loss, label="v loss", alpha=0.5, color="tab:orange")
    ax[0].plot(step_after_epoch.reshape(-1), pi_loss_epochs.reshape(-1), label="pi loss (mean over epoch)", marker="o", color="tab:blue")
    ax[0].plot(step_after_epoch.reshape(-1), v_loss_epochs.reshape(-1), label="v loss (mean over epoch)", marker="o", color="tab:orange")
    ax[0].set_ylabel("loss")
    ax[0].grid()
    ax[0].legend(loc="upper right")

    ax[1].plot(total_loss, label="loss", alpha=0.5, color="tab:green")
    ax[1].plot(step_after_epoch.reshape(-1), total_loss_epochs.reshape(-1), label="loss (mean over epoch)", marker="o", color="tab:green")
    ax[1].set_xlabel("optimization step")
    ax[1].set_ylabel("loss")
    ax[1].grid()
    ax[1].legend(loc="upper right")

    fig.suptitle(f"Training loss for {model_name} over {iterations} iterations with {epochs} epochs each", fontweight="bold")
    fig.tight_layout()

    plt.savefig(os.path.join(model_dir, "loss.png"), dpi=300)


def main():
    plot_loss("alphazero-tictactoe")


if __name__ == "__main__":
    main()