"""Continues training the network from the notebook (models/q_network.pt) with the same online Q-learning update,
and evaluates it greedily every EVAL_EVERY steps on EVAL_EPISODES fresh episodes.

The network with the most evaluation episodes at 500 is saved to models/q_network_continued.pt.

Usage: python continue_training.py [extra steps]
"""
import json
import os
import random
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SEED = 2
EXTRA_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
EVAL_EVERY = 10_000
EVAL_EPISODES = 50
EPS_START = 0.05  # decays linearly to 0 over EXTRA_STEPS
LR = 0.001
GAMMA = 0.99

ROOT = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(ROOT, "models", "q_network.pt")
OUT_PATH = os.path.join(ROOT, "models", "q_network_continued.pt")
HISTORY_PATH = os.path.join(ROOT, "models", "continued_training.json")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Same architecture as QNetwork in the notebook (state dict keys are prefixed with "network.")
net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
net.load_state_dict({k.removeprefix("network."): v for k, v in torch.load(IN_PATH).items()})
optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # the notebook's optimizer state was not saved


def greedy_action(obs):
    with torch.no_grad():
        return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax())


def evaluate(n_episodes, first_seed):
    env = gym.make("CartPole-v1")
    returns = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=first_seed + k)
        done, episode_return = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(greedy_action(obs))
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
    return returns


history = []


def checkpoint(step):
    returns = evaluate(EVAL_EPISODES, 50_000 + step)  # different starting states at every evaluation
    row = dict(step=step, mean=float(np.mean(returns)), n500=int(sum(r >= 500 for r in returns)), min=float(min(returns)))
    history.append(row)
    print(f"step {step:>7}: {row['n500']}/{EVAL_EPISODES} episodes at 500, mean {row['mean']:.1f}", flush=True)
    if row["n500"] > max((h["n500"] for h in history[:-1]), default=-1):
        torch.save({"network." + k: v for k, v in net.state_dict().items()}, OUT_PATH)


env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=SEED + 100)
env.action_space.seed(SEED)  # random exploration actions must be seeded too
checkpoint(0)
for step in range(1, EXTRA_STEPS + 1):
    epsilon = EPS_START * (1 - step / EXTRA_STEPS)
    action = env.action_space.sample() if random.random() < epsilon else greedy_action(obs)
    next_obs, reward, terminated, truncated, _ = env.step(action)

    # Same one-step update as the notebook: TD target from the network itself, squared error, one Adam step
    with torch.no_grad():
        td_target = reward + GAMMA * (1 - float(terminated)) * net(torch.as_tensor(next_obs, dtype=torch.float32)).max()
    pred = net(torch.as_tensor(obs, dtype=torch.float32))[action]
    loss = F.mse_loss(pred, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    obs = next_obs
    if terminated or truncated:
        obs, _ = env.reset()
    if step % EVAL_EVERY == 0:
        checkpoint(step)

json.dump(history, open(HISTORY_PATH, "w"), indent=1)
best = max(history, key=lambda h: h["n500"])
print(f"Best: step {best['step']}, {best['n500']}/{EVAL_EPISODES} at 500. Saved to {OUT_PATH}")
