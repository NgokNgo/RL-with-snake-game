#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from snake_game import SnakeEnv


GAMMA = 0.98
LR = 8e-4
LR_END = 1e-5
LR_DECAY = 0.995
BATCH_SIZE = 128
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000
TARGET_UPDATE_EVERY = 50
MAX_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 99999

EPS_START = 1.0
EPS_END = 0.25
EPS_DECAY = 0.9985

EVAL_EVERY = 25
EVAL_EPISODES = 50


class DQNMLP(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(policy_net: DQNMLP, obs: np.ndarray, epsilon: float, n_actions: int, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)

    with torch.no_grad():
        obs_v = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_vals = policy_net(obs_v)
        return int(torch.argmax(q_vals, dim=1).item())


def train_step(
    policy_net: DQNMLP,
    target_net: DQNMLP,
    replay: ReplayBuffer,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    batch = replay.sample(BATCH_SIZE)

    obs = np.asarray([t.obs for t in batch], dtype=np.float32)
    actions = np.asarray([t.action for t in batch], dtype=np.int64)
    rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
    next_obs = np.asarray([t.next_obs for t in batch], dtype=np.float32)
    dones = np.asarray([t.done for t in batch], dtype=np.float32)

    obs_v = torch.tensor(obs, dtype=torch.float32, device=device)
    actions_v = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
    rewards_v = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_obs_v = torch.tensor(next_obs, dtype=torch.float32, device=device)
    dones_v = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(obs_v).gather(1, actions_v).squeeze(1)

    with torch.no_grad():
        # next_q_values = target_net(next_obs_v).max(dim=1).values
        
        next_actions = policy_net(next_obs_v).argmax(dim=1).unsqueeze(-1)
        next_q_values = target_net(next_obs_v).gather(1, next_actions).squeeze(1)
        
        target_values = rewards_v + GAMMA * next_q_values * (1.0 - dones_v)

    loss = nn.SmoothL1Loss()(q_values, target_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def evaluate(env: SnakeEnv, policy_net: DQNMLP, device: torch.device, episodes: int) -> tuple[float, float, float]:
    rewards: list[float] = []
    scores: list[float] = []
    lengths: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            with torch.no_grad():
                obs_v = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(policy_net(obs_v), dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                rewards.append(total_reward)
                scores.append(float(info.get("score", 0.0)))
                lengths.append(float(steps))
                break

    return float(np.mean(rewards)), float(np.mean(scores)), float(np.mean(lengths))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_env = SnakeEnv(width=11, height=11)
    eval_env = SnakeEnv(width=11, height=11)

    assert train_env.observation_space.shape is not None
    obs_size = int(train_env.observation_space.shape[0])
    n_actions = int(train_env.action_space.n)

    policy_net = DQNMLP(obs_size=obs_size, n_actions=n_actions).to(device)
    target_net = DQNMLP(obs_size=obs_size, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(capacity=BUFFER_SIZE)

    writer = SummaryWriter(comment="-snake-dqn-mlp")
    run_dir = Path(writer.log_dir)
    latest_model_path = run_dir / "dqn_mlp_latest.pth"
    best_model_path = run_dir / "dqn_mlp_best.pth"
    best_score_model_path = run_dir / "dqn_mlp_best_score.pth"
    hparams_path = run_dir / "hparams.json"

    hparams = {
        "algo": "dqn_double",
        "gamma": GAMMA,
        "lr": LR,
        "lr_end": LR_END,
        "lr_decay": LR_DECAY,
        "batch_size": BATCH_SIZE,
        "buffer_size": BUFFER_SIZE,
        "min_replay_size": MIN_REPLAY_SIZE,
        "target_update_every": TARGET_UPDATE_EVERY,
        "max_episodes": MAX_EPISODES,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "eps_decay": EPS_DECAY,
        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,
        "env": {
            "width": 11,
            "height": 11,
        },
        "model": {
            "name": "DQNMLP",
            "hidden_size": 256,
        },
    }
    hparams_path.write_text(json.dumps(hparams, indent=2), encoding="utf-8")

    epsilon = EPS_START
    current_lr = LR
    best_eval_reward = -float("inf")
    best_eval_score = -float("inf")
    total_steps = 0
    learning_flag = False

    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = train_env.reset()
        ep_reward = 0.0
        ep_len = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = select_action(policy_net, obs, epsilon, n_actions, device)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            replay.add(
                Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=float(done),
                )
            )

            obs = next_obs
            ep_reward += reward
            ep_len += 1
            total_steps += 1

            if len(replay) >= MIN_REPLAY_SIZE:
                learning_flag = True
                loss = train_step(policy_net, target_net, replay, optimizer, device)
                writer.add_scalar("train/loss", loss, total_steps)

            if total_steps % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        
        if learning_flag:
            epsilon = max(EPS_END, epsilon * EPS_DECAY)
            # current_lr = max(LR_END, current_lr * LR_DECAY)
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = current_lr

        writer.add_scalar("train/episode_reward", ep_reward, episode)
        writer.add_scalar("train/episode_length", ep_len, episode)
        writer.add_scalar("train/epsilon", epsilon, episode)
        writer.add_scalar("train/lr", current_lr, episode)

        if learning_flag and episode % EVAL_EVERY == 0:
            eval_reward, eval_score, eval_length = evaluate(eval_env, policy_net, device, EVAL_EPISODES)
            writer.add_scalar("eval/reward_mean", eval_reward, episode)
            writer.add_scalar("eval/score_mean", eval_score, episode)
            writer.add_scalar("eval/length_mean", eval_length, episode)

            torch.save(
                {
                    "state_dict": policy_net.state_dict(),
                    "episode": episode,
                    "total_steps": total_steps,
                    "epsilon": epsilon,
                    "lr": current_lr,
                    "eval_reward": eval_reward,
                    "obs_size": obs_size,
                    "n_actions": n_actions,
                },
                latest_model_path,
            )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(
                    {
                        "state_dict": policy_net.state_dict(),
                        "episode": episode,
                        "total_steps": total_steps,
                        "epsilon": epsilon,
                        "lr": current_lr,
                        "eval_reward": eval_reward,
                        "obs_size": obs_size,
                        "n_actions": n_actions,
                    },
                    best_model_path,
                )

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                torch.save(
                    {
                        "state_dict": policy_net.state_dict(),
                        "episode": episode,
                        "total_steps": total_steps,
                        "epsilon": epsilon,
                        "lr": current_lr,
                        "eval_reward": eval_reward,
                        "eval_score": eval_score,
                        "obs_size": obs_size,
                        "n_actions": n_actions,
                    },
                    best_score_model_path,
                )

            print(
                f"Episode {episode} | train_reward={ep_reward:.2f} | "
                f"eval_reward={eval_reward:.2f} | eval_score={eval_score:.2f} | "
                f"epsilon={epsilon:.3f} | lr={current_lr:.6f}"
            )

    print(f"Saved latest model to: {latest_model_path}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved best score model to: {best_score_model_path}")
    print(f"Saved hyperparameters to: {hparams_path}")

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
