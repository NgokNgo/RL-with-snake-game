#!/usr/bin/env python3
import typing as tt
import numpy as np
import gymnasium as gym
from snake_game import SnakeEnv
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter

GAMMA = 0.9
ALPHA = 0.4
ALPHA_DECAY = 0.995
EPSILON = 0.6
EPSILON_DECAY = 0.995
TEST_EPISODES = 20
TRAIN_STEPS_PER_ITER = 100

State = tt.Tuple[int, ...]
Action = int
ValuesKey = tt.Tuple[State, Action]

class Agent:
    def __init__(self):
        self.env = SnakeEnv(width=11, height=11)
        s0, _ = self.env.reset()
        self.state = self.state_to_key(s0)
        self.values: tt.Dict[ValuesKey, float] = defaultdict(float)

    @staticmethod
    def state_to_key(state: tt.Any) -> State:
        if isinstance(state, np.ndarray):
            return tuple(state.astype(np.int8).ravel().tolist())
        if isinstance(state, (list, tuple)):
            return tuple(np.asarray(state).astype(np.int8).ravel().tolist())
        return (int(state),)

    def sample_env(self) -> tt.Tuple[State, Action, float, State]:
        # action = self.env.action_space.sample()
        old_state = self.state

        if np.random.random() < EPSILON:
            action = self.env.action_space.sample()
        else:
            _, action = self.best_value_and_action(old_state)

        new_state_raw, reward, is_done, is_tr, _ = self.env.step(action)
        new_state = self.state_to_key(new_state_raw)

        if is_done or is_tr:
            s_reset, _ = self.env.reset()
            self.state = self.state_to_key(s_reset)
        else:
            self.state = new_state

        return old_state, action, float(reward), new_state

    def best_value_and_action(self, state: State) -> tt.Tuple[float, Action]:
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, state: State, action: Action, reward: float, next_state: State):
        best_val, _ = self.best_value_and_action(next_state)
        new_val = reward + GAMMA * best_val
        old_val = self.values[(state, action)]
        self.values[(state, action)] = old_val * (1 - ALPHA) + new_val * ALPHA

    def play_episode(self, env: gym.Env) -> tt.Tuple[float, int, int]:
        total_reward = 0.0
        score = 0
        episode_length = 0

        state_raw, _ = env.reset()
        state = self.state_to_key(state_raw)

        while True:
            _, action = self.best_value_and_action(state)
            new_state_raw, reward, is_done, is_tr, _ = env.step(action)
            total_reward += reward
            episode_length += 1
            
            score = int(env.snake.score)

            if is_done or is_tr:
                break
            state = self.state_to_key(new_state_raw)

        return total_reward, score, episode_length


if __name__ == "__main__":
    test_env = SnakeEnv(width=11, height=11)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        EPSILON *= EPSILON_DECAY
        ALPHA *= ALPHA_DECAY
        for _ in range(TRAIN_STEPS_PER_ITER):
            state, action, reward, next_state = agent.sample_env()
            agent.value_update(state, action, reward, next_state)

        test_reward = []
        test_scores = []
        test_lengths = []

        for _ in range(TEST_EPISODES):
            reward, score, ep_len = agent.play_episode(test_env)
            test_reward.append(reward)
            test_scores.append(score)
            test_lengths.append(ep_len)
        test_reward = np.mean(test_reward)

        writer.add_scalar("reward", test_reward, iter_no)
        writer.add_scalar("score", np.median(test_scores), iter_no)
        writer.add_scalar("score_max", np.max(test_scores), iter_no)
        writer.add_scalar("score_min", np.min(test_scores), iter_no)
        writer.add_scalar("length", np.median(test_lengths), iter_no)
        writer.add_scalar("length_max", np.max(test_lengths), iter_no)
        writer.add_scalar("length_min", np.min(test_lengths), iter_no)

        if test_reward > best_reward:
            print("%d: Best test reward updated %.3f -> %.3f" % (iter_no, best_reward, test_reward))
            print("Scores: median=%.1f, max=%.1f, min=%.1f" % (np.median(test_scores), np.max(test_scores), np.min(test_scores)))
            print("Episode length: median=%.1f, max=%.1f, min=%.1f" % (np.median(test_lengths), np.max(test_lengths), np.min(test_lengths)))
            best_reward = test_reward
        if test_reward > 10:
            print("Solved in %d iterations!" % iter_no)
            break

    writer.close()