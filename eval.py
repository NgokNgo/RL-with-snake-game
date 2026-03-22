#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
import importlib.util
import inspect
from pathlib import Path
import random
import sys
import typing as tt

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from snake_game import SnakeEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SnakeImageObsWrapper:
    """Convert Snake info map into image-like observation [C, H, W]."""

    def __init__(self, env: SnakeEnv):
        self.env = env
        self.action_space = env.action_space
        height = env.snake.blocks_y
        width = env.snake.blocks_x
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32)

    def _obs_from_info(self, info: dict) -> np.ndarray:
        map_2d = info.get("map", None)
        if map_2d is None:
            height = self.env.snake.blocks_y
            width = self.env.snake.blocks_x
            map_2d = np.zeros((height, width), dtype=np.int8)

        occ = (map_2d == -1).astype(np.float32)
        food = (map_2d == 1).astype(np.float32)

        head = np.zeros_like(occ, dtype=np.float32)
        head_x, head_y = info.get("head", (0, 0))
        if 0 <= head_y < head.shape[0] and 0 <= head_x < head.shape[1]:
            head[head_y, head_x] = 1.0

        return np.stack([occ, food, head], axis=0)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._obs_from_info(info), info

    def step(self, action: int):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._obs_from_info(info), reward, terminated, truncated, info

    def close(self):
        self.env.close()


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deep RL model for Snake")
    parser.add_argument("--model-file", required=True, help="Path to Python file that defines the model class")
    parser.add_argument("--model-class", required=True, help="Model class name inside --model-file")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth/.pt)")
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes")
    parser.add_argument("--hidden-size", type=int, default=256, help="Used when model constructor needs hidden_size")
    parser.add_argument("--width", type=int, default=11, help="Board width")
    parser.add_argument("--height", type=int, default=11, help="Board height")
    parser.add_argument("--render", action="store_true", help="Render environment during eval")
    parser.add_argument(
        "--obs-type",
        default="auto",
        choices=["auto", "vector", "map3ch"],
        help="Observation mode. 'auto' infers from checkpoint metadata.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used to run inference",
    )
    parser.add_argument(
        "--csv",
        default="eval_results.csv",
        help="CSV file path to append evaluation summary",
    )
    parser.add_argument(
        "--model-kwargs",
        default="",
        help="Optional constructor kwargs, format: key=value,key=value",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible evaluation",
    )
    return parser.parse_args()


def parse_model_kwargs(raw: str) -> dict[str, tt.Any]:
    if not raw.strip():
        return {}
    kwargs: dict[str, tt.Any] = {}
    pairs = [part.strip() for part in raw.split(",") if part.strip()]
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid model-kwargs item: {pair}. Use key=value format.")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            kwargs[key] = value.lower() == "true"
            continue
        try:
            kwargs[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            kwargs[key] = float(value)
            continue
        except ValueError:
            pass
        kwargs[key] = value
    return kwargs


def load_model_class(model_file: str, model_class: str) -> type[nn.Module]:
    model_path = Path(model_file)
    module_name = f"eval_model_{model_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model file: {model_file}")
    module = importlib.util.module_from_spec(spec)
    # Register module before execution so decorators like @dataclass can resolve __module__.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, model_class):
        raise AttributeError(f"Class '{model_class}' not found in {model_file}")

    cls = getattr(module, model_class)
    if not inspect.isclass(cls) or not issubclass(cls, nn.Module):
        raise TypeError(f"'{model_class}' is not a torch.nn.Module class")
    return cls


def extract_state_dict(obj: tt.Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return tt.cast(dict[str, torch.Tensor], obj)
    raise ValueError("Unsupported checkpoint format. Expected state_dict or dict with key 'state_dict'.")


def build_model_instance(
    model_cls: type[nn.Module],
    checkpoint: tt.Any,
    obs_shape: tuple[int, ...],
    n_actions: int,
    hidden_size: int,
    extra_kwargs: dict[str, tt.Any],
) -> nn.Module:
    sig = inspect.signature(model_cls.__init__)
    params = set(sig.parameters.keys())
    params.discard("self")

    kwargs = dict(extra_kwargs)
    ckpt = checkpoint if isinstance(checkpoint, dict) else {}

    if "obs_size" in params and "obs_size" not in kwargs:
        kwargs["obs_size"] = int(ckpt.get("obs_size", obs_shape[0]))
    if "hidden_size" in params and "hidden_size" not in kwargs:
        kwargs["hidden_size"] = int(ckpt.get("hidden_size", hidden_size))
    if "n_actions" in params and "n_actions" not in kwargs:
        kwargs["n_actions"] = int(ckpt.get("n_actions", n_actions))
    if "input_shape" in params and "input_shape" not in kwargs:
        if "input_shape" in ckpt:
            kwargs["input_shape"] = tuple(ckpt["input_shape"])
        else:
            kwargs["input_shape"] = obs_shape

    return model_cls(**kwargs)


def load_model(
    model_file: str,
    model_class: str,
    checkpoint: tt.Any,
    obs_shape: tuple[int, ...],
    n_actions: int,
    hidden_size: int,
    extra_kwargs: dict[str, tt.Any],
    device: torch.device,
) -> nn.Module:
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    model_cls = load_model_class(model_file, model_class)
    model = build_model_instance(
        model_cls=model_cls,
        checkpoint=checkpoint,
        obs_shape=obs_shape,
        n_actions=n_actions,
        hidden_size=hidden_size,
        extra_kwargs=extra_kwargs,
    )

    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def pick_action(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=1).item())


def resolve_obs_type(args_obs_type: str, checkpoint: tt.Any) -> str:
    if not isinstance(checkpoint, dict):
        return "vector" if args_obs_type == "auto" else args_obs_type

    inferred = "vector"
    if "input_shape" in checkpoint:
        inferred = "map3ch"
    elif "obs_size" in checkpoint:
        inferred = "vector"

    if args_obs_type == "auto":
        return inferred

    if args_obs_type != inferred and ("input_shape" in checkpoint or "obs_size" in checkpoint):
        print(f"[eval] --obs-type={args_obs_type} mismatched checkpoint; using {inferred} instead.")
        return inferred

    return args_obs_type


def evaluate_policy(
    env,
    model: nn.Module,
    episodes: int,
    device: torch.device,
    seed: int,
) -> tuple[list[float], list[int], list[int]]:
    rewards: list[float] = []
    scores: list[int] = []
    lengths: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_len = 0

        while True:
            obs_v = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(obs_v)
            action = pick_action(logits)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1

            if terminated or truncated:
                score = int(info.get("score", 0))
                rewards.append(ep_reward)
                scores.append(score)
                lengths.append(ep_len)
                break

    return rewards, scores, lengths


def print_stats(name: str, values: list[float]) -> None:
    print(
        f"{name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, "
        f"min={np.min(values):.3f}, max={np.max(values):.3f}"
    )


def summarize(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.max(arr)), float(np.min(arr))


def append_eval_to_csv(
    csv_path: str,
    model_path: str,
    rewards: list[float],
    scores: list[int],
    lengths: list[int],
) -> None:
    csv_file = Path(csv_path)
    model_file = Path(model_path)

    reward_mean, reward_max, reward_min = summarize(rewards)
    score_mean, score_max, score_min = summarize([float(s) for s in scores])
    length_mean, length_max, length_min = summarize([float(l) for l in lengths])

    train_date = datetime.fromtimestamp(model_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    headers = [
        "model_name",
        "train_date",
        "reward_mean",
        "reward_max",
        "reward_min",
        "score_mean",
        "score_max",
        "score_min",
        "episode_length_mean",
        "episode_length_max",
        "episode_length_min",
        "note",
    ]

    row = {
        "model_name": model_file.name,
        "train_date": train_date,
        "reward_mean": f"{reward_mean:.6f}",
        "reward_max": f"{reward_max:.6f}",
        "reward_min": f"{reward_min:.6f}",
        "score_mean": f"{score_mean:.6f}",
        "score_max": f"{score_max:.6f}",
        "score_min": f"{score_min:.6f}",
        "episode_length_mean": f"{length_mean:.6f}",
        "episode_length_max": f"{length_max:.6f}",
        "episode_length_min": f"{length_min:.6f}",
        "note": "",
    }

    csv_file.parent.mkdir(parents=True, exist_ok=True)

    needs_header = True
    if csv_file.exists() and csv_file.stat().st_size > 0:
        with csv_file.open("r", newline="", encoding="utf-8") as f:
            first_line = f.readline().strip()
            needs_header = not first_line

    with csv_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = build_parser()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location=device)
    effective_obs_type = resolve_obs_type(args.obs_type, checkpoint)

    env_base = SnakeEnv(width=args.width, height=args.height, fps=30, render_mode="human" if args.render else None)
    env = SnakeImageObsWrapper(env_base) if effective_obs_type == "map3ch" else env_base

    assert env.observation_space.shape is not None
    obs_shape = tuple(env.observation_space.shape)
    n_actions = int(env.action_space.n)

    model = load_model(
        model_file=args.model_file,
        model_class=args.model_class,
        checkpoint=checkpoint,
        obs_shape=obs_shape,
        n_actions=n_actions,
        hidden_size=args.hidden_size,
        extra_kwargs=parse_model_kwargs(args.model_kwargs),
        device=device,
    )

    rewards, scores, lengths = evaluate_policy(
        env=env,
        model=model,
        episodes=args.episodes,
        device=device,
        seed=args.seed,
    )

    print("Evaluation done")
    print_stats("reward", rewards)
    print_stats("score", [float(s) for s in scores])
    print_stats("episode_length", [float(l) for l in lengths])
    append_eval_to_csv(args.csv, args.weights, rewards, scores, lengths)
    print(f"Saved evaluation summary to: {args.csv}")

    env.close()


if __name__ == "__main__":
    main()
