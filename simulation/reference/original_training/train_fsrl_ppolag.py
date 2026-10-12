#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tianshou.env import DummyVectorEnv

from fsrl.agent import PPOLagAgent
from fsrl.trainer import OnpolicyTrainer
from fsrl.utils import TensorboardLogger
from trainrl import CustomEnv


class SafeCustomEnv(CustomEnv):
    """CustomEnv with an additional per-step safety cost for safe-RL training.

    The reward and terminal conditions remain unchanged. The extra cost is only
    exposed through info["cost"] so safe-RL algorithms can optimize under a
    collision/clearance constraint while remaining comparable at evaluation time.
    """

    def __init__(
        self,
        safety_margin_m: float = 0.03,
        cost_mode: str = "collision_or_margin",
    ):
        super().__init__()
        self.safety_margin_m = max(0.0, float(safety_margin_m))
        self.cost_mode = str(cost_mode)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = dict(info)

        min_lidar_m = float(np.min(self.lidar_readings()))
        unsafe_threshold_m = float(self.COLLISION_DIST + self.safety_margin_m)
        collision = str(info.get("reason", "")) == "collision"
        unsafe_step = float(min_lidar_m < unsafe_threshold_m)
        if self.cost_mode == "collision_only":
            cost = float(collision)
        else:
            cost = 1.0 if collision else unsafe_step

        info["cost"] = float(cost)
        info["unsafe_step_cost"] = float(unsafe_step)
        info["min_lidar_m"] = min_lidar_m
        return obs, reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO-Lagrangian safe-RL baseline with FSRL on CustomEnv.",
    )
    parser.add_argument("--outdir", type=str, default="fsrl_runs/ppolag_collisioncost_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-num", type=int, default=8)
    parser.add_argument("--test-num", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--step-per-epoch", type=int, default=4000)
    parser.add_argument("--episode-per-collect", type=int, default=10)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--testing-num", type=int, default=10)
    parser.add_argument("--cost-limit", type=float, default=5.0)
    parser.add_argument("--safety-margin-m", type=float, default=0.03)
    parser.add_argument(
        "--cost-mode",
        type=str,
        default="collision_or_margin",
        choices=["collision_or_margin", "collision_only"],
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=-1.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--thread", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reward-threshold", type=float, default=1e9)
    return parser.parse_args()


def make_env_factory(
    base_seed: int,
    env_index: int,
    safety_margin_m: float,
    cost_mode: str,
) -> Callable[[], SafeCustomEnv]:
    def _factory() -> SafeCustomEnv:
        env = SafeCustomEnv(safety_margin_m=safety_margin_m, cost_mode=cost_mode)
        env.rng = np.random.default_rng(int(base_seed + env_index))
        return env

    return _factory


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_MODE", "disabled")

    config = {
        "algorithm": "PPOLagAgent",
        "seed": int(args.seed),
        "train_num": int(args.train_num),
        "test_num": int(args.test_num),
        "epoch": int(args.epoch),
        "max_train_episodes": int(args.max_train_episodes),
        "step_per_epoch": int(args.step_per_epoch),
        "episode_per_collect": int(args.episode_per_collect),
        "repeat_per_collect": int(args.repeat_per_collect),
        "batch_size": int(args.batch_size),
        "buffer_size": int(args.buffer_size),
        "testing_num": int(args.testing_num),
        "cost_limit": float(args.cost_limit),
        "safety_margin_m": float(args.safety_margin_m),
        "cost_mode": str(args.cost_mode),
        "lr": float(args.lr),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "eps_clip": float(args.eps_clip),
        "vf_coef": float(args.vf_coef),
        "max_grad_norm": None if float(args.max_grad_norm) < 0 else float(args.max_grad_norm),
        "hidden_sizes": [int(v) for v in args.hidden_sizes],
        "save_interval": int(args.save_interval),
        "thread": int(args.thread),
        "device": args.device,
        "reward_threshold": float(args.reward_threshold),
    }
    with open(outdir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    train_envs = DummyVectorEnv(
        [
            make_env_factory(
                args.seed * 1000,
                env_index,
                args.safety_margin_m,
                args.cost_mode,
            )
            for env_index in range(args.train_num)
        ]
    )
    test_envs = None
    if args.test_num > 0:
        test_envs = DummyVectorEnv(
            [
                make_env_factory(
                    args.seed * 2000 + 10000,
                    env_index,
                    args.safety_margin_m,
                    args.cost_mode,
                )
                for env_index in range(args.test_num)
            ]
        )

    logger = TensorboardLogger(log_dir=str(outdir), log_txt=True, name="ppolag")
    agent = PPOLagAgent(
        env=SafeCustomEnv(
            safety_margin_m=args.safety_margin_m,
            cost_mode=args.cost_mode,
        ),
        logger=logger,
        cost_limit=args.cost_limit,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        max_grad_norm=None if float(args.max_grad_norm) < 0 else float(args.max_grad_norm),
        hidden_sizes=tuple(int(v) for v in args.hidden_sizes),
    )

    def checkpoint_fn():
        return {"model": agent.state_dict}

    logger.setup_checkpoint_fn(checkpoint_fn)

    if isinstance(train_envs, DummyVectorEnv):
        from fsrl.data import FastCollector
        from tianshou.data import VectorReplayBuffer

        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        train_collector = FastCollector(agent.policy, train_envs, buffer, exploration_noise=True)
        test_collector = FastCollector(agent.policy, test_envs) if test_envs is not None else None
    else:
        raise RuntimeError("Expected DummyVectorEnv for train_envs.")

    def stop_fn(reward, cost):
        return reward > args.reward_threshold and cost < args.cost_limit

    trainer = OnpolicyTrainer(
        policy=agent.policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_limit,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_collect=args.episode_per_collect,
        episode_per_test=args.testing_num if test_envs is not None else 0,
        stop_fn=stop_fn if test_envs is not None else None,
        logger=logger,
        resume_from_log=False,
        save_model_interval=args.save_interval,
        verbose=True,
        show_progress=True,
    )

    last_epoch = 0
    last_info = {}
    last_stats = {}
    for epoch, epoch_stat, info in trainer:
        last_epoch = int(epoch)
        last_stats = dict(epoch_stat)
        last_info = dict(info)
        if args.max_train_episodes > 0 and trainer.cum_episode >= int(args.max_train_episodes):
            print(
                f"[F-SRL] Reached max_train_episodes={args.max_train_episodes} "
                f"at epoch={epoch}, cum_episode={trainer.cum_episode}, env_step={trainer.env_step}."
            )
            break

    torch.save(agent.policy.state_dict(), outdir / "policy.pt")
    with open(outdir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "stopped_by_max_train_episodes": bool(args.max_train_episodes > 0 and trainer.cum_episode >= int(args.max_train_episodes)),
                "cum_episode": int(trainer.cum_episode),
                "env_step": int(trainer.env_step),
                "last_epoch": int(last_epoch),
                "last_epoch_stats": last_stats,
                "last_info": last_info,
                "policy_path": str((outdir / "policy.pt").resolve()),
            },
            handle,
            indent=2,
        )
    train_envs.close()
    if test_envs is not None:
        test_envs.close()


if __name__ == "__main__":
    main()
