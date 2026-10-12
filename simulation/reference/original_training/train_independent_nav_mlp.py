#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import pickle
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from baseline_eval_suite import (
    DEFAULT_DECISION,
    DEFAULT_MODEL_CANDIDATES,
    DT,
    EvaluationEnv,
    RLOnlyController,
    SUPPORTED_ENV_VARIANTS,
    SupervisorController,
    build_nav_context_from_obs,
    independent_template_decision,
    is_safe_nav_context,
    nav_context_to_features,
    resolve_existing_path,
    sanitize_decision,
)


class DecisionApplicator(SupervisorController):
    def __init__(self, decision_interval: int):
        super().__init__(name="decision_applicator", decision_interval=decision_interval)

    def infer_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        return dict(DEFAULT_DECISION)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a non-LLM nav-context MLP using short-horizon search supervision.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to PPO model zip. Defaults to first existing local candidate.",
    )
    parser.add_argument("--outdir", type=str, default="independent_nav_mlp_runs/search_supervised_k30_v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--decision-interval", type=int, default=30)
    parser.add_argument("--search-horizon", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=2500)
    parser.add_argument("--max-decisions-per-episode", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--env-variant",
        type=str,
        default="baseline",
        choices=sorted(SUPPORTED_ENV_VARIANTS),
    )
    return parser.parse_args()


def snapshot_env(env: EvaluationEnv) -> Dict[str, object]:
    return {
        "agent_pos": env.agent_pos.copy(),
        "target_pos": env.target_pos.copy(),
        "distance_buffer": deque(list(env.distance_buffer), maxlen=env.distance_buffer.maxlen),
        "MIN_LIDAR": float(env.MIN_LIDAR),
        "MAX_LIDAR": float(env.MAX_LIDAR),
        "path": [pt.copy() for pt in env.path],
        "walls": [tuple(w) for w in env.walls],
        "dynamic_walls": [tuple(w) for w in env.dynamic_walls],
        "all_targets": [np.array(t, dtype=np.float32).copy() for t in env.all_targets],
        "target_outcomes": list(env.target_outcomes),
        "step_count": int(env.step_count),
        "accumulated_distance": float(env.accumulated_distance),
        "distance_to_target_start": float(getattr(env, "distance_to_target_start", 0.0)),
        "_static_variant_walls": [tuple(w) for w in getattr(env, "_static_variant_walls", [])],
        "_moving_obstacles": [
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in obstacle.items()}
            for obstacle in getattr(env, "_moving_obstacles", [])
        ],
        "active_env_variant": str(getattr(env, "active_env_variant", env.env_variant)),
    }


def restore_env(env: EvaluationEnv, snapshot: Dict[str, object]) -> None:
    env.agent_pos = np.asarray(snapshot["agent_pos"], dtype=np.float32).copy()
    env.target_pos = np.asarray(snapshot["target_pos"], dtype=np.float32).copy()
    env.distance_buffer = deque(list(snapshot["distance_buffer"]), maxlen=snapshot["distance_buffer"].maxlen)
    env.MIN_LIDAR = float(snapshot["MIN_LIDAR"])
    env.MAX_LIDAR = float(snapshot["MAX_LIDAR"])
    env.path = [np.asarray(pt, dtype=np.float32).copy() for pt in snapshot["path"]]
    env.walls = [tuple(map(float, wall)) for wall in snapshot["walls"]]
    env.dynamic_walls = [tuple(map(float, wall)) for wall in snapshot["dynamic_walls"]]
    env.all_targets = [np.asarray(t, dtype=np.float32).copy() for t in snapshot["all_targets"]]
    env.target_outcomes = list(snapshot["target_outcomes"])
    env.step_count = int(snapshot["step_count"])
    env.accumulated_distance = float(snapshot["accumulated_distance"])
    env.distance_to_target_start = float(snapshot["distance_to_target_start"])
    env._static_variant_walls = [tuple(map(float, wall)) for wall in snapshot["_static_variant_walls"]]
    env._moving_obstacles = [
        {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in obstacle.items()}
        for obstacle in snapshot["_moving_obstacles"]
    ]
    env.active_env_variant = str(snapshot["active_env_variant"])


def rollout_score(
    env: EvaluationEnv,
    model: PPO,
    applicator: DecisionApplicator,
    obs: np.ndarray,
    nav_context: Dict[str, float],
    template_name: str,
    horizon: int,
) -> float:
    decision = sanitize_decision(independent_template_decision(template_name, nav_context), nav_context=nav_context)
    applicator.last_nav_context = nav_context

    start_distance = float(np.linalg.norm(env.agent_pos[0:2] - env.target_pos))
    done = False
    last_reason = ""

    local_obs = np.asarray(obs, dtype=np.float32).copy()
    for _ in range(horizon):
        rl_action, _ = model.predict(local_obs, deterministic=True)
        rl_action = np.asarray(rl_action, dtype=np.float32).reshape(-1)
        applied_action = applicator.apply_decision(rl_action, decision)
        local_obs, _, terminated, truncated, info = env.step(applied_action)
        done = bool(terminated or truncated)
        last_reason = str(info.get("reason", ""))
        if done:
            break

    final_distance = float(np.linalg.norm(env.agent_pos[0:2] - env.target_pos))
    progress = max(0.0, start_distance - final_distance)
    score = progress
    if last_reason == "target_reached":
        score += 5.0
    elif last_reason == "collision":
        score -= 5.0
    elif last_reason == "max_steps_reached":
        score -= 2.0
    return float(score)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else resolve_existing_path(DEFAULT_MODEL_CANDIDATES)
    )
    model = PPO.load(str(model_path), device="cpu")
    env = EvaluationEnv(env_variant=args.env_variant)
    applicator = DecisionApplicator(decision_interval=args.decision_interval)
    template_names = list(
        (
            "follow_default",
            "cautious_soft_align",
            "cautious_hard_align",
            "explore_soft_align",
            "explore_hard_align",
            "stop_align",
        )
    )

    rng = np.random.default_rng(args.random_state)
    features: List[np.ndarray] = []
    labels: List[str] = []
    metadata_rows: List[Dict[str, object]] = []

    for episode_idx in range(args.episodes):
        if len(features) >= args.max_samples:
            break

        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        steps = 0
        done = False
        decision_points_this_episode = 0
        active_decision = dict(DEFAULT_DECISION)

        while not done:
            nav_context = build_nav_context_from_obs(obs, steps, episode_idx)
            if (
                steps % args.decision_interval == 0
                and not is_safe_nav_context(nav_context)
                and decision_points_this_episode < args.max_decisions_per_episode
                and len(features) < args.max_samples
            ):
                snapshot = snapshot_env(env)
                template_scores: Dict[str, float] = {}
                for template_name in template_names:
                    restore_env(env, snapshot)
                    template_scores[template_name] = rollout_score(
                        env=env,
                        model=model,
                        applicator=applicator,
                        obs=obs,
                        nav_context=nav_context,
                        template_name=template_name,
                        horizon=args.search_horizon,
                    )
                restore_env(env, snapshot)

                best_template = max(template_scores.items(), key=lambda item: item[1])[0]
                features.append(nav_context_to_features(nav_context))
                labels.append(best_template)
                metadata_rows.append(
                    {
                        "episode": int(episode_idx),
                        "step": int(steps),
                        "best_template": best_template,
                        "scores": {k: float(v) for k, v in template_scores.items()},
                        "nav_context": nav_context,
                    }
                )
                active_decision = sanitize_decision(
                    independent_template_decision(best_template, nav_context),
                    nav_context=nav_context,
                )
                decision_points_this_episode += 1

            rl_action, _ = model.predict(obs, deterministic=True)
            rl_action = np.asarray(rl_action, dtype=np.float32).reshape(-1)
            applicator.last_nav_context = nav_context
            applied_action = applicator.apply_decision(rl_action, active_decision)
            obs, _, terminated, truncated, _ = env.step(applied_action)
            steps += 1
            done = bool(terminated or truncated)

        if (episode_idx + 1) % 25 == 0:
            print(
                f"[INDEPENDENT_MLP] episode={episode_idx + 1}/{args.episodes} "
                f"samples={len(features)}/{args.max_samples}"
            )

    env.close()

    if not features:
        raise RuntimeError("No training samples were collected for independent nav MLP.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=object)

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    max_iter=600,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    classifier.fit(X, y)

    class_counts = Counter(str(label) for label in labels)
    training_summary = {
        "method": "search_supervised_independent_mlp",
        "model_path": str(model_path),
        "env_variant": args.env_variant,
        "episodes_scanned": int(min(args.episodes, episode_idx + 1)),
        "decision_interval": int(args.decision_interval),
        "search_horizon": int(args.search_horizon),
        "training_samples": int(len(X)),
        "template_names": template_names,
        "class_counts": dict(sorted(class_counts.items())),
    }

    payload = {
        "model": classifier,
        "template_names": template_names,
        "training_summary": training_summary,
    }

    with open(outdir / "model.pkl", "wb") as handle:
        pickle.dump(payload, handle)
    with open(outdir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)
    with open(outdir / "sample_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata_rows, handle, indent=2)

    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()
