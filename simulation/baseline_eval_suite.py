#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import pickle
import re
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings(
    "ignore",
    message="A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy",
)

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import binomtest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from trainrl import CustomEnv  # noqa: E402


DT = 0.2
LIDAR_MIN_RANGE_M = 0.16
LIDAR_MAX_RANGE_M = 1.00
LIDAR_RANGE_SPAN_M = LIDAR_MAX_RANGE_M - LIDAR_MIN_RANGE_M
LIDAR_REL_ANGLES_RAD = ((CustomEnv.LIDAR_ANGLES + np.pi) % (2.0 * np.pi)) - np.pi
DEFAULT_DECISION = {
    "mode": "follow",
    "speed_scale": 1.0,
    "turn_scale": 1.0,
    "turn_bias": 0.0,
}
INDEPENDENT_DECISION_TEMPLATE_NAMES = (
    "follow_default",
    "cautious_soft_align",
    "cautious_hard_align",
    "explore_soft_align",
    "explore_hard_align",
    "stop_align",
)
SUPPORTED_MODES = {"follow", "cautious", "explore", "stop"}
DEFAULT_MODEL_CANDIDATES = [
    "checkpoints_rl_only/ppo_static_shaping_model.zip",
    os.path.expanduser("~/rl-rlvellm-llm/checkpoints_rl_only/ppo_static_shaping_model.zip"),
]
DEFAULT_TEACHER_TRACE = "30json/local30.json"
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_LOCAL_LLM_MODEL = "llama3.2:3b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_BASE_URL = "http://localhost:11434"
SUPPORTED_ENV_VARIANTS = {
    "baseline",
    "ood_dense",
    "ood_l_walls",
    "ood_u_traps",
    "ood_corridor",
    "ood_dynamic_crossing",
    "ood_mixed",
}
SUPPORTED_BEAM_DROPOUT_FILLS = {"max", "min", "mean"}


def resolve_existing_path(candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No existing file found in candidates: {list(candidates)}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def default_llm_model(provider: str) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_MODEL
    return DEFAULT_LOCAL_LLM_MODEL


def linear_velocity_from_action(action: np.ndarray) -> float:
    return (float(action[0]) + 1.0) * 0.11


def action0_from_linear_velocity(linear_velocity_m_s: float) -> float:
    return float(np.clip((float(linear_velocity_m_s) / 0.11) - 1.0, -1.0, 1.0))


def angular_velocity_from_action(action: np.ndarray) -> float:
    return float(action[1]) * 2.84


def action1_from_angular_velocity(angular_velocity_rad_s: float) -> float:
    return float(np.clip(float(angular_velocity_rad_s) / 2.84, -1.0, 1.0))


def normalized_lidar_to_distance_m(lidar_norm: np.ndarray) -> np.ndarray:
    lidar_norm = np.asarray(lidar_norm, dtype=np.float32)
    clipped = np.clip(lidar_norm, -1.0, 1.0)
    return LIDAR_MIN_RANGE_M + 0.5 * (clipped + 1.0) * LIDAR_RANGE_SPAN_M


def min_lidar_norm_to_distance_m(min_lidar_norm: float) -> float:
    return float(normalized_lidar_to_distance_m(np.asarray([min_lidar_norm], dtype=np.float32))[0])


def distance_m_to_normalized_lidar(lidar_m: np.ndarray) -> np.ndarray:
    lidar_m = np.asarray(lidar_m, dtype=np.float32)
    clipped = np.clip(lidar_m, LIDAR_MIN_RANGE_M, LIDAR_MAX_RANGE_M)
    return np.clip(((clipped - LIDAR_MIN_RANGE_M) / LIDAR_RANGE_SPAN_M) * 2.0 - 1.0, -1.0, 1.0)


def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    return (np.asarray(angle_rad, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi


def heading_sign_from_nav_context(nav_context: Dict[str, float]) -> float:
    heading_deg = float(nav_context.get("heading_deg_to_target", 0.0))
    if abs(heading_deg) < 5.0:
        return 0.0
    return 1.0 if heading_deg > 0 else -1.0


def independent_template_decision(
    template_name: str,
    nav_context: Dict[str, float],
) -> Dict[str, float]:
    heading_sign = heading_sign_from_nav_context(nav_context)
    if template_name == "follow_default":
        return dict(DEFAULT_DECISION)
    if template_name == "cautious_soft_align":
        return {
            "mode": "cautious",
            "speed_scale": 0.55,
            "turn_scale": 1.05,
            "turn_bias": 0.06 * heading_sign,
        }
    if template_name == "cautious_hard_align":
        return {
            "mode": "cautious",
            "speed_scale": 0.35,
            "turn_scale": 1.30,
            "turn_bias": 0.12 * heading_sign,
        }
    if template_name == "explore_soft_align":
        return {
            "mode": "explore",
            "speed_scale": 0.85,
            "turn_scale": 1.15,
            "turn_bias": 0.07 * heading_sign,
        }
    if template_name == "explore_hard_align":
        return {
            "mode": "explore",
            "speed_scale": 0.72,
            "turn_scale": 1.35,
            "turn_bias": 0.12 * heading_sign,
        }
    if template_name == "stop_align":
        return {
            "mode": "stop",
            "speed_scale": 0.0,
            "turn_scale": 1.30,
            "turn_bias": 0.15 * heading_sign,
        }
    raise ValueError(f"Unsupported independent template: {template_name}")


class EvaluationEnv(CustomEnv):
    OOD_MIXED_CHOICES = ("ood_dense", "ood_l_walls", "ood_u_traps", "ood_corridor")

    def __init__(
        self,
        *,
        env_variant: str = "baseline",
        lidar_noise_std: float = 0.0,
        beam_dropout_rate: float = 0.0,
        beam_dropout_fill: str = "max",
    ):
        super().__init__()
        if env_variant not in SUPPORTED_ENV_VARIANTS:
            raise ValueError(f"Unsupported env_variant: {env_variant}")
        if beam_dropout_fill not in SUPPORTED_BEAM_DROPOUT_FILLS:
            raise ValueError(f"Unsupported beam_dropout_fill: {beam_dropout_fill}")

        self.env_variant = str(env_variant)
        self.active_env_variant = str(env_variant)
        self.lidar_noise_std = max(0.0, float(lidar_noise_std))
        self.beam_dropout_rate = min(max(float(beam_dropout_rate), 0.0), 1.0)
        self.beam_dropout_fill = str(beam_dropout_fill)
        self._static_variant_walls: List[Tuple[float, float, float, float]] = []
        self._moving_obstacles: List[Dict[str, float]] = []

    def _clear_dynamic_motion_state(self) -> None:
        self._static_variant_walls = []
        self._moving_obstacles = []

    def _wall_from_rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> Tuple[float, float, float, float]:
        return (
            float(x),
            float(y),
            float(x + width),
            float(y + height),
        )

    def _moving_obstacle_wall(self, obstacle: Dict[str, float]) -> Tuple[float, float, float, float]:
        return self._wall_from_rect(
            obstacle["x"],
            obstacle["y"],
            obstacle["width"],
            obstacle["height"],
        )

    def _moving_obstacle_sweep(self, obstacle: Dict[str, float]) -> Tuple[float, float, float, float]:
        axis = str(obstacle["axis"])
        if axis == "x":
            return self._wall_from_rect(
                obstacle["low"],
                obstacle["y"],
                (obstacle["high"] - obstacle["low"]) + obstacle["width"],
                obstacle["height"],
            )
        return self._wall_from_rect(
            obstacle["x"],
            obstacle["low"],
            obstacle["width"],
            (obstacle["high"] - obstacle["low"]) + obstacle["height"],
        )

    def _refresh_dynamic_walls(self) -> None:
        dynamic_walls: List[Tuple[float, float, float, float]] = list(self._static_variant_walls)
        for obstacle in self._moving_obstacles:
            dynamic_walls.append(self._moving_obstacle_wall(obstacle))
        self.dynamic_walls = dynamic_walls

    def _commit_static_layout(self) -> None:
        self._static_variant_walls = [tuple(float(v) for v in wall) for wall in self.dynamic_walls]
        self._refresh_dynamic_walls()

    def _advance_dynamic_obstacles(self, dt_s: float) -> None:
        if not self._moving_obstacles:
            return

        for obstacle in self._moving_obstacles:
            axis = str(obstacle["axis"])
            low = float(obstacle["low"])
            high = float(obstacle["high"])
            direction = float(obstacle["direction"])
            speed = float(obstacle["speed"])
            position = float(obstacle[axis]) + direction * speed * dt_s

            while position < low or position > high:
                if position < low:
                    position = low + (low - position)
                    direction *= -1.0
                elif position > high:
                    position = high - (position - high)
                    direction *= -1.0

            obstacle[axis] = float(position)
            obstacle["direction"] = float(direction)

        self._refresh_dynamic_walls()

    def _sample_moving_crossing_obstacle(self) -> Optional[Dict[str, float]]:
        t = self.wall_thickness
        for _ in range(80):
            if float(self.rng.random()) < 0.5:
                axis = "x"
                width = t
                height = float(self.rng.uniform(0.65, 1.05))
                y = float(self.rng.uniform(-1.5, 1.5 - height))
                low = float(self.rng.uniform(-2.9, -1.3))
                high = float(self.rng.uniform(1.3, 2.9 - width))
                x = float(self.rng.uniform(low, high))
            else:
                axis = "y"
                width = float(self.rng.uniform(0.65, 1.05))
                height = t
                x = float(self.rng.uniform(-1.5, 1.5 - width))
                low = float(self.rng.uniform(-2.9, -1.3))
                high = float(self.rng.uniform(1.3, 2.9 - height))
                y = float(self.rng.uniform(low, high))

            if axis == "x":
                sweep = self._wall_from_rect(low, y, (high - low) + width, height)
            else:
                sweep = self._wall_from_rect(x, low, width, (high - low) + height)

            if not self._wall_within_bounds(sweep):
                continue
            if not self._agent_clear_of_wall(sweep, padding=0.45):
                continue

            conflict = False
            for existing_wall in self._static_variant_walls:
                if self.distance_between_walls(sweep, existing_wall) < 0.55:
                    conflict = True
                    break
            if not conflict:
                for existing_obstacle in self._moving_obstacles:
                    if self.distance_between_walls(sweep, self._moving_obstacle_sweep(existing_obstacle)) < 0.8:
                        conflict = True
                        break
            if conflict:
                continue

            return {
                "axis": axis,
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "low": float(low),
                "high": float(high),
                "direction": float(-1.0 if float(self.rng.random()) < 0.5 else 1.0),
                "speed": float(self.rng.uniform(0.20, 0.38)),
            }

        return None

    def _wall_within_bounds(self, wall: Sequence[float]) -> bool:
        low = -5.0 + self.wall_thickness - 1e-6
        high = 5.0 + 1e-6
        x1, y1, x2, y2 = [float(v) for v in wall]
        return (
            low <= x1 < x2 <= high
            and low <= y1 < y2 <= high
        )

    def _agent_clear_of_wall(self, wall: Sequence[float], padding: float = 0.35) -> bool:
        x1, y1, x2, y2 = [float(v) for v in wall]
        return not (
            (x1 - padding <= float(self.agent_pos[0]) <= x2 + padding)
            and (y1 - padding <= float(self.agent_pos[1]) <= y2 + padding)
        )

    def _is_valid_wall_position_relaxed(
        self,
        wall: Sequence[float],
        *,
        min_dist: float = 0.8,
        agent_padding: float = 0.35,
    ) -> bool:
        if not self._wall_within_bounds(wall):
            return False
        if not self._agent_clear_of_wall(wall, padding=agent_padding):
            return False
        for existing_wall in self.dynamic_walls:
            if self.distance_between_walls(wall, existing_wall) < min_dist:
                return False
        return True

    def _can_place_group(
        self,
        walls: Sequence[Sequence[float]],
        *,
        min_dist: float = 0.8,
        agent_padding: float = 0.35,
    ) -> bool:
        for wall in walls:
            if not self._wall_within_bounds(wall):
                return False
            if not self._agent_clear_of_wall(wall, padding=agent_padding):
                return False
            for existing_wall in self.dynamic_walls:
                if self.distance_between_walls(wall, existing_wall) < min_dist:
                    return False
        return True

    def _try_add_group(
        self,
        walls: Sequence[Sequence[float]],
        *,
        min_dist: float = 0.8,
        agent_padding: float = 0.35,
    ) -> bool:
        if not self._can_place_group(walls, min_dist=min_dist, agent_padding=agent_padding):
            return False
        self.dynamic_walls.extend(
            [tuple(float(v) for v in wall) for wall in walls]
        )
        return True

    def _add_extra_random_walls(
        self,
        target_n: int,
        *,
        min_dist: float = 0.8,
        min_length: float = 0.6,
        max_length: float = 1.5,
        max_attempts: int = 1000,
    ) -> None:
        attempts = 0
        while target_n > 0 and attempts < max_attempts:
            axis = "x" if float(self.rng.random()) < 0.5 else "y"
            length = float(self.rng.uniform(min_length, max_length))
            wall = self.random_wall(length, axis=axis)
            if self._is_valid_wall_position_relaxed(wall, min_dist=min_dist):
                self.dynamic_walls.append(tuple(float(v) for v in wall))
                target_n -= 1
            attempts += 1

    def _sample_l_group(self) -> List[Tuple[float, float, float, float]]:
        t = self.wall_thickness
        orientation = int(self.rng.integers(4))
        h_len = float(self.rng.uniform(1.2, 2.5))
        v_len = float(self.rng.uniform(1.2, 2.5))

        if orientation == 0:  # right + up
            x = float(self.rng.uniform(-5.0 + t, 5.0 - h_len))
            y = float(self.rng.uniform(-5.0 + t, 5.0 - v_len))
            walls = [(x, y, x + h_len, y + t), (x, y, x + t, y + v_len)]
        elif orientation == 1:  # left + up
            x = float(self.rng.uniform(-5.0 + h_len, 5.0 - t))
            y = float(self.rng.uniform(-5.0 + t, 5.0 - v_len))
            walls = [(x - h_len, y, x, y + t), (x - t, y, x, y + v_len)]
        elif orientation == 2:  # left + down
            x = float(self.rng.uniform(-5.0 + h_len, 5.0 - t))
            y = float(self.rng.uniform(-5.0 + v_len, 5.0 - t))
            walls = [(x - h_len, y - t, x, y), (x - t, y - v_len, x, y)]
        else:  # right + down
            x = float(self.rng.uniform(-5.0 + t, 5.0 - h_len))
            y = float(self.rng.uniform(-5.0 + v_len, 5.0 - t))
            walls = [(x, y - t, x + h_len, y), (x, y - v_len, x + t, y)]

        return [tuple(float(v) for v in wall) for wall in walls]

    def _sample_u_group(self) -> List[Tuple[float, float, float, float]]:
        t = self.wall_thickness
        orientation = int(self.rng.integers(4))
        width = float(self.rng.uniform(1.2, 1.9))
        depth = float(self.rng.uniform(1.2, 2.0))

        if orientation == 0:  # open up
            x = float(self.rng.uniform(-5.0 + t, 5.0 - width))
            y = float(self.rng.uniform(-5.0 + t, 5.0 - depth))
            walls = [
                (x, y, x + t, y + depth),
                (x + width - t, y, x + width, y + depth),
                (x, y, x + width, y + t),
            ]
        elif orientation == 1:  # open down
            x = float(self.rng.uniform(-5.0 + t, 5.0 - width))
            y = float(self.rng.uniform(-5.0 + depth, 5.0 - t))
            walls = [
                (x, y - depth, x + t, y),
                (x + width - t, y - depth, x + width, y),
                (x, y - t, x + width, y),
            ]
        elif orientation == 2:  # open right
            x = float(self.rng.uniform(-5.0 + t, 5.0 - depth))
            y = float(self.rng.uniform(-5.0 + t, 5.0 - width))
            walls = [
                (x, y, x + t, y + width),
                (x, y, x + depth, y + t),
                (x, y + width - t, x + depth, y + width),
            ]
        else:  # open left
            x = float(self.rng.uniform(-5.0 + depth, 5.0 - t))
            y = float(self.rng.uniform(-5.0 + t, 5.0 - width))
            walls = [
                (x - t, y, x, y + width),
                (x - depth, y, x, y + t),
                (x - depth, y + width - t, x, y + width),
            ]

        return [tuple(float(v) for v in wall) for wall in walls]

    def _sample_corridor_group(self) -> List[Tuple[float, float, float, float]]:
        t = self.wall_thickness
        orientation = int(self.rng.integers(2))
        corridor_width = float(self.rng.uniform(0.75, 1.05))
        length = float(self.rng.uniform(5.0, 7.0))

        if orientation == 0:  # vertical corridor
            x = float(self.rng.uniform(-2.6, 2.6 - corridor_width))
            y = float(self.rng.uniform(-4.6, 4.6 - length))
            walls = [
                (x, y, x + t, y + length),
                (x + corridor_width, y, x + corridor_width + t, y + length),
            ]
        else:  # horizontal corridor
            x = float(self.rng.uniform(-4.6, 4.6 - length))
            y = float(self.rng.uniform(-2.6, 2.6 - corridor_width))
            walls = [
                (x, y, x + length, y + t),
                (x, y + corridor_width, x + length, y + corridor_width + t),
            ]

        return [tuple(float(v) for v in wall) for wall in walls]

    def _populate_variant(self, variant: str) -> None:
        self.dynamic_walls = []
        self.active_env_variant = variant
        self._clear_dynamic_motion_state()

        if variant == "baseline":
            super().place_dynamic_walls()
            self._commit_static_layout()
            return

        if variant == "ood_dense":
            super().place_dynamic_walls()
            self._add_extra_random_walls(
                4,
                min_dist=0.65,
                min_length=0.6,
                max_length=1.2,
                max_attempts=1200,
            )
            self._commit_static_layout()
            return

        if variant == "ood_l_walls":
            groups_added = 0
            attempts = 0
            while groups_added < 3 and attempts < 400:
                if self._try_add_group(self._sample_l_group(), min_dist=0.85):
                    groups_added += 1
                attempts += 1
            self._add_extra_random_walls(4, min_dist=0.8, min_length=0.6, max_length=1.4)
        elif variant == "ood_u_traps":
            groups_added = 0
            attempts = 0
            while groups_added < 2 and attempts < 300:
                if self._try_add_group(self._sample_u_group(), min_dist=0.85):
                    groups_added += 1
                attempts += 1
            self._add_extra_random_walls(5, min_dist=0.8, min_length=0.6, max_length=1.4)
        elif variant == "ood_corridor":
            added = False
            attempts = 0
            while not added and attempts < 200:
                added = self._try_add_group(self._sample_corridor_group(), min_dist=0.9, agent_padding=0.5)
                attempts += 1
            self._add_extra_random_walls(6, min_dist=0.75, min_length=0.6, max_length=1.5)
        elif variant == "ood_dynamic_crossing":
            self._add_extra_random_walls(3, min_dist=0.9, min_length=0.6, max_length=1.2)
            self._commit_static_layout()
            obstacle = self._sample_moving_crossing_obstacle()
            if obstacle is None:
                super().place_dynamic_walls()
                self._commit_static_layout()
                return
            self._moving_obstacles.append(obstacle)
            self._refresh_dynamic_walls()
            return
        else:
            raise ValueError(f"Unsupported variant population request: {variant}")

        if len(self.dynamic_walls) < 6:
            super().place_dynamic_walls()
        self._commit_static_layout()

    def place_dynamic_walls(self):
        variant = self.env_variant
        if variant == "ood_mixed":
            variant = str(self.rng.choice(np.asarray(self.OOD_MIXED_CHOICES, dtype=object)))
        self._populate_variant(variant)

    def _dropout_fill_value(self, lidar_m: np.ndarray) -> float:
        if self.beam_dropout_fill == "min":
            return float(LIDAR_MIN_RANGE_M)
        if self.beam_dropout_fill == "mean":
            return float(np.mean(lidar_m))
        return float(LIDAR_MAX_RANGE_M)

    def _apply_observation_corruption(self, obs: np.ndarray) -> np.ndarray:
        if self.lidar_noise_std <= 0.0 and self.beam_dropout_rate <= 0.0:
            return np.asarray(obs, dtype=np.float32).copy()

        corrupted = np.asarray(obs, dtype=np.float32).copy()
        lidar_m = normalized_lidar_to_distance_m(corrupted[6:])

        if self.lidar_noise_std > 0.0:
            lidar_m = lidar_m + self.rng.normal(0.0, self.lidar_noise_std, size=lidar_m.shape)

        lidar_m = np.clip(lidar_m, LIDAR_MIN_RANGE_M, LIDAR_MAX_RANGE_M)

        if self.beam_dropout_rate > 0.0:
            mask = self.rng.random(lidar_m.shape[0]) < self.beam_dropout_rate
            if np.any(mask):
                lidar_m = lidar_m.copy()
                lidar_m[mask] = self._dropout_fill_value(lidar_m)

        corrupted[6:] = distance_m_to_normalized_lidar(lidar_m)
        return corrupted

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["env_variant"] = self.active_env_variant
        return self._apply_observation_corruption(obs), info

    def step(self, action):
        info = {}

        pre_lidar = self.lidar_readings()
        pre_min_lidar = float(np.min(pre_lidar))

        linear_velocity = (action[0] + 1) * 0.11
        angular_velocity = action[1] * 2.84
        wL, wR = self.terskin(linear_velocity, angular_velocity)
        v, w = self.ilerikin(wR, wL)

        theta = self.agent_pos[2]
        self.agent_pos[0] += v * np.cos(theta) * DT
        self.agent_pos[1] += v * np.sin(theta) * DT
        self.agent_pos[2] = (self.agent_pos[2] + w * DT) % (2 * np.pi)
        self.agent_pos[0:2] = np.clip(self.agent_pos[0:2], -5 + self.wall_thickness, 5 - self.wall_thickness)
        self.path.append(self.agent_pos[:2].copy())

        distance_to_target = np.linalg.norm(self.agent_pos[0:2] - self.target_pos)
        self.distance_buffer.append(distance_to_target)
        distance_diff = self.distance_buffer[-1] - self.distance_buffer[0]
        step_distance = v * DT
        self.accumulated_distance += step_distance

        self._advance_dynamic_obstacles(DT)

        lidar_data = self.lidar_readings()
        post_min_lidar = float(np.min(lidar_data))
        min_lidar_distance = min(pre_min_lidar, post_min_lidar)

        target_reached = distance_to_target < self.GOAL_REACHED_DIST
        collision = min_lidar_distance < self.COLLISION_DIST
        timeout = self.step_count >= self.max_steps - 0.1

        theta_to_target = self.calculate_theta_to_target(self.agent_pos, self.target_pos)
        reward = self._reward(
            distance_to_target,
            target_reached,
            collision,
            linear_velocity,
            angular_velocity,
            min_lidar_distance,
            distance_diff,
        )

        normalized_distance = distance_to_target / 100.0
        normalized_linear_velocity = ((linear_velocity - 0.0) / (0.22 - 0.0)) * 2.0 - 1.0
        normalized_angular_velocity = ((angular_velocity - (-2.84)) / (2.84 - (-2.84))) * 2.0 - 1.0
        self.MIN_LIDAR = min(self.MIN_LIDAR, np.min(lidar_data))
        self.MAX_LIDAR = max(self.MAX_LIDAR, np.max(lidar_data))
        if self.MIN_LIDAR == self.MAX_LIDAR:
            self.MAX_LIDAR += 0.01
        normalized_lidar = np.clip(
            ((lidar_data - self.MIN_LIDAR) / (self.MAX_LIDAR - self.MIN_LIDAR)) * 2.0 - 1.0,
            -1.0,
            1.0,
        )
        normalized_distance_diff = distance_diff / 100.0

        done = target_reached or collision or timeout
        if done:
            if target_reached:
                self.target_outcomes[-1] = "reached"
                info["reason"] = "target_reached"
            elif collision:
                self.target_outcomes[-1] = "collision"
                info["reason"] = "collision"
            elif timeout:
                self.target_outcomes[-1] = "timeout"
                info["reason"] = "max_steps_reached"

        truncated = bool(timeout)
        terminated = bool(target_reached or collision)
        self.step_count += 1

        obs = np.concatenate(
            [
                [
                    np.sin(theta_to_target),
                    np.cos(theta_to_target),
                    normalized_linear_velocity,
                    normalized_angular_velocity,
                    normalized_distance,
                    normalized_distance_diff,
                ],
                normalized_lidar,
            ]
        ).astype(np.float32)
        info = dict(info)
        info["env_variant"] = self.active_env_variant
        info["moving_obstacles"] = int(len(self._moving_obstacles))
        return self._apply_observation_corruption(obs), float(reward), terminated, truncated, info


def build_nav_context_from_obs(obs: np.ndarray, step_idx: int, episode_idx: Optional[int] = None) -> Dict[str, float]:
    obs = np.asarray(obs, dtype=np.float32)

    sin_th, cos_th = float(obs[0]), float(obs[1])
    heading_rad = math.atan2(sin_th, cos_th)
    heading_deg = float(np.degrees(heading_rad))

    n_lin_v = float(obs[2])
    n_ang_v = float(obs[3])

    approx_lin_v = (n_lin_v + 1.0) * 0.22 / 2.0
    approx_ang_v = (n_ang_v + 1.0) * 2.84 - 2.84

    approx_dist = float(obs[4]) * 100.0
    approx_ddiff = float(obs[5]) * 100.0

    lidar_norm = obs[6:]
    min_lidar_norm = float(np.min(lidar_norm))
    mean_lidar_norm = float(np.mean(lidar_norm))

    return {
        "step": int(step_idx),
        "episode": None if episode_idx is None else int(episode_idx),
        "approx_distance_to_target_m": approx_dist,
        "approx_distance_delta_100steps_m": approx_ddiff,
        "heading_deg_to_target": heading_deg,
        "approx_linear_velocity_m_s": approx_lin_v,
        "approx_angular_velocity_rad_s": approx_ang_v,
        "min_lidar_normalized": min_lidar_norm,
        "mean_lidar_normalized": mean_lidar_norm,
        "proximity_category": (
            "very_close"
            if min_lidar_norm < -0.2
            else ("close" if min_lidar_norm < 0.0 else "comfortable")
        ),
        "progress_trend": (
            "getting_closer"
            if approx_ddiff < -0.01
            else ("getting_farther" if approx_ddiff > 0.01 else "stagnant")
        ),
    }


def nav_context_to_features(nav_context: Dict[str, float]) -> np.ndarray:
    heading_rad = math.radians(float(nav_context["heading_deg_to_target"]))

    proximity = str(nav_context["proximity_category"])
    progress = str(nav_context["progress_trend"])

    prox_one_hot = [
        1.0 if proximity == "comfortable" else 0.0,
        1.0 if proximity == "close" else 0.0,
        1.0 if proximity == "very_close" else 0.0,
    ]
    progress_one_hot = [
        1.0 if progress == "getting_closer" else 0.0,
        1.0 if progress == "stagnant" else 0.0,
        1.0 if progress == "getting_farther" else 0.0,
    ]

    features = [
        float(nav_context["approx_distance_to_target_m"]),
        float(nav_context["approx_distance_delta_100steps_m"]),
        math.sin(heading_rad),
        math.cos(heading_rad),
        float(nav_context["approx_linear_velocity_m_s"]),
        float(nav_context["approx_angular_velocity_rad_s"]),
        float(nav_context["min_lidar_normalized"]),
        float(nav_context["mean_lidar_normalized"]),
        *prox_one_hot,
        *progress_one_hot,
    ]
    return np.asarray(features, dtype=np.float32)


def is_safe_nav_context(nav_context: Dict[str, float]) -> bool:
    return (
        nav_context["proximity_category"] == "comfortable"
        and nav_context["progress_trend"] == "getting_closer"
    )


def normalize_mode(
    mode: str,
    speed_scale: float,
    turn_scale: float,
    turn_bias: float,
    nav_context: Optional[Dict[str, float]] = None,
) -> str:
    mode = str(mode).strip().lower()
    if mode in SUPPORTED_MODES:
        return mode
    if mode in {"reorient", "correcting", "adjusting"}:
        return "explore"
    if mode == "comfortable":
        return "follow"
    if speed_scale <= 0.05:
        return "stop"
    if speed_scale < 0.8 or turn_scale > 1.05 or abs(turn_bias) > 0.05:
        return "cautious"
    if nav_context and nav_context.get("progress_trend") != "getting_closer":
        return "explore"
    return "follow"


def sanitize_decision(
    decision: Optional[Dict[str, float]],
    nav_context: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if not decision:
        return dict(DEFAULT_DECISION)

    speed_scale = float(decision.get("speed_scale", DEFAULT_DECISION["speed_scale"]))
    turn_scale = float(decision.get("turn_scale", DEFAULT_DECISION["turn_scale"]))
    turn_bias = float(decision.get("turn_bias", DEFAULT_DECISION["turn_bias"]))
    mode = normalize_mode(
        str(decision.get("mode", DEFAULT_DECISION["mode"])),
        speed_scale=speed_scale,
        turn_scale=turn_scale,
        turn_bias=turn_bias,
        nav_context=nav_context,
    )

    return {
        "mode": mode,
        "speed_scale": max(0.0, min(1.0, speed_scale)),
        "turn_scale": max(0.5, min(2.0, turn_scale)),
        "turn_bias": max(-1.0, min(1.0, turn_bias)),
    }


def build_supervisor_prompt(nav_context: Dict[str, float]) -> str:
    base = {
        "role": "High-level navigation safety supervisor LLM",
        "task": (
            "You supervise a low-level RL controller that already navigates quite well. "
            "Your job is NOT to replace it, but to apply small, safety-oriented corrections "
            "ONLY when there is risk (obstacles too close) or poor progress (not getting closer to target)."
        ),
        "instructions": [
            "If proximity_category is 'comfortable' AND progress_trend is 'getting_closer', "
            "then keep mode='follow' with speed_scale near 1.0, turn_scale near 1.0 and turn_bias near 0.0.",
            "If proximity_category is 'very_close', prefer mode='cautious' or 'stop' with speed_scale <= 0.4 "
            "and optionally slightly higher turn_scale to turn away from obstacles.",
            "If proximity_category is 'close' (but not 'very_close'), reduce speed (for example speed_scale between 0.3 and 0.6) "
            "while keeping turn_scale near 1.0 unless progress is getting_farther.",
            "If progress_trend is 'getting_farther' and distance is still large, you may increase turn_scale slightly "
            "(for example up to 1.3 to 1.5) to change heading, but avoid extreme oscillations.",
            "Treat the RL controller as the primary policy. Your modifications must be smooth and conservative, "
            "avoiding large changes unless there is a clear safety issue.",
            "Always return a single JSON object, no explanation text.",
        ],
        "nav_context": nav_context,
    }
    return (
        "You are a high-level navigation safety supervisor for a mobile robot.\n"
        "A reinforcement learning (RL) controller produces base actions, and you receive a JSON 'nav_context' "
        "that summarizes the current state.\n\n"
        "You must decide a high-level 'mode' and scaling factors that slightly modulate the RL controller.\n"
        "IMPORTANT: The RL controller is already well-trained. You mostly let it act, and intervene only when necessary.\n\n"
        f"{json.dumps(base, indent=2)}\n\n"
        "Respond ONLY with a JSON object with keys: "
        "'mode', 'speed_scale', 'turn_scale', 'turn_bias'.\n"
        "Do NOT add any extra text or markdown.\n"
        "Example valid response:\n"
        "{\n"
        '  "mode": "cautious",\n'
        '  "speed_scale": 0.5,\n'
        '  "turn_scale": 1.2,\n'
        '  "turn_bias": 0.1\n'
        "}\n"
    )


@dataclass
class EpisodeRecord:
    controller: str
    episode: int
    seed: int
    outcome: str
    reason: str
    steps: int
    total_reward: float
    path_length_m: float
    efficiency: float
    initial_distance_m: float
    final_distance_m: float
    decision_updates: int


class BaseController:
    def __init__(self, name: str):
        self.name = name
        self.episode_query_count = 0

    def reset_run(self) -> None:
        pass

    def reset_episode(self) -> None:
        self.episode_query_count = 0

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        raise NotImplementedError

    def apply_decision(self, rl_action: np.ndarray, decision: Dict[str, float]) -> np.ndarray:
        return np.asarray(rl_action, dtype=np.float32).copy()

    def finalize_episode(self, record: EpisodeRecord) -> None:
        del record

    def extra_summary(self) -> Dict[str, object]:
        return {}

    def save_extra_outputs(self, outdir: Path) -> None:
        del outdir


class RLOnlyController(BaseController):
    def __init__(self):
        super().__init__(name="rl_only")

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        return dict(DEFAULT_DECISION), None


class SupervisorController(BaseController):
    def __init__(self, name: str, decision_interval: int):
        super().__init__(name=name)
        self.decision_interval = max(1, int(decision_interval))
        self.last_decision = dict(DEFAULT_DECISION)
        self.last_nav_context: Optional[Dict[str, float]] = None

    def reset_episode(self) -> None:
        super().reset_episode()
        self.last_decision = dict(DEFAULT_DECISION)
        self.last_nav_context = None

    def infer_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        nav_context = build_nav_context_from_obs(obs, step_idx, episode_idx)
        self.last_nav_context = nav_context

        if step_idx % self.decision_interval != 0 and step_idx > 0:
            return dict(self.last_decision), None

        if is_safe_nav_context(nav_context):
            self.last_decision = dict(DEFAULT_DECISION)
            return dict(self.last_decision), None

        raw_decision = self.infer_decision(nav_context)
        self.last_decision = sanitize_decision(raw_decision, nav_context)
        self.episode_query_count += 1

        log_entry = None
        if log:
            log_entry = {
                "episode": int(episode_idx) if episode_idx is not None else None,
                "step": int(step_idx),
                "nav_context": nav_context,
                "decision": self.last_decision,
            }
        return dict(self.last_decision), log_entry

    def apply_decision(self, rl_action: np.ndarray, decision: Dict[str, float]) -> np.ndarray:
        orig = np.asarray(rl_action, dtype=np.float32).copy()
        a = orig.copy()
        nav = self.last_nav_context or {}
        proximity = nav.get("proximity_category", "comfortable")
        progress = nav.get("progress_trend", "stagnant")
        mode = str(decision.get("mode", "follow"))

        if proximity == "comfortable" and progress == "getting_closer":
            return np.clip(orig, -1.0, 1.0)

        if mode == "stop":
            a[:] = 0.0
            return np.clip(a, -1.0, 1.0)

        speed_scale = float(decision.get("speed_scale", 1.0))
        turn_scale = float(decision.get("turn_scale", 1.0))
        turn_bias = float(decision.get("turn_bias", 0.0))

        if proximity == "very_close":
            speed_scale = min(speed_scale, 0.3)
            turn_scale = max(turn_scale, 1.2)
        elif proximity == "close":
            speed_scale = min(speed_scale, 0.6)

        if progress == "getting_farther":
            turn_scale = min(max(turn_scale, 1.1), 1.6)

        a[0] = a[0] * speed_scale
        a[1] = a[1] * turn_scale + turn_bias

        a[0] = float(np.clip(a[0], orig[0] - 0.3, orig[0] + 0.3))
        a[1] = float(np.clip(a[1], orig[1] - 0.5, orig[1] + 0.5))
        return np.clip(a, -1.0, 1.0)


class GateOnlyController(SupervisorController):
    def __init__(self, decision_interval: int):
        super().__init__(name="gate_only", decision_interval=decision_interval)

    def infer_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        return dict(DEFAULT_DECISION)


class APFReactiveController(SupervisorController):
    def __init__(self):
        # APF is intended as a low-latency reactive baseline, so it updates every step.
        super().__init__(name="apf_reactive", decision_interval=1)
        self.last_obs: Optional[np.ndarray] = None

    def reset_episode(self) -> None:
        super().reset_episode()
        self.last_obs = None

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        return super().maybe_update(step_idx=step_idx, obs=obs, episode_idx=episode_idx, log=log)

    def infer_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        if self.last_obs is None:
            return dict(DEFAULT_DECISION)

        obs = self.last_obs
        heading_rad = math.atan2(float(obs[0]), float(obs[1]))
        lidar_norm = obs[6:]
        lidar_m = normalized_lidar_to_distance_m(lidar_norm)
        rel_angles = LIDAR_REL_ANGLES_RAD

        front_mask = np.abs(rel_angles) <= math.radians(35.0)
        left_mask = (rel_angles >= math.radians(15.0)) & (rel_angles <= math.radians(135.0))
        right_mask = (rel_angles <= -math.radians(15.0)) & (rel_angles >= -math.radians(135.0))

        influence_radius = 0.55
        influence = np.clip(
            (influence_radius - lidar_m) / max(influence_radius - LIDAR_MIN_RANGE_M, 1e-6),
            0.0,
            1.0,
        )

        repulse_lateral = float(np.sum(-np.sin(rel_angles) * (influence ** 2)))
        repulse_front = float(np.sum(np.maximum(np.cos(rel_angles), 0.0) * (influence ** 2)))
        front_min = float(np.min(lidar_m[front_mask]))
        global_min = float(np.min(lidar_m))
        left_clearance = float(np.mean(lidar_m[left_mask]))
        right_clearance = float(np.mean(lidar_m[right_mask]))

        target_steer = math.sin(heading_rad)
        steer_signal = 0.55 * target_steer + 0.75 * math.tanh(repulse_lateral)

        # Break ties near frontal obstacles by steering toward the more open side.
        if abs(steer_signal) < 0.03 and front_min < 0.40:
            steer_signal = 0.10 * (1.0 if left_clearance > right_clearance else -1.0)

        speed_scale = float(np.clip(1.0 - 0.18 * repulse_front, 0.0, 1.0))
        turn_scale = 1.0 + min(0.9, 0.45 * abs(steer_signal) + 0.15 * repulse_front)
        turn_bias = float(np.clip(0.22 * math.tanh(steer_signal), -0.25, 0.25))
        mode = "follow"

        if global_min < 0.42 or front_min < 0.48:
            mode = "cautious"
            speed_scale = min(speed_scale, 0.60)
            turn_scale = max(turn_scale, 1.10)

        if front_min < 0.30:
            mode = "cautious"
            speed_scale = min(speed_scale, 0.30)
            turn_scale = max(turn_scale, 1.25)

        if front_min < 0.22:
            mode = "stop"
            speed_scale = 0.0
            turn_scale = max(turn_scale, 1.35)
            turn_bias = float(np.clip(turn_bias, -0.18, 0.18))

        if str(nav_context.get("progress_trend", "stagnant")) == "getting_farther":
            mode = "explore" if mode != "stop" else mode
            turn_scale = max(turn_scale, 1.15)

        return {
            "mode": mode,
            "speed_scale": speed_scale,
            "turn_scale": turn_scale,
            "turn_bias": turn_bias,
        }


class CBFStyleController(BaseController):
    def __init__(self):
        super().__init__(name="cbf_style")
        self.last_obs: Optional[np.ndarray] = None
        self.last_nav_context: Optional[Dict[str, float]] = None
        self.filter_interventions = 0
        self.full_stop_interventions = 0
        self.safe_velocity_caps: List[float] = []

    def reset_run(self) -> None:
        self.last_obs = None
        self.last_nav_context = None
        self.filter_interventions = 0
        self.full_stop_interventions = 0
        self.safe_velocity_caps = []

    def reset_episode(self) -> None:
        super().reset_episode()
        self.last_obs = None
        self.last_nav_context = None

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        del episode_idx, log
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        self.last_nav_context = build_nav_context_from_obs(obs, step_idx, episode_idx=None)
        self.episode_query_count += 1
        return dict(DEFAULT_DECISION), None

    def apply_decision(self, rl_action: np.ndarray, decision: Dict[str, float]) -> np.ndarray:
        del decision
        if self.last_obs is None:
            return np.asarray(rl_action, dtype=np.float32).copy()

        obs = self.last_obs
        action = np.asarray(rl_action, dtype=np.float32).copy()
        lidar_m = normalized_lidar_to_distance_m(obs[6:])
        rel_angles = LIDAR_REL_ANGLES_RAD

        desired_v = linear_velocity_from_action(action)
        desired_w = angular_velocity_from_action(action)

        front_mask = np.abs(rel_angles) <= math.radians(75.0)
        left_mask = (rel_angles >= math.radians(20.0)) & (rel_angles <= math.radians(140.0))
        right_mask = (rel_angles <= -math.radians(20.0)) & (rel_angles >= -math.radians(140.0))
        forward_cos = np.maximum(np.cos(rel_angles), 0.0)
        active_mask = front_mask & (forward_cos > 0.10)

        front_min = float(np.min(lidar_m[front_mask]))
        global_min = float(np.min(lidar_m))
        left_clearance = float(np.mean(lidar_m[left_mask]))
        right_clearance = float(np.mean(lidar_m[right_mask]))

        # Discrete-time CBF-style speed cap:
        # For forward-facing rays, enforce a one-step-ahead minimum clearance margin.
        d_safe = 0.21
        d_stop = 0.185
        alpha = 0.55

        safe_v_bound = 0.22
        if np.any(active_mask):
            ray_bounds = []
            for d_i, c_i in zip(lidar_m[active_mask], forward_cos[active_mask]):
                h_i = float(d_i) - d_safe
                # h_{k+1} >= (1-alpha) h_k under a forward radial approximation
                numerator = alpha * max(h_i, 0.0)
                ray_bounds.append(numerator / max(DT * float(c_i), 1e-6))
            if ray_bounds:
                safe_v_bound = float(np.clip(min(ray_bounds), 0.0, 0.22))

        if front_min <= d_stop or global_min <= CustomEnv.COLLISION_DIST + 0.01:
            safe_v_bound = 0.0

        self.safe_velocity_caps.append(float(safe_v_bound))
        filtered_v = min(desired_v, safe_v_bound)

        influence_radius = 0.60
        influence = np.clip(
            (influence_radius - lidar_m) / max(influence_radius - LIDAR_MIN_RANGE_M, 1e-6),
            0.0,
            1.0,
        )
        repulse_lateral = float(np.sum(-np.sin(rel_angles) * (influence ** 2)))
        steer_sign = 0.0
        if abs(repulse_lateral) > 1e-5:
            steer_sign = 1.0 if repulse_lateral > 0.0 else -1.0
        elif abs(left_clearance - right_clearance) > 1e-5:
            steer_sign = 1.0 if left_clearance > right_clearance else -1.0

        intervention = filtered_v + 1e-6 < desired_v
        if intervention:
            self.filter_interventions += 1
            angular_boost = 0.0
            if steer_sign != 0.0:
                angular_boost = 0.60 if filtered_v <= 0.02 else 0.35
                desired_w = float(np.clip(desired_w + steer_sign * angular_boost, -2.84, 2.84))
            if filtered_v <= 0.02:
                self.full_stop_interventions += 1

        action[0] = action0_from_linear_velocity(filtered_v)
        action[1] = action1_from_angular_velocity(desired_w)
        return np.clip(action, -1.0, 1.0)

    def extra_summary(self) -> Dict[str, object]:
        if self.safe_velocity_caps:
            caps = np.asarray(self.safe_velocity_caps, dtype=np.float64)
            cap_stats = {
                "mean": float(np.mean(caps)),
                "median": float(np.median(caps)),
                "p95": float(np.quantile(caps, 0.95)),
                "min": float(np.min(caps)),
            }
        else:
            cap_stats = {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0}
        return {
            "filter_type": "discrete_time_cbf_style_shield",
            "filter_interventions": int(self.filter_interventions),
            "full_stop_interventions": int(self.full_stop_interventions),
            "safe_velocity_cap_m_s": cap_stats,
        }


class FormalCBFController(BaseController):
    def __init__(self):
        super().__init__(name="cbf_formal")
        self.last_obs: Optional[np.ndarray] = None
        self.last_nav_context: Optional[Dict[str, float]] = None
        self.lookahead_m = 0.08
        self.safe_radius_m = 0.12
        self.activation_distance_m = 0.65
        self.max_active_rays = 18
        self.alpha = 3.0
        self.intervention_count = 0
        self.zero_speed_count = 0
        self.infeasible_fallback_count = 0
        self.qp_solve_count = 0
        self.qp_failure_count = 0
        self.active_constraint_counts: List[int] = []
        self.min_barrier_values: List[float] = []

    def reset_run(self) -> None:
        self.last_obs = None
        self.last_nav_context = None
        self.intervention_count = 0
        self.zero_speed_count = 0
        self.infeasible_fallback_count = 0
        self.qp_solve_count = 0
        self.qp_failure_count = 0
        self.active_constraint_counts = []
        self.min_barrier_values = []

    def reset_episode(self) -> None:
        super().reset_episode()
        self.last_obs = None
        self.last_nav_context = None

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        del episode_idx, log
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        self.last_nav_context = build_nav_context_from_obs(obs, step_idx, episode_idx=None)
        self.episode_query_count += 1
        return dict(DEFAULT_DECISION), None

    def _build_cbf_constraints(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
        lidar_m = normalized_lidar_to_distance_m(obs[6:])
        rel_angles = LIDAR_REL_ANGLES_RAD
        sector_mask = (np.abs(rel_angles) <= math.radians(110.0)) & (lidar_m <= self.activation_distance_m)
        active_idx = np.flatnonzero(sector_mask)
        if active_idx.size > self.max_active_rays:
            closest_order = np.argsort(lidar_m[active_idx])[: self.max_active_rays]
            active_idx = active_idx[closest_order]

        constraints: List[Tuple[np.ndarray, float]] = []
        min_h = float('inf')
        for idx in active_idx.tolist():
            d_i = float(lidar_m[idx])
            phi_i = float(rel_angles[idx])
            r_x = self.lookahead_m - d_i * math.cos(phi_i)
            r_y = -d_i * math.sin(phi_i)
            h_i = (r_x * r_x) + (r_y * r_y) - (self.safe_radius_m * self.safe_radius_m)
            min_h = min(min_h, h_i)
            a_i = np.asarray([2.0 * r_x, 2.0 * self.lookahead_m * r_y], dtype=np.float64)
            b_i = -self.alpha * h_i
            constraints.append((a_i, float(b_i)))

        if not constraints:
            min_h = float((self.activation_distance_m ** 2) - (self.safe_radius_m ** 2))

        bounds = [
            (np.asarray([1.0, 0.0], dtype=np.float64), 0.0),
            (np.asarray([-1.0, 0.0], dtype=np.float64), -0.22),
            (np.asarray([0.0, 1.0], dtype=np.float64), -2.84),
            (np.asarray([0.0, -1.0], dtype=np.float64), -2.84),
        ]
        constraints.extend(bounds)
        A = np.vstack([a for a, _ in constraints])
        b = np.asarray([c for _, c in constraints], dtype=np.float64)
        return A, b, float(min_h), int(len(active_idx))

    @staticmethod
    def _is_feasible(u: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
        return bool(np.all((A @ u) >= (b - tol)))

    def _analytic_projection_fallback(
        self,
        u_des: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        candidates: List[np.ndarray] = []
        for i in range(A.shape[0]):
            a_i = A[i]
            norm_sq = float(a_i @ a_i)
            if norm_sq <= 1e-10:
                continue
            lam = (b[i] - float(a_i @ u_des)) / norm_sq
            u_i = u_des + lam * a_i
            if self._is_feasible(u_i, A, b):
                candidates.append(u_i)

        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[0]):
                M = np.vstack([A[i], A[j]])
                det = float(np.linalg.det(M))
                if abs(det) <= 1e-10:
                    continue
                u_ij = np.linalg.solve(M, np.asarray([b[i], b[j]], dtype=np.float64))
                if self._is_feasible(u_ij, A, b):
                    candidates.append(u_ij)

        if candidates:
            best = min(candidates, key=lambda u: float(np.sum((u - u_des) ** 2)))
            return np.asarray(best, dtype=np.float64)

        self.infeasible_fallback_count += 1
        desired_w = float(np.clip(u_des[1], -2.84, 2.84))
        return np.asarray([0.0, desired_w], dtype=np.float64)

    def _solve_cbf_qp(
        self,
        u_des: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        if self._is_feasible(u_des, A, b):
            return u_des.copy(), False

        self.qp_solve_count += 1
        bounds = Bounds(lb=np.asarray([0.0, -2.84], dtype=np.float64), ub=np.asarray([0.22, 2.84], dtype=np.float64))
        linear_constraint = LinearConstraint(A, lb=b, ub=np.full_like(b, np.inf))

        def objective(u: np.ndarray) -> float:
            diff = u - u_des
            return 0.5 * float(diff @ diff)

        def objective_jac(u: np.ndarray) -> np.ndarray:
            return u - u_des

        x0 = np.asarray([
            float(np.clip(u_des[0], 0.0, 0.22)),
            float(np.clip(u_des[1], -2.84, 2.84)),
        ], dtype=np.float64)

        result = minimize(
            objective,
            x0=x0,
            jac=objective_jac,
            method='SLSQP',
            bounds=bounds,
            constraints=[linear_constraint],
            options={'ftol': 1e-9, 'maxiter': 60, 'disp': False},
        )

        if result.success and self._is_feasible(np.asarray(result.x, dtype=np.float64), A, b, tol=1e-6):
            return np.asarray(result.x, dtype=np.float64), True

        self.qp_failure_count += 1
        return self._analytic_projection_fallback(u_des, A, b), True

    def apply_decision(self, rl_action: np.ndarray, decision: Dict[str, float]) -> np.ndarray:
        del decision
        if self.last_obs is None:
            return np.asarray(rl_action, dtype=np.float32).copy()

        action = np.asarray(rl_action, dtype=np.float32).copy()
        desired_u = np.asarray(
            [linear_velocity_from_action(action), angular_velocity_from_action(action)],
            dtype=np.float64,
        )
        A, b, min_h, active_count = self._build_cbf_constraints(self.last_obs)
        self.active_constraint_counts.append(active_count)
        self.min_barrier_values.append(float(min_h))

        filtered_u, intervened = self._solve_cbf_qp(desired_u, A, b)
        filtered_u[0] = float(np.clip(filtered_u[0], 0.0, 0.22))
        filtered_u[1] = float(np.clip(filtered_u[1], -2.84, 2.84))

        if intervened or abs(filtered_u[0] - desired_u[0]) > 1e-6 or abs(filtered_u[1] - desired_u[1]) > 1e-6:
            self.intervention_count += 1
        if filtered_u[0] <= 1e-4:
            self.zero_speed_count += 1

        action[0] = action0_from_linear_velocity(float(filtered_u[0]))
        action[1] = action1_from_angular_velocity(float(filtered_u[1]))
        return np.clip(action, -1.0, 1.0)

    def extra_summary(self) -> Dict[str, object]:
        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
            arr = np.asarray(values, dtype=np.float64)
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.quantile(arr, 0.95)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        return {
            "filter_type": "lookahead_zeroing_cbf_qp",
            "lookahead_m": float(self.lookahead_m),
            "safe_radius_m": float(self.safe_radius_m),
            "activation_distance_m": float(self.activation_distance_m),
            "alpha": float(self.alpha),
            "projection_interventions": int(self.intervention_count),
            "qp_solve_count": int(self.qp_solve_count),
            "qp_failure_count": int(self.qp_failure_count),
            "zero_speed_steps": int(self.zero_speed_count),
            "infeasible_fallback_count": int(self.infeasible_fallback_count),
            "active_cbf_constraints": _stats([float(v) for v in self.active_constraint_counts]),
            "min_barrier_h": _stats(self.min_barrier_values),
        }


class RuleBasedController(SupervisorController):
    def __init__(self, decision_interval: int):
        super().__init__(name="rule_based", decision_interval=decision_interval)

    def infer_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        proximity = str(nav_context["proximity_category"])
        progress = str(nav_context["progress_trend"])
        min_lidar = float(nav_context["min_lidar_normalized"])
        distance = float(nav_context["approx_distance_to_target_m"])
        heading_deg = float(nav_context["heading_deg_to_target"])

        if abs(heading_deg) < 5.0:
            heading_sign = 0.0
        else:
            heading_sign = 1.0 if heading_deg > 0 else -1.0

        bias_toward_target = 0.10 * heading_sign

        if proximity == "very_close":
            if min_lidar < -0.55:
                return {
                    "mode": "stop",
                    "speed_scale": 0.0,
                    "turn_scale": 1.4,
                    "turn_bias": 0.15 * heading_sign,
                }
            return {
                "mode": "cautious",
                "speed_scale": 0.22,
                "turn_scale": 1.35,
                "turn_bias": 0.12 * heading_sign,
            }

        if proximity == "close":
            if progress == "getting_farther":
                return {
                    "mode": "cautious",
                    "speed_scale": 0.40,
                    "turn_scale": 1.30,
                    "turn_bias": bias_toward_target,
                }
            if progress == "stagnant":
                return {
                    "mode": "cautious",
                    "speed_scale": 0.48,
                    "turn_scale": 1.15,
                    "turn_bias": 0.06 * heading_sign,
                }
            return {
                "mode": "cautious",
                "speed_scale": 0.55,
                "turn_scale": 1.0,
                "turn_bias": 0.0,
            }

        if progress == "getting_farther":
            return {
                "mode": "explore",
                "speed_scale": 0.90 if distance > 2.0 else 0.75,
                "turn_scale": 1.35,
                "turn_bias": bias_toward_target,
            }

        if progress == "stagnant":
            return {
                "mode": "explore",
                "speed_scale": 0.95,
                "turn_scale": 1.15,
                "turn_bias": 0.05 * heading_sign,
            }

        return dict(DEFAULT_DECISION)


class ImitationMLP:
    def __init__(self, teacher_trace_path: Path, random_state: int = 42):
        self.teacher_trace_path = Path(teacher_trace_path)
        self.random_state = random_state
        self.mode_model: Optional[Pipeline] = None
        self.reg_model: Optional[Pipeline] = None
        self.training_summary: Dict[str, object] = {}

    def fit(self) -> None:
        with open(self.teacher_trace_path, "r", encoding="utf-8") as handle:
            teacher_data = json.load(handle)

        features = []
        mode_targets = []
        reg_targets = []

        for episode in teacher_data:
            for call in episode.get("llm_calls", []):
                nav_context = call.get("nav_context", {})
                decision = sanitize_decision(call.get("decision", {}), nav_context)
                features.append(nav_context_to_features(nav_context))
                mode_targets.append(decision["mode"])
                reg_targets.append(
                    [
                        decision["speed_scale"],
                        decision["turn_scale"],
                        decision["turn_bias"],
                    ]
                )

        if not features:
            raise ValueError(f"No llm_calls found in teacher trace: {self.teacher_trace_path}")

        X = np.asarray(features, dtype=np.float32)
        y_mode = np.asarray(mode_targets, dtype=object)
        y_reg = np.asarray(reg_targets, dtype=np.float32)

        self.mode_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 32),
                        activation="relu",
                        max_iter=500,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.reg_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(32, 32),
                        activation="relu",
                        max_iter=500,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self.mode_model.fit(X, y_mode)
        self.reg_model.fit(X, y_reg)
        self.training_summary = {
            "teacher_trace_path": str(self.teacher_trace_path),
            "training_samples": int(len(X)),
            "mode_classes": sorted({str(m) for m in y_mode.tolist()}),
        }

    def predict(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        if self.mode_model is None or self.reg_model is None:
            raise RuntimeError("ImitationMLP.fit() must be called before predict().")

        x = nav_context_to_features(nav_context).reshape(1, -1)
        mode = str(self.mode_model.predict(x)[0])
        speed_scale, turn_scale, turn_bias = self.reg_model.predict(x)[0].tolist()
        return sanitize_decision(
            {
                "mode": mode,
                "speed_scale": speed_scale,
                "turn_scale": turn_scale,
                "turn_bias": turn_bias,
            },
            nav_context=nav_context,
        )


class MLPImitationController(SupervisorController):
    def __init__(self, decision_interval: int, imitation_model: ImitationMLP):
        super().__init__(name="mlp_imitation", decision_interval=decision_interval)
        self.imitation_model = imitation_model

    def infer_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        return self.imitation_model.predict(nav_context)


class IndependentDecisionMLP:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model: Optional[Pipeline] = None
        self.template_names: List[str] = []
        self.training_summary: Dict[str, object] = {}

    def fit_from_artifact(self) -> None:
        with open(self.model_path, "rb") as handle:
            payload = pickle.load(handle)

        self.model = payload["model"]
        self.template_names = list(payload["template_names"])
        self.training_summary = dict(payload.get("training_summary", {}))

    def predict(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("IndependentDecisionMLP.fit_from_artifact() must be called before predict().")
        x = nav_context_to_features(nav_context).reshape(1, -1)
        template_name = str(self.model.predict(x)[0])
        return sanitize_decision(
            independent_template_decision(template_name, nav_context),
            nav_context=nav_context,
        )


class IndependentMLPController(SupervisorController):
    def __init__(self, decision_interval: int, independent_model: IndependentDecisionMLP):
        super().__init__(name="mlp_independent", decision_interval=decision_interval)
        self.independent_model = independent_model

    def infer_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        return self.independent_model.predict(nav_context)


class ActualLLMController(SupervisorController):
    def __init__(
        self,
        decision_interval: int,
        provider: str = DEFAULT_LLM_PROVIDER,
        model_name: Optional[str] = None,
        base_url: str = DEFAULT_LLM_BASE_URL,
        timeout_s: float = 20.0,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        super().__init__(name="actual_llm", decision_interval=decision_interval)
        self.provider = provider
        self.model_name = model_name or default_llm_model(provider)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.api_key_env = api_key_env
        self.client = None

        if self.provider == "gemini":
            from google import genai
            from google.genai import types

            api_key = os.getenv(self.api_key_env)
            kwargs = {"api_key": api_key} if api_key else {}
            # The Gemini SDK expects HttpOptions.timeout in milliseconds.
            # Keep retries low so long evaluations do not hang on a single call.
            kwargs["http_options"] = types.HttpOptions(
                timeout=max(1000, int(round(self.timeout_s * 1000.0))),
                retry_options=types.HttpRetryOptions(
                    attempts=1,
                ),
            )
            self.client = genai.Client(**kwargs)

        self.total_calls = 0
        self.parse_failures = 0
        self.request_failures = 0
        self.call_latencies: List[float] = []
        self.episode_traces: List[Dict[str, object]] = []
        self.current_episode_calls: List[Dict[str, object]] = []

    def reset_run(self) -> None:
        self.total_calls = 0
        self.parse_failures = 0
        self.request_failures = 0
        self.call_latencies = []
        self.episode_traces = []
        self.current_episode_calls = []

    def reset_episode(self) -> None:
        super().reset_episode()
        self.current_episode_calls = []

    def build_prompt(self, nav_context: Dict[str, float]) -> str:
        return build_supervisor_prompt(nav_context)

    def _call_local_ollama(self, prompt: str) -> Tuple[str, bool, Optional[str]]:
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "")), False, None
        except Exception as exc:
            return json.dumps(self.last_decision), True, str(exc)

    def _call_gemini(self, prompt: str) -> Tuple[str, bool, Optional[str]]:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return str(response.text), False, None
        except Exception as exc:
            return json.dumps(self.last_decision), True, str(exc)

    def call_model(self, prompt: str) -> Tuple[str, bool, Optional[str]]:
        if self.provider == "local_ollama":
            return self._call_local_ollama(prompt)
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def parse_response(self, text: str) -> Tuple[Dict[str, float], bool, Optional[str]]:
        cleaned = text.strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        try:
            payload = json.loads(cleaned)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict JSON, got {type(payload).__name__}")
        except Exception as exc:
            return dict(self.last_decision), True, str(exc)

        decision = sanitize_decision(payload, self.last_nav_context)
        self.last_decision = decision
        return decision, False, None

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        nav_context = build_nav_context_from_obs(obs, step_idx, episode_idx)
        self.last_nav_context = nav_context

        if step_idx % self.decision_interval != 0 and step_idx > 0:
            return dict(self.last_decision), None

        if is_safe_nav_context(nav_context):
            self.last_decision = dict(DEFAULT_DECISION)
            return dict(self.last_decision), None

        prompt = self.build_prompt(nav_context)
        t0 = time.time()
        raw_response, request_failed, request_error = self.call_model(prompt)
        duration = float(time.time() - t0)
        decision, parse_failed, parse_error = self.parse_response(raw_response)

        self.total_calls += 1
        self.episode_query_count += 1
        self.call_latencies.append(duration)
        self.request_failures += int(request_failed)
        self.parse_failures += int(parse_failed)

        log_entry = {
            "episode": int(episode_idx) if episode_idx is not None else None,
            "step": int(step_idx),
            "nav_context": nav_context,
            "raw_response": raw_response,
            "decision": decision,
            "decision_duration_s": duration,
            "request_failed": bool(request_failed),
            "parse_failed": bool(parse_failed),
        }
        if request_error:
            log_entry["request_error"] = request_error
        if parse_error:
            log_entry["parse_error"] = parse_error
        self.current_episode_calls.append(log_entry)

        if log:
            return decision, log_entry
        return decision, None

    def finalize_episode(self, record: EpisodeRecord) -> None:
        self.episode_traces.append(
            {
                "episode": record.episode,
                "seed": record.seed,
                "outcome": record.outcome,
                "reason": record.reason,
                "steps": record.steps,
                "total_reward": record.total_reward,
                "path_length_m": record.path_length_m,
                "initial_distance_m": record.initial_distance_m,
                "final_distance_m": record.final_distance_m,
                "efficiency": record.efficiency,
                "llm_calls": list(self.current_episode_calls),
            }
        )

    def extra_summary(self) -> Dict[str, object]:
        if self.total_calls == 0:
            return {
                "provider": self.provider,
                "model_name": self.model_name,
                "total_calls": 0,
            }

        latencies = np.asarray(self.call_latencies, dtype=np.float64)
        very_close_calls = [
            call for trace in self.episode_traces for call in trace["llm_calls"]
            if call.get("nav_context", {}).get("proximity_category") == "very_close"
        ]
        close_calls = [
            call for trace in self.episode_traces for call in trace["llm_calls"]
            if call.get("nav_context", {}).get("proximity_category") == "close"
        ]
        very_close_safe = sum(
            1
            for call in very_close_calls
            if float(call.get("decision", {}).get("speed_scale", 1.0)) <= 0.4
        )
        close_safe = sum(
            1
            for call in close_calls
            if float(call.get("decision", {}).get("speed_scale", 1.0)) <= 0.6
        )

        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url if self.provider == "local_ollama" else None,
            "total_calls": int(self.total_calls),
            "request_failures": int(self.request_failures),
            "request_failure_rate": float(self.request_failures / max(1, self.total_calls)),
            "parse_failures": int(self.parse_failures),
            "parse_failure_rate": float(self.parse_failures / max(1, self.total_calls)),
            "latency_s": {
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "p95": float(np.quantile(latencies, 0.95)),
                "max": float(np.max(latencies)),
            },
            "logical_consistency": {
                "very_close_calls": int(len(very_close_calls)),
                "very_close_speed_le_0_4_rate": float(very_close_safe / max(1, len(very_close_calls))),
                "close_calls": int(len(close_calls)),
                "close_speed_le_0_6_rate": float(close_safe / max(1, len(close_calls))),
            },
        }

    def save_extra_outputs(self, outdir: Path) -> None:
        with open(outdir / f"{self.name}_traces.json", "w", encoding="utf-8") as handle:
            json.dump(self.episode_traces, handle, indent=2)


class ActualLLMAsyncController(ActualLLMController):
    def __init__(
        self,
        decision_interval: int,
        provider: str = DEFAULT_LLM_PROVIDER,
        model_name: Optional[str] = None,
        base_url: str = DEFAULT_LLM_BASE_URL,
        timeout_s: float = 20.0,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        super().__init__(
            decision_interval=decision_interval,
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            api_key_env=api_key_env,
        )
        self.name = "actual_llm_async"
        self.pending_decision: Optional[Dict[str, object]] = None
        self.applied_pending_count = 0
        self.discarded_pending_count = 0
        self.skipped_triggers_while_pending = 0
        self.pending_delay_steps: List[int] = []

    def reset_run(self) -> None:
        super().reset_run()
        self.pending_decision = None
        self.applied_pending_count = 0
        self.discarded_pending_count = 0
        self.skipped_triggers_while_pending = 0
        self.pending_delay_steps = []

    def reset_episode(self) -> None:
        super().reset_episode()
        self.pending_decision = None

    def _activate_pending_if_ready(
        self,
        *,
        step_idx: int,
        nav_context: Dict[str, float],
    ) -> Optional[Dict[str, object]]:
        if self.pending_decision is None:
            return None
        if step_idx < int(self.pending_decision["ready_step"]):
            return None

        pending = dict(self.pending_decision)
        self.pending_decision = None

        if is_safe_nav_context(nav_context):
            self.discarded_pending_count += 1
            self.last_decision = dict(DEFAULT_DECISION)
            return {
                "applied_pending": False,
                "discarded_pending": True,
                "pending_ready_step": int(pending["ready_step"]),
                "pending_delay_steps": int(pending["delay_steps"]),
                "pending_initiated_step": int(pending["initiated_step"]),
                "pending_reason": "state_safe_on_arrival",
            }

        self.last_decision = dict(pending["decision"])
        self.applied_pending_count += 1
        self.pending_delay_steps.append(int(pending["delay_steps"]))
        return {
            "applied_pending": True,
            "discarded_pending": False,
            "pending_ready_step": int(pending["ready_step"]),
            "pending_delay_steps": int(pending["delay_steps"]),
            "pending_initiated_step": int(pending["initiated_step"]),
        }

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        nav_context = build_nav_context_from_obs(obs, step_idx, episode_idx)
        self.last_nav_context = nav_context

        pending_event = self._activate_pending_if_ready(step_idx=step_idx, nav_context=nav_context)

        if is_safe_nav_context(nav_context):
            self.last_decision = dict(DEFAULT_DECISION)
            return dict(self.last_decision), (pending_event if log and pending_event else None)

        if step_idx % self.decision_interval != 0 and step_idx > 0:
            return dict(self.last_decision), (pending_event if log and pending_event else None)

        if self.pending_decision is not None:
            self.skipped_triggers_while_pending += 1
            log_entry = None
            if log:
                log_entry = {
                    "episode": int(episode_idx) if episode_idx is not None else None,
                    "step": int(step_idx),
                    "nav_context": nav_context,
                    "decision": dict(self.last_decision),
                    "request_skipped_while_pending": True,
                    "pending_ready_step": int(self.pending_decision["ready_step"]),
                    "pending_delay_steps": int(self.pending_decision["delay_steps"]),
                }
                if pending_event:
                    log_entry["pending_event"] = pending_event
            return dict(self.last_decision), log_entry

        prompt = self.build_prompt(nav_context)
        t0 = time.time()
        raw_response, request_failed, request_error = self.call_model(prompt)
        duration = float(time.time() - t0)
        parsed_decision, parse_failed, parse_error = self.parse_response(raw_response)
        parsed_decision = sanitize_decision(parsed_decision, nav_context)
        delay_steps = max(1, int(math.ceil(duration / DT)))

        self.total_calls += 1
        self.episode_query_count += 1
        self.call_latencies.append(duration)
        self.request_failures += int(request_failed)
        self.parse_failures += int(parse_failed)

        self.pending_decision = {
            "decision": dict(parsed_decision),
            "initiated_step": int(step_idx),
            "ready_step": int(step_idx + delay_steps),
            "delay_steps": int(delay_steps),
        }

        log_entry = {
            "episode": int(episode_idx) if episode_idx is not None else None,
            "step": int(step_idx),
            "nav_context": nav_context,
            "raw_response": raw_response,
            "decision": dict(self.last_decision),
            "pending_decision": dict(parsed_decision),
            "decision_duration_s": duration,
            "request_failed": bool(request_failed),
            "parse_failed": bool(parse_failed),
            "async_initiated": True,
            "pending_ready_step": int(step_idx + delay_steps),
            "pending_delay_steps": int(delay_steps),
        }
        if request_error:
            log_entry["request_error"] = request_error
        if parse_error:
            log_entry["parse_error"] = parse_error
        if pending_event:
            log_entry["pending_event"] = pending_event
        self.current_episode_calls.append(log_entry)

        if log:
            return dict(self.last_decision), log_entry
        return dict(self.last_decision), None

    def extra_summary(self) -> Dict[str, object]:
        summary = super().extra_summary()
        if not summary:
            return {}
        if self.pending_delay_steps:
            delay_stats = {
                "mean": float(np.mean(np.asarray(self.pending_delay_steps, dtype=np.float64))),
                "median": float(np.median(np.asarray(self.pending_delay_steps, dtype=np.float64))),
                "p95": float(np.quantile(np.asarray(self.pending_delay_steps, dtype=np.float64), 0.95)),
                "max": int(np.max(np.asarray(self.pending_delay_steps, dtype=np.int32))),
            }
        else:
            delay_stats = {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
        summary.update(
            {
                "async_mode": "latency_delayed_apply",
                "applied_pending_count": int(self.applied_pending_count),
                "discarded_pending_count": int(self.discarded_pending_count),
                "skipped_triggers_while_pending": int(self.skipped_triggers_while_pending),
                "pending_delay_steps": delay_stats,
            }
        )
        return summary


class ActualLLMEmergencyController(ActualLLMController):
    def __init__(
        self,
        decision_interval: int,
        provider: str = DEFAULT_LLM_PROVIDER,
        model_name: Optional[str] = None,
        base_url: str = DEFAULT_LLM_BASE_URL,
        timeout_s: float = 20.0,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        super().__init__(
            decision_interval=decision_interval,
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            api_key_env=api_key_env,
        )
        self.name = "actual_llm_emergency"
        self.last_llm_step = 0
        self.safe_fallback_count = 0
        self.emergency_override_count = 0
        self.emergency_stop_count = 0

    def reset_run(self) -> None:
        super().reset_run()
        self.safe_fallback_count = 0
        self.emergency_override_count = 0
        self.emergency_stop_count = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self.last_llm_step = 0

    @staticmethod
    def _heading_sign(nav_context: Dict[str, float]) -> float:
        heading_deg = float(nav_context.get("heading_deg_to_target", 0.0))
        if abs(heading_deg) < 5.0:
            return 0.0
        return 1.0 if heading_deg > 0.0 else -1.0

    def _safe_fallback_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        proximity = str(nav_context["proximity_category"])
        progress = str(nav_context["progress_trend"])
        heading_sign = self._heading_sign(nav_context)

        if proximity == "very_close":
            return sanitize_decision(
                {
                    "mode": "cautious",
                    "speed_scale": 0.18 if progress == "getting_closer" else 0.10,
                    "turn_scale": 1.35 if progress == "getting_closer" else 1.50,
                    "turn_bias": 0.05 * heading_sign,
                },
                nav_context,
            )

        if proximity == "close":
            return sanitize_decision(
                {
                    "mode": "cautious",
                    "speed_scale": 0.45 if progress == "getting_closer" else 0.30,
                    "turn_scale": 1.10 if progress == "getting_closer" else 1.20,
                    "turn_bias": 0.03 * heading_sign,
                },
                nav_context,
            )
        return dict(DEFAULT_DECISION)

    def _apply_emergency_filter(
        self,
        decision: Dict[str, float],
        nav_context: Dict[str, float],
        step_idx: int,
        *,
        request_failed: bool = False,
        parse_failed: bool = False,
        held_step: bool = False,
    ) -> Tuple[Dict[str, float], bool, Optional[str]]:
        proximity = str(nav_context["proximity_category"])
        progress = str(nav_context["progress_trend"])
        min_lidar_norm = float(nav_context["min_lidar_normalized"])
        min_lidar_m = min_lidar_norm_to_distance_m(min_lidar_norm)
        heading_sign = self._heading_sign(nav_context)
        age = max(0, int(step_idx) - int(self.last_llm_step))

        emergency = dict(decision)
        reason: Optional[str] = None

        stop_now = (
            min_lidar_m <= 0.22
            or (
                held_step
                and proximity == "very_close"
                and progress == "getting_farther"
                and age >= max(2, self.decision_interval // 4)
            )
            or ((request_failed or parse_failed) and min_lidar_m <= 0.26)
        )

        slow_now = (
            min_lidar_m <= 0.28
            or (
                held_step
                and proximity == "very_close"
                and age >= max(3, self.decision_interval // 2)
            )
            or ((request_failed or parse_failed) and proximity in {"close", "very_close"})
        )

        if stop_now:
            emergency = sanitize_decision(
                {
                    "mode": "stop",
                    "speed_scale": 0.0,
                    "turn_scale": max(float(decision.get("turn_scale", 1.0)), 1.40),
                    "turn_bias": 0.05 * heading_sign,
                },
                nav_context,
            )
            reason = "emergency_stop"
        elif slow_now:
            emergency = sanitize_decision(
                {
                    "mode": "cautious",
                    "speed_scale": min(float(decision.get("speed_scale", 1.0)), 0.15 if proximity == "very_close" else 0.30),
                    "turn_scale": max(float(decision.get("turn_scale", 1.0)), 1.30 if proximity == "very_close" else 1.15),
                    "turn_bias": 0.04 * heading_sign if heading_sign != 0.0 else float(decision.get("turn_bias", 0.0)),
                },
                nav_context,
            )
            reason = "emergency_slowdown"

        changed = emergency != sanitize_decision(decision, nav_context)
        if changed:
            self.emergency_override_count += 1
            if reason == "emergency_stop":
                self.emergency_stop_count += 1
        return emergency, changed, reason

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        nav_context = build_nav_context_from_obs(obs, step_idx, episode_idx)
        self.last_nav_context = nav_context

        if step_idx % self.decision_interval != 0 and step_idx > 0:
            decision, _, _ = self._apply_emergency_filter(
                dict(self.last_decision),
                nav_context,
                step_idx,
                held_step=True,
            )
            self.last_decision = decision
            return dict(self.last_decision), None

        if is_safe_nav_context(nav_context):
            self.last_decision = dict(DEFAULT_DECISION)
            self.last_llm_step = int(step_idx)
            return dict(self.last_decision), None

        prompt = self.build_prompt(nav_context)
        t0 = time.time()
        raw_response, request_failed, request_error = self.call_model(prompt)
        duration = float(time.time() - t0)
        decision, parse_failed, parse_error = self.parse_response(raw_response)

        used_safe_fallback = False
        if request_failed or parse_failed:
            decision = self._safe_fallback_decision(nav_context)
            self.last_decision = decision
            used_safe_fallback = True
            self.safe_fallback_count += 1

        decision, emergency_applied, emergency_reason = self._apply_emergency_filter(
            decision,
            nav_context,
            step_idx,
            request_failed=request_failed,
            parse_failed=parse_failed,
            held_step=False,
        )

        self.last_decision = decision
        self.last_llm_step = int(step_idx)
        self.total_calls += 1
        self.episode_query_count += 1
        self.call_latencies.append(duration)
        self.request_failures += int(request_failed)
        self.parse_failures += int(parse_failed)

        log_entry = {
            "episode": int(episode_idx) if episode_idx is not None else None,
            "step": int(step_idx),
            "nav_context": nav_context,
            "raw_response": raw_response,
            "decision": decision,
            "decision_duration_s": duration,
            "request_failed": bool(request_failed),
            "parse_failed": bool(parse_failed),
            "used_safe_fallback": bool(used_safe_fallback),
            "emergency_applied": bool(emergency_applied),
            "emergency_reason": emergency_reason,
        }
        if request_error:
            log_entry["request_error"] = request_error
        if parse_error:
            log_entry["parse_error"] = parse_error
        self.current_episode_calls.append(log_entry)

        if log:
            return decision, log_entry
        return decision, None

    def extra_summary(self) -> Dict[str, object]:
        summary = super().extra_summary()
        summary.update(
            {
                "safe_fallback_count": int(self.safe_fallback_count),
                "safe_fallback_rate_per_call": float(self.safe_fallback_count / max(1, self.total_calls)),
                "emergency_override_count": int(self.emergency_override_count),
                "emergency_override_rate_per_call": float(self.emergency_override_count / max(1, self.total_calls)),
                "emergency_stop_count": int(self.emergency_stop_count),
            }
        )
        return summary


class ActualLLMHardenedController(ActualLLMController):
    def __init__(
        self,
        decision_interval: int,
        provider: str = DEFAULT_LLM_PROVIDER,
        model_name: Optional[str] = None,
        base_url: str = DEFAULT_LLM_BASE_URL,
        timeout_s: float = 20.0,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        super().__init__(
            decision_interval=decision_interval,
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            api_key_env=api_key_env,
        )
        self.name = "actual_llm_hardened"
        self.last_llm_step = 0
        self.consecutive_very_close_steps = 0
        self.consecutive_risk_steps = 0
        self.safe_fallback_count = 0
        self.hold_validator_interventions = 0
        self.hold_emergency_stops = 0

    def reset_run(self) -> None:
        super().reset_run()
        self.safe_fallback_count = 0
        self.hold_validator_interventions = 0
        self.hold_emergency_stops = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self.last_llm_step = 0
        self.consecutive_very_close_steps = 0
        self.consecutive_risk_steps = 0

    @staticmethod
    def _heading_sign(nav_context: Dict[str, float]) -> float:
        heading_deg = float(nav_context.get("heading_deg_to_target", 0.0))
        if abs(heading_deg) < 5.0:
            return 0.0
        return 1.0 if heading_deg > 0.0 else -1.0

    def _update_risk_counters(self, nav_context: Dict[str, float]) -> None:
        proximity = str(nav_context["proximity_category"])
        if proximity == "very_close":
            self.consecutive_very_close_steps += 1
        else:
            self.consecutive_very_close_steps = 0

        if proximity in {"close", "very_close"}:
            self.consecutive_risk_steps += 1
        else:
            self.consecutive_risk_steps = 0

    def _safe_fallback_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        proximity = str(nav_context["proximity_category"])
        progress = str(nav_context["progress_trend"])
        distance = float(nav_context["approx_distance_to_target_m"])
        heading_sign = self._heading_sign(nav_context)

        if proximity == "very_close":
            if progress == "getting_farther":
                return sanitize_decision(
                    {
                        "mode": "stop",
                        "speed_scale": 0.0,
                        "turn_scale": 1.6,
                        "turn_bias": 0.12 * heading_sign,
                    },
                    nav_context,
                )
            return sanitize_decision(
                {
                    "mode": "cautious",
                    "speed_scale": 0.15,
                    "turn_scale": 1.4,
                    "turn_bias": 0.08 * heading_sign,
                },
                nav_context,
            )

        if proximity == "close":
            if progress == "getting_farther":
                return sanitize_decision(
                    {
                        "mode": "cautious",
                        "speed_scale": 0.25,
                        "turn_scale": 1.3,
                        "turn_bias": 0.08 * heading_sign,
                    },
                    nav_context,
                )
            if progress == "stagnant":
                return sanitize_decision(
                    {
                        "mode": "cautious",
                        "speed_scale": 0.35,
                        "turn_scale": 1.2,
                        "turn_bias": 0.05 * heading_sign,
                    },
                    nav_context,
                )
            return sanitize_decision(
                {
                    "mode": "cautious",
                    "speed_scale": 0.4,
                    "turn_scale": 1.05,
                    "turn_bias": 0.0,
                },
                nav_context,
            )

        if progress == "getting_farther":
            return sanitize_decision(
                {
                    "mode": "explore",
                    "speed_scale": 0.80 if distance > 2.0 else 0.65,
                    "turn_scale": 1.25,
                    "turn_bias": 0.08 * heading_sign,
                },
                nav_context,
            )
        return dict(DEFAULT_DECISION)

    def _apply_hold_validator(
        self,
        decision: Dict[str, float],
        nav_context: Dict[str, float],
        step_idx: int,
        held_step: bool,
    ) -> Tuple[Dict[str, float], bool, List[str]]:
        hardened = dict(decision)
        reasons: List[str] = []
        proximity = str(nav_context["proximity_category"])
        progress = str(nav_context["progress_trend"])
        heading_sign = self._heading_sign(nav_context)
        stale_threshold = max(4, self.decision_interval // 2)
        age = max(0, int(step_idx) - int(self.last_llm_step))

        def strengthen_bias(min_abs: float) -> None:
            if heading_sign != 0.0:
                hardened["turn_bias"] = heading_sign * max(abs(float(hardened.get("turn_bias", 0.0))), min_abs)

        if proximity == "very_close":
            if hardened.get("mode") != "stop":
                hardened["mode"] = "cautious"

            speed_cap = 0.15 if held_step else 0.20
            turn_floor = 1.35

            if progress != "getting_closer" or self.consecutive_very_close_steps >= 2:
                speed_cap = 0.10
                turn_floor = 1.50
                strengthen_bias(0.08)
                reasons.append("very_close_escalation")

            if held_step and (
                self.consecutive_very_close_steps >= 4
                or (progress == "getting_farther" and age >= max(3, stale_threshold // 2))
            ):
                prev_mode = str(hardened.get("mode", "follow"))
                prev_speed = float(hardened.get("speed_scale", 1.0))
                hardened["mode"] = "stop"
                hardened["speed_scale"] = 0.0
                hardened["turn_scale"] = max(float(hardened.get("turn_scale", 1.0)), 1.6)
                strengthen_bias(0.10)
                reasons.append("emergency_stop")
                if prev_mode != "stop" or prev_speed > 0.0:
                    self.hold_emergency_stops += 1
            else:
                hardened["speed_scale"] = min(float(hardened.get("speed_scale", 1.0)), speed_cap)
                hardened["turn_scale"] = max(float(hardened.get("turn_scale", 1.0)), turn_floor)
                reasons.append("very_close_clamp")

        elif proximity == "close" and (progress != "getting_closer" or (held_step and age >= stale_threshold)):
            hardened["mode"] = "cautious" if hardened.get("mode") != "stop" else hardened["mode"]
            speed_cap = 0.35
            turn_floor = 1.2
            if progress == "getting_farther" or self.consecutive_risk_steps >= max(3, stale_threshold // 2):
                speed_cap = 0.25
                turn_floor = 1.3
                strengthen_bias(0.06)
            hardened["speed_scale"] = min(float(hardened.get("speed_scale", 1.0)), speed_cap)
            hardened["turn_scale"] = max(float(hardened.get("turn_scale", 1.0)), turn_floor)
            reasons.append("close_nonprogress_clamp")

        elif held_step and progress == "getting_farther" and age >= stale_threshold and hardened.get("mode") == "follow":
            hardened["mode"] = "explore"
            hardened["turn_scale"] = max(float(hardened.get("turn_scale", 1.0)), 1.15)
            strengthen_bias(0.04)
            reasons.append("stale_progress_turn")

        new_decision = sanitize_decision(hardened, nav_context)
        changed = new_decision != sanitize_decision(decision, nav_context)
        if changed:
            self.hold_validator_interventions += 1
        return new_decision, changed, reasons

    def maybe_update(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_idx: Optional[int] = None,
        log: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]]]:
        nav_context = build_nav_context_from_obs(obs, step_idx, episode_idx)
        self.last_nav_context = nav_context
        self._update_risk_counters(nav_context)

        if step_idx % self.decision_interval != 0 and step_idx > 0:
            decision, _, _ = self._apply_hold_validator(
                dict(self.last_decision),
                nav_context,
                step_idx,
                held_step=True,
            )
            self.last_decision = decision
            return dict(self.last_decision), None

        if is_safe_nav_context(nav_context):
            self.last_decision = dict(DEFAULT_DECISION)
            self.last_llm_step = int(step_idx)
            return dict(self.last_decision), None

        prompt = self.build_prompt(nav_context)
        t0 = time.time()
        raw_response, request_failed, request_error = self.call_model(prompt)
        duration = float(time.time() - t0)
        decision, parse_failed, parse_error = self.parse_response(raw_response)

        used_safe_fallback = False
        if request_failed or parse_failed:
            decision = self._safe_fallback_decision(nav_context)
            self.last_decision = decision
            used_safe_fallback = True
            self.safe_fallback_count += 1

        decision, validator_changed, validator_reasons = self._apply_hold_validator(
            decision,
            nav_context,
            step_idx,
            held_step=False,
        )

        self.last_decision = decision
        self.last_llm_step = int(step_idx)
        self.total_calls += 1
        self.episode_query_count += 1
        self.call_latencies.append(duration)
        self.request_failures += int(request_failed)
        self.parse_failures += int(parse_failed)

        log_entry = {
            "episode": int(episode_idx) if episode_idx is not None else None,
            "step": int(step_idx),
            "nav_context": nav_context,
            "raw_response": raw_response,
            "decision": decision,
            "decision_duration_s": duration,
            "request_failed": bool(request_failed),
            "parse_failed": bool(parse_failed),
            "used_safe_fallback": bool(used_safe_fallback),
            "validator_applied": bool(validator_changed),
            "validator_reasons": validator_reasons,
        }
        if request_error:
            log_entry["request_error"] = request_error
        if parse_error:
            log_entry["parse_error"] = parse_error
        self.current_episode_calls.append(log_entry)

        if log:
            return decision, log_entry
        return decision, None

    def extra_summary(self) -> Dict[str, object]:
        summary = super().extra_summary()
        summary.update(
            {
                "safe_fallback_count": int(self.safe_fallback_count),
                "safe_fallback_rate_per_call": float(self.safe_fallback_count / max(1, self.total_calls)),
                "hold_validator_interventions": int(self.hold_validator_interventions),
                "hold_emergency_stops": int(self.hold_emergency_stops),
            }
        )
        return summary


class ActualLLMGuardedController(ActualLLMHardenedController):
    def __init__(
        self,
        decision_interval: int,
        provider: str = DEFAULT_LLM_PROVIDER,
        model_name: Optional[str] = None,
        base_url: str = DEFAULT_LLM_BASE_URL,
        timeout_s: float = 20.0,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        super().__init__(
            decision_interval=decision_interval,
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            api_key_env=api_key_env,
        )
        self.name = "actual_llm_guarded"

    def _safe_fallback_decision(self, nav_context: Dict[str, float]) -> Dict[str, float]:
        proximity = str(nav_context["proximity_category"])
        progress = str(nav_context["progress_trend"])
        distance = float(nav_context["approx_distance_to_target_m"])
        heading_sign = self._heading_sign(nav_context)

        if proximity == "very_close":
            if progress == "getting_farther":
                return sanitize_decision(
                    {
                        "mode": "cautious",
                        "speed_scale": 0.10,
                        "turn_scale": 1.45,
                        "turn_bias": 0.06 * heading_sign,
                    },
                    nav_context,
                )
            return sanitize_decision(
                {
                    "mode": "cautious",
                    "speed_scale": 0.20,
                    "turn_scale": 1.30,
                    "turn_bias": 0.04 * heading_sign,
                },
                nav_context,
            )

        if proximity == "close":
            if progress == "getting_farther":
                return sanitize_decision(
                    {
                        "mode": "cautious",
                        "speed_scale": 0.35,
                        "turn_scale": 1.20,
                        "turn_bias": 0.04 * heading_sign,
                    },
                    nav_context,
                )
            if progress == "stagnant":
                return sanitize_decision(
                    {
                        "mode": "cautious",
                        "speed_scale": 0.45,
                        "turn_scale": 1.10,
                        "turn_bias": 0.02 * heading_sign,
                    },
                    nav_context,
                )
            return sanitize_decision(
                {
                    "mode": "cautious",
                    "speed_scale": 0.50,
                    "turn_scale": 1.0,
                    "turn_bias": 0.0,
                },
                nav_context,
            )

        if progress == "getting_farther":
            return sanitize_decision(
                {
                    "mode": "explore",
                    "speed_scale": 0.90 if distance > 2.0 else 0.75,
                    "turn_scale": 1.15,
                    "turn_bias": 0.04 * heading_sign,
                },
                nav_context,
            )
        return dict(DEFAULT_DECISION)

    def _apply_hold_validator(
        self,
        decision: Dict[str, float],
        nav_context: Dict[str, float],
        step_idx: int,
        held_step: bool,
    ) -> Tuple[Dict[str, float], bool, List[str]]:
        guarded = dict(decision)
        reasons: List[str] = []
        proximity = str(nav_context["proximity_category"])
        progress = str(nav_context["progress_trend"])
        heading_sign = self._heading_sign(nav_context)
        age = max(0, int(step_idx) - int(self.last_llm_step))
        stale_threshold = max(self.decision_interval, 20)
        min_lidar = float(nav_context.get("min_lidar_normalized", 0.0))

        def strengthen_bias(min_abs: float) -> None:
            if heading_sign != 0.0:
                guarded["turn_bias"] = heading_sign * max(abs(float(guarded.get("turn_bias", 0.0))), min_abs)

        if proximity == "very_close":
            if guarded.get("mode") != "stop":
                guarded["mode"] = "cautious"

            guarded["speed_scale"] = min(float(guarded.get("speed_scale", 1.0)), 0.25 if not held_step else 0.20)
            guarded["turn_scale"] = max(float(guarded.get("turn_scale", 1.0)), 1.20)
            reasons.append("very_close_guard")

            if progress != "getting_closer" or self.consecutive_very_close_steps >= 3:
                guarded["speed_scale"] = min(float(guarded.get("speed_scale", 1.0)), 0.15)
                guarded["turn_scale"] = max(float(guarded.get("turn_scale", 1.0)), 1.35)
                strengthen_bias(0.04)
                reasons.append("very_close_slow_turn")

            if held_step and progress == "getting_farther" and self.consecutive_very_close_steps >= 5 and age >= stale_threshold:
                guarded["speed_scale"] = min(float(guarded.get("speed_scale", 1.0)), 0.08)
                guarded["turn_scale"] = max(float(guarded.get("turn_scale", 1.0)), 1.45)
                strengthen_bias(0.06)
                reasons.append("stale_very_close_recovery")

            if held_step and min_lidar < -0.45 and progress == "getting_farther" and self.consecutive_very_close_steps >= 7 and age >= stale_threshold:
                prev_mode = str(guarded.get("mode", "follow"))
                prev_speed = float(guarded.get("speed_scale", 1.0))
                guarded["mode"] = "stop"
                guarded["speed_scale"] = 0.0
                guarded["turn_scale"] = max(float(guarded.get("turn_scale", 1.0)), 1.5)
                strengthen_bias(0.06)
                reasons.append("rare_emergency_stop")
                if prev_mode != "stop" or prev_speed > 0.0:
                    self.hold_emergency_stops += 1

        elif held_step and proximity == "close" and progress == "getting_farther" and age >= stale_threshold:
            guarded["mode"] = "cautious"
            guarded["speed_scale"] = min(float(guarded.get("speed_scale", 1.0)), 0.40)
            guarded["turn_scale"] = max(float(guarded.get("turn_scale", 1.0)), 1.15)
            strengthen_bias(0.03)
            reasons.append("close_recovery")

        new_decision = sanitize_decision(guarded, nav_context)
        changed = new_decision != sanitize_decision(decision, nav_context)
        if changed:
            self.hold_validator_interventions += 1
        return new_decision, changed, reasons


def outcome_from_reason(reason: str) -> str:
    if "target_reached" in reason:
        return "reached"
    if "collision" in reason:
        return "collision"
    return "timeout"


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr[0])
        return value, value

    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, arr.size, size=arr.size)
        samples.append(float(np.mean(arr[indices])))
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def rate_stats(
    records: Sequence[EpisodeRecord],
    outcome_name: str,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, float]:
    flags = np.asarray(
        [1.0 if record.outcome == outcome_name else 0.0 for record in records],
        dtype=np.float64,
    )
    ci_low, ci_high = bootstrap_ci(flags, n_bootstrap=n_bootstrap, random_state=random_state)
    return {
        "count": int(np.sum(flags)),
        "rate": float(np.mean(flags)),
        "std": float(np.std(flags)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def metric_stats(
    values: Sequence[float],
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(arr, n_bootstrap=n_bootstrap, random_state=random_state)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def exact_mcnemar(
    reference_records: Sequence[EpisodeRecord],
    candidate_records: Sequence[EpisodeRecord],
    outcome_name: str = "collision",
) -> Dict[str, float]:
    if len(reference_records) != len(candidate_records):
        raise ValueError("McNemar comparison requires paired episode lists with equal length.")

    ref_flags = np.asarray([record.outcome == outcome_name for record in reference_records], dtype=bool)
    cand_flags = np.asarray([record.outcome == outcome_name for record in candidate_records], dtype=bool)

    b = int(np.sum(ref_flags & ~cand_flags))
    c = int(np.sum(~ref_flags & cand_flags))
    discordant = b + c

    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue
    )

    return {
        "discordant_ref_yes_candidate_no": b,
        "discordant_ref_no_candidate_yes": c,
        "p_value": p_value,
    }


def summarize_controller(
    records: Sequence[EpisodeRecord],
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, object]:
    if not records:
        return {}

    summary = {
        "episodes": len(records),
        "reached": rate_stats(records, "reached", n_bootstrap=n_bootstrap, random_state=random_state),
        "collision": rate_stats(records, "collision", n_bootstrap=n_bootstrap, random_state=random_state),
        "timeout": rate_stats(records, "timeout", n_bootstrap=n_bootstrap, random_state=random_state),
        "steps": metric_stats([record.steps for record in records], n_bootstrap, random_state),
        "path_length_m": metric_stats([record.path_length_m for record in records], n_bootstrap, random_state),
        "efficiency": metric_stats([record.efficiency for record in records], n_bootstrap, random_state),
        "total_reward": metric_stats([record.total_reward for record in records], n_bootstrap, random_state),
        "decision_updates": metric_stats([record.decision_updates for record in records], n_bootstrap, random_state),
    }
    return summary


def summarize_controller_fast(records: Sequence[EpisodeRecord]) -> Dict[str, object]:
    if not records:
        return {}

    outcome_counts = {"reached": 0, "collision": 0, "timeout": 0}
    for record in records:
        outcome_counts[record.outcome] = outcome_counts.get(record.outcome, 0) + 1

    def mean_of(attr: str) -> float:
        values = [float(getattr(record, attr)) for record in records]
        return float(np.mean(values))

    n = len(records)
    return {
        "episodes_completed": n,
        "reached": {"count": outcome_counts.get("reached", 0), "rate": outcome_counts.get("reached", 0) / n},
        "collision": {"count": outcome_counts.get("collision", 0), "rate": outcome_counts.get("collision", 0) / n},
        "timeout": {"count": outcome_counts.get("timeout", 0), "rate": outcome_counts.get("timeout", 0) / n},
        "steps_mean": mean_of("steps"),
        "path_length_m_mean": mean_of("path_length_m"),
        "efficiency_mean": mean_of("efficiency"),
        "total_reward_mean": mean_of("total_reward"),
        "decision_updates_mean": mean_of("decision_updates"),
    }


def build_status_payload(
    *,
    state: str,
    config: Dict[str, object],
    controller_names: Sequence[str],
    current_controller: Optional[str] = None,
    controller_index: int = 0,
    total_controllers: int = 0,
    current_episode: int = 0,
    total_episodes: int = 0,
    completed_controllers: Optional[Sequence[str]] = None,
    controller_progress: Optional[Dict[str, object]] = None,
    controller_aux: Optional[Dict[str, object]] = None,
    message: Optional[str] = None,
) -> Dict[str, object]:
    payload = {
        "state": state,
        "updated_at": utc_now_iso(),
        "config": config,
        "current_controller": current_controller,
        "controller_index": controller_index,
        "total_controllers": total_controllers,
        "current_episode": current_episode,
        "total_episodes": total_episodes,
        "completed_controllers": list(completed_controllers or []),
        "controller_progress": controller_progress or {},
        "controller_aux": controller_aux or {},
    }
    if message:
        payload["message"] = message
    return payload


def save_episode_records(path: Path, records: Sequence[EpisodeRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def run_controller(
    model: PPO,
    controller: BaseController,
    scenario_seeds: Iterable[int],
    *,
    env_variant: str = "baseline",
    lidar_noise_std: float = 0.0,
    beam_dropout_rate: float = 0.0,
    beam_dropout_fill: str = "max",
    deterministic: bool = True,
    progress_every: int = 100,
    progress_callback=None,
) -> List[EpisodeRecord]:
    env = EvaluationEnv(
        env_variant=env_variant,
        lidar_noise_std=lidar_noise_std,
        beam_dropout_rate=beam_dropout_rate,
        beam_dropout_fill=beam_dropout_fill,
    )
    records: List[EpisodeRecord] = []
    scenario_seeds = list(scenario_seeds)
    total_episodes = len(scenario_seeds)
    controller.reset_run()

    for episode_idx, episode_seed in enumerate(scenario_seeds):
        controller.reset_episode()
        env.rng = np.random.default_rng(int(episode_seed))
        obs, _ = env.reset()

        initial_distance = float(obs[4]) * 100.0
        path_length = 0.0
        total_reward = 0.0
        steps = 0
        done = False
        last_reason = ""

        while not done:
            rl_action, _ = model.predict(obs, deterministic=deterministic)
            rl_action = np.asarray(rl_action, dtype=np.float32).reshape(-1)

            decision, _ = controller.maybe_update(
                step_idx=steps,
                obs=obs,
                episode_idx=episode_idx,
                log=isinstance(controller, ActualLLMController),
            )
            applied_action = controller.apply_decision(rl_action, decision)
            path_length += linear_velocity_from_action(applied_action) * DT

            obs, reward, terminated, truncated, info = env.step(applied_action)
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            last_reason = str(info.get("reason", ""))

        final_distance = float(np.linalg.norm(env.agent_pos[0:2] - env.target_pos))
        progress = max(0.0, initial_distance - final_distance)
        efficiency = progress / path_length if path_length > 1e-6 else 0.0

        records.append(
            EpisodeRecord(
                controller=controller.name,
                episode=episode_idx,
                seed=int(episode_seed),
                outcome=outcome_from_reason(last_reason),
                reason=last_reason,
                steps=steps,
                total_reward=total_reward,
                path_length_m=path_length,
                efficiency=efficiency,
                initial_distance_m=initial_distance,
                final_distance_m=final_distance,
                decision_updates=controller.episode_query_count,
            )
        )
        controller.finalize_episode(records[-1])

        if progress_every > 0 and ((episode_idx + 1) % progress_every == 0 or (episode_idx + 1) == total_episodes):
            print(
                f"[PROGRESS] controller={controller.name} "
                f"episode={episode_idx + 1}/{total_episodes}"
            )
            if progress_callback is not None:
                progress_callback(controller, records, episode_idx + 1, total_episodes)

    env.close()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified paired baseline evaluation for PPO safety supervisors.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to PPO model zip. Defaults to the first existing local candidate.",
    )
    parser.add_argument(
        "--teacher-trace",
        type=str,
        default=DEFAULT_TEACHER_TRACE,
        help="Teacher LLM trace JSON for training the imitation MLP.",
    )
    parser.add_argument(
        "--independent-mlp-model",
        type=str,
        default="independent_nav_mlp_runs/search_supervised_k30_v1/model.pkl",
        help="Path to a search-supervised, non-LLM MLP artifact for mlp_independent.",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="rl_only,gate_only,rule_based,mlp_imitation",
        help=(
            "Comma-separated controllers to run. Supports apf_reactive, cbf_style, cbf_formal, actual_llm, "
            "actual_llm_async, actual_llm_emergency, actual_llm_hardened, mlp_independent, and "
            "actual_llm_guarded."
        ),
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of paired evaluation episodes.")
    parser.add_argument("--decision-interval", type=int, default=30, help="Shared high-level decision interval.")
    parser.add_argument("--scenario-seed", type=int, default=42, help="Seed for generating episode seeds.")
    parser.add_argument(
        "--scenario-seeds-file",
        type=str,
        default=None,
        help="Optional JSON file with an existing episode seed bank to replay exactly.",
    )
    parser.add_argument("--mlp-random-state", type=int, default=42, help="Random state for imitation MLP.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for CI estimates.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N episodes.")
    parser.add_argument(
        "--env-variant",
        type=str,
        default="baseline",
        choices=sorted(SUPPORTED_ENV_VARIANTS),
        help="Environment/layout variant for evaluation episodes.",
    )
    parser.add_argument(
        "--lidar-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise standard deviation in meters added to LiDAR observations seen by the policy.",
    )
    parser.add_argument(
        "--beam-dropout-rate",
        type=float,
        default=0.0,
        help="Fraction of LiDAR beams randomly replaced at each step in the observed input.",
    )
    parser.add_argument(
        "--beam-dropout-fill",
        type=str,
        default="max",
        choices=sorted(SUPPORTED_BEAM_DROPOUT_FILLS),
        help="Replacement value for dropped LiDAR beams: max, min, or mean range.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=DEFAULT_LLM_PROVIDER,
        choices=["local_ollama", "gemini"],
        help="Backend for actual_llm controller.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Model name for actual_llm controller. If omitted, provider-specific default is used.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=DEFAULT_LLM_BASE_URL,
        help="Base URL for local Ollama-compatible backend.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for each live LLM request.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        type=str,
        default="GEMINI_API_KEY",
        help="Environment variable to read Gemini API key from when using gemini provider.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="baseline_eval_outputs",
        help="Directory for summary JSON and per-episode CSV files.",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default="status.json",
        help="Filename for runtime progress/status snapshots inside outdir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller_names = [name.strip() for name in args.controllers.split(",") if name.strip()]
    llm_model = args.llm_model or default_llm_model(args.llm_provider)

    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else resolve_existing_path(DEFAULT_MODEL_CANDIDATES)
    )
    teacher_trace_path = Path(args.teacher_trace).expanduser().resolve()
    independent_mlp_model_path = Path(args.independent_mlp_model).expanduser().resolve()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    status_path = outdir / args.status_file

    if args.scenario_seeds_file:
        with open(Path(args.scenario_seeds_file).expanduser().resolve(), "r", encoding="utf-8") as handle:
            scenario_seeds = np.asarray(json.load(handle), dtype=np.int64)
        if len(scenario_seeds) != args.episodes:
            raise ValueError(
                f"--episodes ({args.episodes}) does not match seed bank length ({len(scenario_seeds)})."
            )
    else:
            scenario_rng = np.random.default_rng(args.scenario_seed)
            scenario_seeds = scenario_rng.integers(0, 2**31 - 1, size=args.episodes, dtype=np.int64)
    with open(outdir / "scenario_seeds.json", "w", encoding="utf-8") as handle:
        json.dump([int(seed) for seed in scenario_seeds.tolist()], handle, indent=2)

    config = {
        "model_path": str(model_path),
        "teacher_trace": str(teacher_trace_path),
        "independent_mlp_model": str(independent_mlp_model_path),
        "controllers": controller_names,
        "episodes": int(args.episodes),
        "decision_interval": int(args.decision_interval),
        "scenario_seed": int(args.scenario_seed),
        "scenario_seeds_file": args.scenario_seeds_file,
        "bootstrap_samples": int(args.bootstrap_samples),
        "llm_provider": args.llm_provider,
        "llm_model": llm_model,
        "llm_base_url": args.llm_base_url,
        "llm_timeout": args.llm_timeout,
        "env_variant": args.env_variant,
        "lidar_noise_std": float(args.lidar_noise_std),
        "beam_dropout_rate": float(args.beam_dropout_rate),
        "beam_dropout_fill": args.beam_dropout_fill,
        "progress_every": int(args.progress_every),
    }
    write_json_atomic(
        status_path,
        build_status_payload(
            state="starting",
            config=config,
            controller_names=controller_names,
            total_controllers=len(controller_names),
            total_episodes=int(args.episodes),
            message="Initializing evaluator.",
        ),
    )

    model = PPO.load(str(model_path), device="cpu")

    imitation_model = None
    if "mlp_imitation" in controller_names:
        imitation_model = ImitationMLP(
            teacher_trace_path=teacher_trace_path,
            random_state=args.mlp_random_state,
        )
        imitation_model.fit()

    independent_mlp = None
    if "mlp_independent" in controller_names:
        independent_mlp = IndependentDecisionMLP(model_path=independent_mlp_model_path)
        independent_mlp.fit_from_artifact()

    controller_map: Dict[str, BaseController] = {
        "rl_only": RLOnlyController(),
        "gate_only": GateOnlyController(decision_interval=args.decision_interval),
        "rule_based": RuleBasedController(decision_interval=args.decision_interval),
        "apf_reactive": APFReactiveController(),
        "cbf_style": CBFStyleController(),
        "cbf_formal": FormalCBFController(),
    }
    if imitation_model is not None:
        controller_map["mlp_imitation"] = MLPImitationController(
            decision_interval=args.decision_interval,
            imitation_model=imitation_model,
        )
    if independent_mlp is not None:
        controller_map["mlp_independent"] = IndependentMLPController(
            decision_interval=args.decision_interval,
            independent_model=independent_mlp,
        )
    if "actual_llm" in controller_names:
        controller_map["actual_llm"] = ActualLLMController(
            decision_interval=args.decision_interval,
            provider=args.llm_provider,
            model_name=llm_model,
            base_url=args.llm_base_url,
            timeout_s=args.llm_timeout,
            api_key_env=args.llm_api_key_env,
        )
    if "actual_llm_async" in controller_names:
        controller_map["actual_llm_async"] = ActualLLMAsyncController(
            decision_interval=args.decision_interval,
            provider=args.llm_provider,
            model_name=llm_model,
            base_url=args.llm_base_url,
            timeout_s=args.llm_timeout,
            api_key_env=args.llm_api_key_env,
        )
    if "actual_llm_emergency" in controller_names:
        controller_map["actual_llm_emergency"] = ActualLLMEmergencyController(
            decision_interval=args.decision_interval,
            provider=args.llm_provider,
            model_name=llm_model,
            base_url=args.llm_base_url,
            timeout_s=args.llm_timeout,
            api_key_env=args.llm_api_key_env,
        )
    if "actual_llm_hardened" in controller_names:
        controller_map["actual_llm_hardened"] = ActualLLMHardenedController(
            decision_interval=args.decision_interval,
            provider=args.llm_provider,
            model_name=llm_model,
            base_url=args.llm_base_url,
            timeout_s=args.llm_timeout,
            api_key_env=args.llm_api_key_env,
        )
    if "actual_llm_guarded" in controller_names:
        controller_map["actual_llm_guarded"] = ActualLLMGuardedController(
            decision_interval=args.decision_interval,
            provider=args.llm_provider,
            model_name=llm_model,
            base_url=args.llm_base_url,
            timeout_s=args.llm_timeout,
            api_key_env=args.llm_api_key_env,
        )

    unknown = [name for name in controller_names if name not in controller_map]
    if unknown:
        raise ValueError(f"Unknown controller names: {unknown}")

    results: Dict[str, List[EpisodeRecord]] = {}
    completed_controllers: List[str] = []

    def persist_runtime_status(
        *,
        state: str,
        current_controller: Optional[str],
        controller_index: int,
        current_episode: int,
        message: Optional[str] = None,
    ) -> None:
        controller_progress = {
            name: summarize_controller_fast(records)
            for name, records in results.items()
            if records
        }
        controller_aux = {
            name: aux
            for name, aux in (
                (name, controller_map[name].extra_summary())
                for name in controller_names
            )
            if aux
        }
        write_json_atomic(
            status_path,
            build_status_payload(
                state=state,
                config=config,
                controller_names=controller_names,
                current_controller=current_controller,
                controller_index=controller_index,
                total_controllers=len(controller_names),
                current_episode=current_episode,
                total_episodes=int(args.episodes),
                completed_controllers=completed_controllers,
                controller_progress=controller_progress,
                controller_aux=controller_aux,
                message=message,
            ),
        )

    def make_progress_callback(controller_name: str, controller_index: int):
        def _callback(controller: BaseController, records: List[EpisodeRecord], current_episode: int, total_episodes: int) -> None:
            results[controller_name] = list(records)
            save_episode_records(outdir / f"{controller_name}_episodes_partial.csv", records)
            controller.save_extra_outputs(outdir)
            persist_runtime_status(
                state="running",
                current_controller=controller_name,
                controller_index=controller_index,
                current_episode=current_episode,
                message=f"Running {controller_name}: {current_episode}/{total_episodes} episodes completed.",
            )
        return _callback

    try:
        for controller_index, controller_name in enumerate(controller_names, start=1):
            controller = controller_map[controller_name]
            print(f"[RUN] controller={controller_name} episodes={args.episodes}")
            persist_runtime_status(
                state="running",
                current_controller=controller_name,
                controller_index=controller_index,
                current_episode=0,
                message=f"Starting controller {controller_name}.",
            )
            records = run_controller(
                model,
                controller,
                scenario_seeds,
                env_variant=args.env_variant,
                lidar_noise_std=args.lidar_noise_std,
                beam_dropout_rate=args.beam_dropout_rate,
                beam_dropout_fill=args.beam_dropout_fill,
                progress_every=args.progress_every,
                progress_callback=make_progress_callback(controller_name, controller_index),
            )
            results[controller_name] = records
            save_episode_records(outdir / f"{controller_name}_episodes.csv", records)
            controller.save_extra_outputs(outdir)
            completed_controllers.append(controller_name)
            persist_runtime_status(
                state="running",
                current_controller=controller_name,
                controller_index=controller_index,
                current_episode=int(args.episodes),
                message=f"Completed controller {controller_name}.",
            )

        controller_summaries = {
            name: summarize_controller(
                records,
                n_bootstrap=args.bootstrap_samples,
                random_state=args.scenario_seed,
            )
            for name, records in results.items()
        }
        controller_aux = {
            name: aux
            for name, aux in (
                (name, controller_map[name].extra_summary())
                for name in controller_names
            )
            if aux
        }
        summary = {
            "config": config,
            "teacher_model": None if imitation_model is None else imitation_model.training_summary,
            "controller_summaries": controller_summaries,
            "controller_aux": controller_aux,
            "paired_tests_vs_rl_only": {},
        }

        reference_records = results.get("rl_only")
        if reference_records:
            for name, records in results.items():
                if name == "rl_only":
                    continue
                summary["paired_tests_vs_rl_only"][name] = {
                    "collision_mcnemar": exact_mcnemar(reference_records, records, outcome_name="collision"),
                    "reached_mcnemar": exact_mcnemar(reference_records, records, outcome_name="reached"),
                    "timeout_mcnemar": exact_mcnemar(reference_records, records, outcome_name="timeout"),
                }

        with open(outdir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        write_json_atomic(
            status_path,
            build_status_payload(
                state="completed",
                config=config,
                controller_names=controller_names,
                current_controller=None,
                controller_index=len(controller_names),
                total_controllers=len(controller_names),
                current_episode=int(args.episodes),
                total_episodes=int(args.episodes),
                completed_controllers=completed_controllers,
                controller_progress=controller_summaries,
                controller_aux=controller_aux,
                message="Evaluation completed successfully.",
            ),
        )

        print(json.dumps(summary["controller_summaries"], indent=2))
        if summary["paired_tests_vs_rl_only"]:
            print(json.dumps(summary["paired_tests_vs_rl_only"], indent=2))
    except KeyboardInterrupt:
        persist_runtime_status(
            state="interrupted",
            current_controller=controller_names[len(completed_controllers)] if len(completed_controllers) < len(controller_names) else None,
            controller_index=len(completed_controllers) + 1,
            current_episode=0,
            message="Evaluation interrupted by user.",
        )
        raise
    except Exception as exc:
        persist_runtime_status(
            state="failed",
            current_controller=controller_names[len(completed_controllers)] if len(completed_controllers) < len(controller_names) else None,
            controller_index=len(completed_controllers) + 1,
            current_episode=0,
            message=f"Evaluation failed: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
