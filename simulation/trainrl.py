# ====================== PPO SABİT HİPERPARAMETRE (NO SHAPING, NO MOTIF) ======================
import os
import math
import json
import pickle
from collections import deque

import numpy as np
import torch as T
import torch.nn as nn
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# ========= Global seeds =========
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
global_seed = 42
rng = np.random.default_rng(seed=global_seed)
T.manual_seed(global_seed)
os.environ['PYTHONHASHSEED'] = str(global_seed)

# ========= Plot helpers =========
def plot_target_positions(targets, outcomes, figure_file):
    targets = np.array(targets)
    plt.figure(figsize=(6, 6))
    reached_count = outcomes.count('reached')
    collision_count = outcomes.count('collision')
    timeout_count = outcomes.count('timeout')
    for target, outcome in zip(targets, outcomes):
        if outcome == 'reached':
            plt.scatter(target[0], target[1], c='green', marker='+', s=60)
        elif outcome == 'collision':
            plt.scatter(target[0], target[1], c='red', marker='x', s=60)
        elif outcome == 'timeout':
            plt.scatter(target[0], target[1], c='blue', marker='o', s=30, alpha=0.6)
    plt.xlim(-5, 5); plt.ylim(-5, 5)
    plt.title('Target Positions')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    text_str = f"Reached: {reached_count}\nCollisions: {collision_count}\nTimeouts: {timeout_count}"
    plt.gcf().text(0.80, 0.85, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.savefig(figure_file, bbox_inches='tight')
    plt.close()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.figure()
    plt.plot(x, running_avg, label='Avg(100)')
    plt.plot(x, scores, alpha=0.5, label='Scores')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(figure_file, bbox_inches='tight')
    plt.close()

# ========= Ortam =========
class CustomEnv(gym.Env):
    GOAL_REACHED_DIST = 0.5
    COLLISION_DIST = 0.17
    LIDAR_ANGLES = np.linspace(0, 2 * np.pi, 120, endpoint=False)

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.rng = np.random.default_rng(seed=rng.integers(1e9))

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float64),
            high=np.array([1.0, 1.0], dtype=np.float64),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1] + [-1]*120, dtype=np.float32),
            high=np.array([ 1,  1,  1,  1,  1,  1] + [ 1]*120, dtype=np.float32),
            dtype=np.float32
        )

        self.wall_thickness = 0.30
        self.max_steps = 1000
        self.D = 66e-3; self.R = self.D/2; self.L = 160e-3

        # episode state
        self.agent_pos = np.array([-1.0, -1.3, np.pi/2], dtype=np.float32)
        self.target_pos = np.array([4.0, 4.0], dtype=np.float32)
        self.distance_buffer = deque(maxlen=5)
        self.MIN_LIDAR = 0.16; self.MAX_LIDAR = 1.00

        # logs
        self.path = []
        self.walls = []
        self.dynamic_walls = []
        self.all_targets = []
        self.target_outcomes = []

        # artık motif ve shaping yok
        self.step_count = 0
        self.accumulated_distance = 0.0

    # --- yardımcılar ---
    def ilerikin(self, wR, wL):
        v = 0.5 * self.R * (wR + wL)
        w = (self.R / self.L) * (wR - wL)
        return v, w

    def terskin(self, v, w):
        wL = (1 / self.R) * (v - w * self.L / 2)
        wR = (1 / self.R) * (v + w * self.L / 2)
        return wL, wR

    def lidar_readings(self):
        lidar_distances = np.full(len(self.LIDAR_ANGLES), 1.00)
        distances = np.linspace(0.01, 1.00, 100)
        for i, angle in enumerate(self.LIDAR_ANGLES):
            dx = distances * np.cos(self.agent_pos[2] + angle)
            dy = distances * np.sin(self.agent_pos[2] + angle)
            points_x = self.agent_pos[0] + dx
            points_y = self.agent_pos[1] + dy
            for wall in self.walls + self.dynamic_walls:
                in_wall = (wall[0] <= points_x) & (points_x <= wall[2]) & (wall[1] <= points_y) & (points_y <= wall[3])
                if np.any(in_wall):
                    lidar_distances[i] = distances[np.argmax(in_wall)]
                    break
        return np.clip(lidar_distances, 0.16, 1.00)

    # --- map generation (12 engel) ---
    def place_dynamic_walls(self):
        self.dynamic_walls = []
        horiz_lengths = [3.0, 2.0, 0.5, 1.5, 1.25, 0.75]
        vert_lengths  = [3.0, 2.0, 1.5, 1.25, 1.0, 0.75]
        target_n = 12
        while len(self.dynamic_walls) < target_n:
            i = len(self.dynamic_walls)
            if i < 6:
                L = horiz_lengths[i];  wall = self.random_wall(L, axis='x')
            else:
                L = vert_lengths[i-6]; wall = self.random_wall(L, axis='y')
            if self.is_valid_wall_position(wall):
                self.dynamic_walls.append(wall)

    def is_valid_wall_position(self, wall):
        threshold = 0.3
        if (wall[0]-threshold <= self.agent_pos[0] <= wall[2]+threshold) and (wall[1]-threshold <= self.agent_pos[1] <= wall[3]+threshold):
            return False
        for existing_wall in self.dynamic_walls:
            if self.distance_between_walls(wall, existing_wall) < 1.0:
                return False
        return True

    def distance_between_walls(self, wall1, wall2):
        x1, y1 = (wall1[0] + wall1[2]) / 2, (wall1[1] + wall1[3]) / 2
        x2, y2 = (wall2[0] + wall2[2]) / 2, (wall2[1] + wall2[3]) / 2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def random_wall(self, length, axis='x'):
        if axis == 'x':
            x_start = self.rng.uniform(-5 + self.wall_thickness, 5 - length)
            y_pos   = self.rng.uniform(-5 + self.wall_thickness, 5 - self.wall_thickness)
            return (x_start, y_pos, x_start + length, y_pos + self.wall_thickness)
        else:
            y_start = self.rng.uniform(-5 + self.wall_thickness, 5 - length)
            x_pos   = self.rng.uniform(-5 + self.wall_thickness, 5 - self.wall_thickness)
            return (x_pos, y_start, x_pos + self.wall_thickness, y_start + length)

    # --- env API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([
            self.rng.uniform(-3, 3),
            self.rng.uniform(-3, 3),
            self.rng.uniform(0, 2 * np.pi)
        ], dtype=np.float32)
        self.place_dynamic_walls()

        threshold = 0.3
        while True:
            self.target_pos = np.array([self.rng.uniform(-4.3, 4.3), self.rng.uniform(-4.3, 4.3)], dtype=np.float32)
            if np.linalg.norm(self.agent_pos[0:2] - self.target_pos) < 3.0:
                continue
            ok = True
            for wall in self.walls + self.dynamic_walls:
                if (wall[0]-threshold <= self.target_pos[0] <= wall[2]+threshold) and (wall[1]-threshold <= self.target_pos[1] <= wall[3]+threshold):
                    ok = False; break
            if ok: break

        self.all_targets.append(self.target_pos.copy())
        self.target_outcomes.append(None)

        self.step_count = 0
        self.accumulated_distance = 0.0
        self.path = [self.agent_pos[:2].copy()]

        distance_to_target = np.linalg.norm(self.agent_pos[0:2] - self.target_pos)
        self.distance_to_target_start = distance_to_target - 0.5
        self.distance_buffer = deque([distance_to_target] * 5, maxlen=5)

        lidar_data = self.lidar_readings()
        self.MIN_LIDAR = min(self.MIN_LIDAR, np.min(lidar_data))
        self.MAX_LIDAR = max(self.MAX_LIDAR, np.max(lidar_data))
        if self.MIN_LIDAR == self.MAX_LIDAR: self.MAX_LIDAR += 0.01
        normalized_lidar = np.clip(((lidar_data - self.MIN_LIDAR)/(self.MAX_LIDAR - self.MIN_LIDAR))*2 - 1, -1, 1)

        theta_to_target = self.calculate_theta_to_target(self.agent_pos, self.target_pos)
        sin_theta, cos_theta = np.sin(theta_to_target), np.cos(theta_to_target)
        normalized_distance = (distance_to_target) / (100)

        state = np.concatenate([[sin_theta, cos_theta, 0, 0, normalized_distance, 0.0], normalized_lidar]).astype(np.float32)
        return state, {}

    def step(self, action):
        info = {}

        # --- PRE-STEP lidar (sadece çarpışma için)
        pre_lidar = self.lidar_readings()
        pre_min_lidar = float(np.min(pre_lidar))

        # aksiyon -> hızlar (speed guard YOK)
        linear_velocity  = (action[0] + 1) * 0.11
        angular_velocity = action[1] * 2.84
        wL, wR = self.terskin(linear_velocity, angular_velocity)
        v, w  = self.ilerikin(wR, wL)

        # hareket
        dt = 0.2
        theta = self.agent_pos[2]
        self.agent_pos[0] += v * np.cos(theta) * dt
        self.agent_pos[1] += v * np.sin(theta) * dt
        self.agent_pos[2] = (self.agent_pos[2] + w * dt) % (2 * np.pi)
        self.agent_pos[0:2] = np.clip(self.agent_pos[0:2], -5 + self.wall_thickness, 5 - self.wall_thickness)
        self.path.append(self.agent_pos[:2].copy())

        # metrikler
        distance_to_target = np.linalg.norm(self.agent_pos[0:2] - self.target_pos)
        self.distance_buffer.append(distance_to_target)
        distance_diff = self.distance_buffer[-1] - self.distance_buffer[0]
        step_distance = v * dt
        self.accumulated_distance += step_distance

        # POST-STEP lidar
        lidar_data = self.lidar_readings()
        post_min_lidar = float(np.min(lidar_data))
        min_lidar_distance = min(pre_min_lidar, post_min_lidar)

        target_reached = distance_to_target < self.GOAL_REACHED_DIST
        collision = min_lidar_distance < self.COLLISION_DIST
        timeout = self.step_count >= self.max_steps - 0.1

        theta_to_target = self.calculate_theta_to_target(self.agent_pos, self.target_pos)

        # --- SABİT ana ödül (çekirdek, shaping yok) ---
        reward = self._reward(distance_to_target, target_reached, collision,
                              linear_velocity, angular_velocity, min_lidar_distance, distance_diff)

        # normalize obs
        normalized_distance = (distance_to_target) / (100)
        normalized_linear_velocity = ((linear_velocity - 0) / (0.22 - 0)) * 2 - 1
        normalized_angular_velocity = ((angular_velocity - (-2.84)) / (2.84 - (-2.84))) * 2 - 1
        self.MIN_LIDAR = min(self.MIN_LIDAR, np.min(lidar_data))
        self.MAX_LIDAR = max(self.MAX_LIDAR, np.max(lidar_data))
        if self.MIN_LIDAR == self.MAX_LIDAR: self.MAX_LIDAR += 0.01
        normalized_lidar = np.clip(((lidar_data - self.MIN_LIDAR)/(self.MAX_LIDAR - self.MIN_LIDAR))*2 - 1, -1, 1)
        normalized_distance_diff = (distance_diff / 100)

        done = target_reached or collision or timeout
        if done:
            if target_reached:
                self.target_outcomes[-1] = 'reached'; info['reason'] = 'target_reached'
            elif collision:
                self.target_outcomes[-1] = 'collision'; info['reason'] = 'collision'
            elif timeout:
                self.target_outcomes[-1] = 'timeout'; info['reason'] = 'max_steps_reached'

        truncated = bool(timeout)
        terminated = bool(target_reached or collision)
        self.step_count += 1

        state = np.concatenate([[np.sin(theta_to_target),
                                 np.cos(theta_to_target),
                                 normalized_linear_velocity, normalized_angular_velocity,
                                 normalized_distance, normalized_distance_diff], normalized_lidar]).astype(np.float32)
        return state, float(reward), terminated, truncated, info
    @property
    def robot_pos(self):
        """
        VecEnv.get_attr('robot_pos') ile erişeceğimiz,
        ajanın dünya koordinatındaki (x, y) konumu.
        """
        return self.agent_pos[:2].copy()

    @property
    def obstacles(self):
        """
        Tüm duvarları tek listede topla.
        Her eleman (x1, y1, x2, y2) dikdörtgeni.
        """
        return np.array(self.walls + self.dynamic_walls, dtype=np.float32)

    def calculate_theta_to_target(self, agent_pos, target_pos):
        delta_x = target_pos[0] - agent_pos[0]
        delta_y = target_pos[1] - agent_pos[1]
        target_theta = np.arctan2(delta_y, delta_x)
        return target_theta - agent_pos[2]

    def _reward(self, distance_to_target, target_reached, collision, linear_velocity, angular_velocity, min_lidar_distance, distance_diff):
        # Eski sabit çekirdek ödül (aynen)
        reward = 0.0
        if target_reached:
            reward += 1.0
        if collision:
            reward = -1.0
        if self.step_count >= self.max_steps-0.1:
            reward += -0.6
        if not (target_reached or collision or self.step_count >= self.max_steps):
            reward += (0.0006 * (1 / max(1e-12, 2 * distance_to_target))) \
                      - (0.00035 * (self.step_count / self.max_steps)) \
                      - (abs(angular_velocity) * 0.00018) \
                      + (-0.0004 * (distance_diff)) \
                      + (linear_velocity * 0.00022)
        return reward

# ========= Feature extractor (opsiyonel) =========
class CustomMLPWithLayerNorm(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        input_dim = observation_space.shape[0]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.Tanh()
        )
    def forward(self, x): return self.model(x)

# ========= Eğitim ilerleme & checkpoint callback =========
class CustomEpisodeCallback(BaseCallback):
    def __init__(self, max_episodes=20000, checkpoint_freq=1000, checkpoint_folder='./checkpoints_static_shaping', verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.max_episodes = max_episodes
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_folder = checkpoint_folder
        self.episode_results = {'reached': 0, 'collision': 0, 'timeout': 0}
        self.scores = []
        os.makedirs(self.checkpoint_folder, exist_ok=True)

    def _on_step(self) -> bool:
        episode_logged = False
        for env_index in range(self.training_env.num_envs):
            if self.locals['dones'][env_index]:
                self.episode_count += 1
                reason = self.locals['infos'][env_index].get('reason', '')
                if 'collision' in reason:
                    self.episode_results['collision'] += 1
                elif 'target_reached' in reason:
                    self.episode_results['reached'] += 1
                elif 'max_steps_reached' in reason or 'timeout' in reason:
                    self.episode_results['timeout'] += 1
                score = float(self.locals['infos'][env_index].get('episode', {}).get('r', 0.0))
                self.scores.append(score)

                if self.episode_count % self.checkpoint_freq == 0:
                    checkpoint_path = os.path.join(self.checkpoint_folder, f"ppo_static_shaping_ep_{self.episode_count}.zip")
                    self.model.save(checkpoint_path)
                    vecnorm_path = os.path.join(self.checkpoint_folder, f"ppo_static_shaping_vecnorm_ep_{self.episode_count}.pkl")
                    self.training_env.save(vecnorm_path)
                    info_dict = {
                        "episode_count": self.episode_count,
                        "total_timesteps": self.model.num_timesteps,
                        "scores": self.scores,
                        "episode_results": self.episode_results
                    }
                    info_path = os.path.join(self.checkpoint_folder, f"ppo_static_shaping_info_ep_{self.episode_count}.pkl")
                    with open(info_path, "wb") as f:
                        pickle.dump(info_dict, f)
                    if self.verbose > 0:
                        print(f"[STATIC] Checkpoint kaydedildi: Episode {self.episode_count}")

                if not episode_logged:
                    print(f"[STATIC] Episode {self.episode_count} - Reached: {self.episode_results['reached']}, "
                          f"Collisions: {self.episode_results['collision']}, "
                          f"Timeouts: {self.episode_results['timeout']}")
                    episode_logged = True

        if self.episode_count >= self.max_episodes:
            print(f"[STATIC] Max episodes ({self.max_episodes}) reached. Stopping training.")
            return False
        return True

    def _on_training_end(self) -> None:
        all_targets = self.training_env.get_attr('all_targets')
        target_outcomes = self.training_env.get_attr('target_outcomes')
        all_targets_flat = [target for sublist in all_targets for target in sublist]
        target_outcomes_flat = [outcome for sublist in target_outcomes for outcome in sublist]
        np.savez('training_data_ppo_static_shaping.npz',
                 all_targets=np.array(all_targets_flat, dtype=np.float32),
                 target_outcomes=np.array(target_outcomes_flat, dtype=object),
                 scores=np.array(self.scores, dtype=np.float32))
        try:
            plot_target_positions(all_targets_flat, target_outcomes_flat, 'ppo_static_shaping_target_positions.png')
            x = [i + 1 for i in range(len(self.scores))]
            plot_learning_curve(x, self.scores, 'ppo_static_shaping_learning_curve.png')
        except Exception as e:
            print("[PLOT] error:", e)

# ========= Eğitim başlatıcı (SABİT HP) =========
INITIAL_LR = 0.0005129457142555377
FIXED_BATCH_SIZE = 256
FIXED_N_STEPS = 1024
FIXED_GAMMA = 0.99038378401138
FIXED_CLIP_RANGE = 0.2252229522012027
FIXED_GAE_LAMBDA = 0.9851847558275494
FIXED_N_EPOCHS = 10
FIXED_ENT_COEF = 0.0
FIXED_VF_COEF = 0.5
FIXED_MAX_GRAD_NORM = 0.5
FIXED_TARGET_KL = None
FIXED_CLIP_RANGE_VF = None

def make_env_fn():
    return Monitor(CustomEnv())

def train_ppo_static(env=None, max_episodes=20000):
    print("[STATIC] Sabit HP (shaping ve motif yok) ile model oluşturuluyor...")
    if env is None:
        env = VecNormalize(SubprocVecEnv([make_env_fn for _ in range(16)]), norm_obs=False, norm_reward=True)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[128, 128]))
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=INITIAL_LR,
        n_steps=FIXED_N_STEPS,
        batch_size=FIXED_BATCH_SIZE,
        gamma=FIXED_GAMMA,
        clip_range=FIXED_CLIP_RANGE,
        clip_range_vf=FIXED_CLIP_RANGE_VF,
        gae_lambda=FIXED_GAE_LAMBDA,
        n_epochs=FIXED_N_EPOCHS,
        ent_coef=FIXED_ENT_COEF,
        vf_coef=FIXED_VF_COEF,
        max_grad_norm=FIXED_MAX_GRAD_NORM,
        target_kl=FIXED_TARGET_KL,
        verbose=1
    )
    episode_callback = CustomEpisodeCallback(max_episodes=max_episodes, checkpoint_freq=1000, verbose=1)
    model.learn(total_timesteps=int(1e10), callback=episode_callback)
    model_save_path = "ppo_static_shaping_model.zip"
    vecnormalize_save_path = "ppo_static_shaping_vecnorm.pkl"
    model.save(model_save_path)
    env.save(vecnormalize_save_path)
    print(f"[STATIC] Model {model_save_path} ve VecNormalize {vecnormalize_save_path} olarak kaydedildi.")
    return model

if __name__ == '__main__':
    train_ppo_static(env=None, max_episodes=20000)
# ====================== END ======================

