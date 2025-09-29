# train_car_keyboard_env.py
import os
import numpy as np

# --- Headless pygame (prevents window/audio init during training) ---
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# Your environment
from controller.car_env import CarEnv


class SaveBestModelCallback(BaseCallback):
    """
    Tracks mean episodic reward over the last 50 episodes (env 0)
    and saves the best model to `save_path`.
    """
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -np.inf
        self.save_path = save_path
        self.episode_rewards = []
        self.current_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # In VecEnvs, rewards/dones are arrays of len = n_envs
        reward = float(self.locals["rewards"][0])
        self.current_rewards.append(reward)

        if self.locals["dones"][0]:
            episode_reward = float(np.sum(self.current_rewards))
            self.episode_rewards.append(episode_reward)
            self.current_rewards = []
            self.episode_count += 1

            if self.episode_count % 50 == 0:
                mean_reward = float(np.mean(self.episode_rewards[-50:]))
                if self.verbose:
                    print(f"Episode {self.episode_count}: "
                          f"Mean reward over last 50 episodes: {mean_reward:.3f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    if self.verbose:
                        print(f"New best model saved with mean reward: {mean_reward:.3f}")

                self.logger.record("mean_reward/last_50_episodes", mean_reward)

        return True


# Factory to create fresh env instances (one per subprocess)
def make_env(track_center_csv = "track_data/Yas_center_pits.csv",seed = 0):
    def _init():
        # Configure your env however you like for training
        # (avoid follow camera / rendering during training)
        env = CarEnv(
            render_mode="train",
            follow_camera=False,
            follow_zoom=9.0,
            track_center_csv = "track_data/Yas_center_pits.csv",
        )
        # Optional: seed per-env for reproducibility
        if seed is not None:
            env.reset(seed=seed)
        # Validate env API once (first worker will print warnings if any)
        check_env(env, warn=True)
        return env
    return _init


if __name__ == "__main__":
    # ---- Parallel envs ----
    num_envs = 5
    # Different seeds per worker (optional)
    base_seed = 42
    envs = [make_env(track_center_csv = "track_data/Yas_center_pits.csv",seed=1),make_env(track_center_csv = "track_data/Yas_center.csv",seed=2),make_env(track_center_csv = "track_data/Yas_center_pits.csv",seed=1),make_env(track_center_csv = "track_data/Yas_center.csv",seed=2),make_env(track_center_csv = "track_data/Yas_center_pits.csv",seed=1),make_env(track_center_csv = "track_data/Yas_center.csv",seed=2)]
    env = SubprocVecEnv(envs)

    # ---- Logging (stdout + TensorBoard) ----
    log_path = "./ppo_logs/"
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["stdout", "tensorboard"])

    # ---- Model path ----
    best_model_path = "weights/keyboard"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # ---- Load or initialize PPO (same hyperparams as your example) ----
    if os.path.exists(f"{best_model_path}.zip"):
        print("Loading previously saved model...")
        model = PPO.load(best_model_path, env=env)
        model.set_logger(logger)
    else:
        print("No saved model found, initializing a new model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=20480,
            batch_size=2048,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=1.0,
            max_grad_norm=0.5,
            n_epochs=10,
            tensorboard_log=log_path,
            policy_kwargs=dict(net_arch=[512, 512, 512]),
        )
        model.set_logger(logger)

    print(model.policy)

    # ---- Train ----
    save_best_callback = SaveBestModelCallback(save_path=best_model_path, verbose=1)
    timesteps = 100_000_000
    model.learn(total_timesteps=timesteps, callback=save_best_callback)
