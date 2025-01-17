import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

import gymnasium as gym

from gymnasium.wrappers import TimeLimit
import wandb
from wandb.integration.sb3 import WandbCallback

import humanoid_bench
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=20000000, type=int)
parser.add_argument("--wandb_entity", default="change this to your wandb entity, like lyt0112-peking-university", type=str)
parser.add_argument("--model_path", default="/home/descfly/humanoid-bench-transfer/models/k7teg6s0/model.zip", type=str)
parser.add_argument("--train_or_test", default="train", type=str)
ARGS = parser.parse_args()


def make_env(
    rank,
    seed=0
):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        
        env = gym.make(ARGS.env_name)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        
        env.action_space.seed(ARGS.seed + rank)
        
        return env

    return _init

class EvalCallback(BaseCallback):
    
    def __init__(self, eval_every: int = 100000, verbose: int = 0):
        super(EvalCallback, self).__init__(verbose=verbose)

        self.eval_every = eval_every
        self.eval_env = DummyVecEnv([make_env(1)])

    def _on_step(self) -> bool:
        
        if self.num_timesteps % self.eval_every == 0:
            self.record_video()

        return True
    
    def record_video(self) -> None:

        print("recording video")
        video = []

        obs = self.eval_env.reset()
        for i in range(1000):
            action = self.model.predict(obs, deterministic=True)[0]
            obs, _, _, _ = self.eval_env.step(action)
            pixels = self.eval_env.render().transpose(2,0,1)
            video.append(pixels)

        video = np.stack(video)
        wandb.log({"results/video": wandb.Video(video, fps=100, format="gif")})


class LogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.aux_rewards = {}
        self.aux_returns = {}
        for key in info_keywords:
            self.aux_rewards[key] = np.zeros(ARGS.num_envs)
            self.aux_returns[key] = deque(maxlen=100)


    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            for key in self.aux_rewards.keys():
                self.aux_rewards[key][idx] += infos[idx][key]

            if self.locals['dones'][idx]:
                for key in self.aux_rewards.keys():
                    self.aux_returns[key].append(self.aux_rewards[key][idx])
                    self.aux_rewards[key][idx] = 0
        return True

    def _on_rollout_end(self) -> None:
        
        for key in self.aux_returns.keys():
            self.logger.record("aux_returns_{}/mean".format(key), np.mean(self.aux_returns[key]))


class EpisodeLogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.returns_info = {
            "results/return": [],
            "results/episode_length": [],
            "results/success": [],
            "results/success_subtasks": [],
        }


    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            curr_info = infos[idx]
            if "episode" in curr_info:
                self.returns_info["results/return"].append(curr_info["episode"]["r"])
                self.returns_info["results/episode_length"].append(curr_info["episode"]["l"])
                cur_info_success = 0
                if "success" in curr_info:
                    cur_info_success = curr_info["success"]
                self.returns_info["results/success"].append(cur_info_success)
                cur_info_success_subtasks = 0
                if "success_subtasks" in curr_info:
                    cur_info_success_subtasks = curr_info["success_subtasks"]
                self.returns_info["results/success_subtasks"].append(cur_info_success_subtasks)
        return True

    def _on_rollout_end(self) -> None:
        
        for key in self.returns_info.keys():
            if self.returns_info[key]:
                self.logger.record(key, np.mean(self.returns_info[key]))
                self.returns_info[key] = []


def visualize_policy(model, env_name, num_episodes=5):
    """
    Visualize the trained policy by running episodes and saving videos locally and to wandb.
    
    Args:
        model: Trained PPO model
        env_name: Name of the environment
        num_episodes: Number of episodes to visualize
    """
    import os
    import imageio  # Add this import at the top of the file
    
    # Create videos directory if it doesn't exist
    os.makedirs("videos", exist_ok=True)
    
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=1000)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_frames = []
        
        while not (done or truncated):
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()
            episode_frames.append(frame)
        
        # Convert frames to video and save
        video = np.stack(episode_frames)
        
        # Save locally as MP4
        video_path = f"videos/episode_{episode}.mp4"
        imageio.mimsave(video_path, episode_frames, fps=100)
        print(f"Saved video to {video_path}")
        
        # Also log to wandb
        wandb.log({f"evaluation/episode_{episode}_video": wandb.Video(video, fps=100, format="gif")})
    
    env.close()

def train(run, model, max_steps):
    """
    Train the PPO agent.
    
    Args:
        env_name: Name of the environment
        num_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        max_steps: Maximum number of training steps
        wandb_entity: WandB entity name
    
    Returns:
        trained model
    """
    model.learn(total_timesteps=max_steps, log_interval=1, 
               callback=[WandbCallback(model_save_path=f"models/{run.id}",verbose=2), 
                        EvalCallback(), LogCallback(info_keywords=[]), 
                        EpisodeLogCallback()])
    
    model.save("ppo")
    print("Training finished")
    return model

def test(model, env_name, num_episodes=5):
    """
    Test and visualize the trained policy.
    
    Args:
        model: Trained PPO model
        env_name: Name of the environment
        num_episodes: Number of episodes to visualize
    """
    print("Starting visualization...")
    visualize_policy(model, env_name, num_episodes)
    print("Visualization completed")

def main(argv):
    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    
    run = wandb.init(
        entity=ARGS.wandb_entity,
        project="humanoid-bench",
        name=f"ppo_{ARGS.env_name}",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", 
                learning_rate=float(ARGS.learning_rate), batch_size=512)
    
    if ARGS.train_or_test == "train":
        # Train the model, comment out to use existing model
        model = train(
            run,
            model,
            ARGS.max_steps
        )
    elif ARGS.train_or_test == "test":
        # Test the trained model
        model = PPO.load(ARGS.model_path)
        test(model, ARGS.env_name)
    else:
        raise ValueError(f"Invalid value for train_or_test: {ARGS.train_or_test}")

if __name__ == '__main__':
    main(None)
