import os
import sys

import numpy as np
import gymnasium as gym

class HumanoidWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if "SLURM_STEP_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
        if "SLURM_JOB_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

        super().__init__(env)
        self.env = env
        self.cfg = cfg

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        return obs, reward, done, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()


def make_env(cfg):
    """
    Make Humanoid environment.
    """
    if not cfg.task.startswith("metasim_"):
        raise ValueError("Unknown task:", cfg.task)
    import metasim.scripts.train.register_gym_env

    from metasim.cfg.scenario import ScenarioMetaCfg
    from metasim.constants import SimType
    from metasim.utils.setup_util import get_robot, get_task

    # env = gym.make(
    #     id=cfg.task,
    # )

    task = get_task(cfg.task.split("metasim_")[-1])
    robot = get_robot(cfg.robot)
    scenario = ScenarioMetaCfg(task=task, robot=robot)
    num_envs = 1
    headless = cfg.headless
    sim_type =SimType(cfg.sim)

    env = gym.make(
        id='metasim',
        scenario=scenario,
        sim_type=sim_type,
        num_envs=num_envs,
        headless=headless,
    )

    # env = HumanoidWrapper(env, cfg)
    # env.max_episode_steps = env.get_wrapper_attr("max_episode_steps")

    return env
