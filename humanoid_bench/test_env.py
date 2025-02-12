import argparse
import pathlib
import cv2
import gymnasium as gym

import mujoco

import humanoid_bench
from .env import ROBOTS, TASKS

from humanoid_bench.traj_recorder import TrajRecorder

def test_env(args):
    kwargs = vars(args).copy()
    kwargs.pop("env")
    kwargs.pop("render_mode")
    if kwargs["keyframe"] is None:
        kwargs.pop("keyframe")
    print(f"arguments: {kwargs}")

    # Test offscreen rendering
    print(f"Test offscreen mode...")
    env = gym.make(args.env, render_mode="rgb_array", **kwargs)
    ob, _ = env.reset()
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    print(f"ac_space = {env.action_space.shape}")

    img = env.render()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_env_img.png", rgb_img)

    # Test online rendering with interactive viewer
    print(f"Test onscreen mode...")
    env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
    ob, _ = env.reset()
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
            assert (
                v.shape == env.observation_space.spaces[k].shape
            ), f"{v.shape} != {env.observation_space.spaces[k].shape}"
        assert ob.keys() == env.observation_space.spaces.keys()
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
        assert env.observation_space.shape == ob.shape
    print(f"ac_space = {env.action_space.shape}")
    # print("observation:", ob)
    env.render()
    ret = 0

    dof_names = list([mujoco.mj_id2name(env.get_wrapper_attr('model'), mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(env.get_wrapper_attr('model').njnt)])
    recorder = TrajRecorder(dof_names,
                            robot_name=env.get_wrapper_attr('robot').name,
                            save_id=args.env,
                            file_save_folder="/home/haozhechen/Projects/RoboVerse/third_party/humanoid-bench/Trajectory")
    
    while True:
        action = env.action_space.sample()
        ob, rew, terminated, truncated, info = env.step(action)
        formatted_action = {
            "dof_pos_target":{
                k:v for k,v in zip(dof_names[1:],action)
            }
        }
        formatted_state = {
            env.get_wrapper_attr('robot').name:{
                "pos": env.get_wrapper_attr('data').qpos[:3],
                "rot": env.get_wrapper_attr('data').qpos[3:7],
                "dof_pos": {k:v for k,v in zip(dof_names[1:],env.get_wrapper_attr('data').qpos[7:])}
            }
        }
        # recorder.update(formatted_action, formatted_state,verbose=True)

        img = env.render()
        ret += rew

        if args.render_mode == "rgb_array":
            cv2.imshow("test_env", img[:, :, ::-1])
            cv2.waitKey(1)

        if terminated or truncated:
            ret = 0
            env.reset()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HumanoidBench environment test")
    parser.add_argument("--env", help="e.g. h1-walk-v0")
    parser.add_argument("--keyframe", default=None)
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--mean_path", default=None)
    parser.add_argument("--var_path", default=None)
    parser.add_argument("--policy_type", default=None)
    parser.add_argument("--blocked_hands", default="False")
    parser.add_argument("--small_obs", default="False")
    parser.add_argument("--obs_wrapper", default="False")
    parser.add_argument("--sensors", default="")
    parser.add_argument("--render_mode", default="rgb_array")  # "human" or "rgb_array".
    # NOTE: to get (nicer) 'human' rendering to work, you need to fix the compatibility issue between mujoco>3.0 and gymnasium: https://github.com/Farama-Foundation/Gymnasium/issues/749
    args = parser.parse_args()

    test_env(args)