#!/usr/bin/env python3
"""
Standalone script to replay dataset trajectories from HDF5 files.
Supports both regular robots and GR1 robots.
"""

import os
import sys
import argparse
import json
import h5py
import numpy as np
import time
from pathlib import Path
from termcolor import colored

# Add the icrt directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../icrt"))

import robocasa
import robocasa.macros as macros
from robocasa.environments.kitchen.play_env.play_env import *

from robocasa.utils.mimicdroid_utils import make_env, reset_to, get_zero_action, EnvArgs

# Disable sites for cleaner rendering
macros.SHOW_SITES = False


def load_dataset_info(hdf5_path):
    """Load dataset information and return data structure."""
    with h5py.File(hdf5_path, "r") as f:
        if "data" in f:
            data = f["data"]
        else:
            data = f

        # Get episode names
        ep_names = sorted(list(data.keys()))
        print(colored(f"Found {len(ep_names)} episodes in {hdf5_path}", "green"))

        # Get first episode info for environment setup
        first_ep = ep_names[0]
        ep_group = data[first_ep]

        # Extract environment arguments
        env_args = (
            json.loads(data.attrs["env_args"])
            if "data" in f
            else json.loads(f.attrs["env_args"])
        )
        if isinstance(env_args, str):
            env_args = json.loads(env_args)

        # Extract episode metadata
        ep_meta = ep_group.attrs.get("ep_meta", {})
        if isinstance(ep_meta, str):
            ep_meta = json.loads(ep_meta)

        return hdf5_path, ep_names, env_args, ep_meta


def create_keys_info():
    """Create keys_info structure for environment setup."""
    # Default keys based on eval_casa.py
    image_keys = []
    proprio_keys = []
    action_keys = []
    low_dim_keys = []

    return {
        "image_keys": image_keys,
        "proprio_keys": proprio_keys,
        "action_keys": action_keys,
        "low_dim_keys": low_dim_keys,
    }


def replay_episode(env, hdf5_path, ep_name, env_args: EnvArgs):
    """Replay a single episode from the dataset."""
    print(colored(f"Replaying episode: {ep_name}", "blue"))

    ep_group = h5py.File(hdf5_path, "r")["data"][ep_name]

    # Load states and actions
    states = ep_group["states"][:]  # shape: (T, state_dim)
    actions = ep_group["actions"][:]  # shape: (T, action_dim)

    # Get episode metadata
    ep_meta = ep_group.attrs.get("ep_meta", {})
    if isinstance(ep_meta, str):
        ep_meta = json.loads(ep_meta)

    # Get model file
    model_file = ep_group.attrs.get("model_file", None)

    # Get additional state info if available
    non_robot_qpos_idx = ep_group.attrs.get("non_robot_qpos_idx", None)
    qpos_state = ep_group.attrs.get("initial_qpos", None)

    # Create initial state
    initial_state = {
        "states": states[0],
        "ep_meta": ep_meta,
    }

    if model_file is not None:
        initial_state["model"] = model_file

    if non_robot_qpos_idx is not None:
        initial_state["non_robot_qpos_idx"] = non_robot_qpos_idx

    if qpos_state is not None:
        initial_state["qpos_state"] = qpos_state

    # Reset environment to initial state
    print(colored("Resetting environment to initial state...", "yellow"))
    reset_to(
        env,
        initial_state,
        replace_robot_joints=True,
        change_to_gr1="GR1" in env_args.robots,
    )

    # Replay actions
    print(colored(f"Replaying {len(actions)} actions...", "yellow"))

    for i, action in enumerate(actions):
        if env_args.render:
            # Render the environment
            env.render()
            time.sleep(0.05)  # Small delay for visualization

        # Step the environment
        obs, reward, done, info = env.step(action)

        if hasattr(env_args, "verbose") and env_args.verbose:
            print(f"Step {i+1}/{len(actions)}: Action={action[:3]}... (first 3 dims)")

        # Check for early termination
        if done:
            print(colored(f"Episode terminated early at step {i+1}", "red"))
            break

    # Check success
    success = env._check_success()
    print(
        colored(f"Episode completed. Success: {success}", "green" if success else "red")
    )

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Replay dataset trajectories from HDF5 files"
    )
    parser.add_argument(
        "--hdf5_path", type=str, required=True, help="Path to HDF5 dataset file"
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Episode index to replay (default: 0)",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default="DemoTwoHand",
        help="Robot type (default: DemoTwoHand)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during replay"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=1,
        help="Maximum number of episodes to replay",
    )
    parser.add_argument("--task_name", type=str, default=None, help="Task name")
    parser.add_argument(
        "--use_camera_obs",
        action="store_true",
        default=False,
        help="Whether to return camera observations",
    )
    parser.add_argument("--reset_mode", type=str, default=None, help="Reset mode")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.hdf5_path):
        print(colored(f"Error: HDF5 file not found: {args.hdf5_path}", "red"))
        return

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.hdf5_path):
        print(colored(f"Error: HDF5 file not found: {args.hdf5_path}", "red"))
        return

    # Load dataset information
    print(colored("Loading dataset information...", "yellow"))
    hdf5_path, ep_names, env_args, ep_meta = load_dataset_info(args.hdf5_path)

    # Create keys_info
    keys_info = create_keys_info()

    env_args_obj = EnvArgs(
        robots=args.robots,
        task_name=args.task_name,
        render=args.render,
        control_freq=20,
        controller="WHOLE_BODY_MINK_IK" if "GR1" in args.robots else None,
        use_camera_obs=args.use_camera_obs,
    )

    # Create environment
    print(colored("Creating environment...", "yellow"))
    env, env_kwargs = make_env(file_name=hdf5_path, env_args=env_args_obj)

    print(
        colored(
            f"Environment created successfully. Robot: {env_args_obj.robots}", "green"
        )
    )
    print(
        colored(
            f"Controller: {env_args_obj.controller if env_args_obj.controller else 'Default'}",
            "green",
        )
    )

    # Replay episodes
    success_count = 0
    total_episodes = min(args.max_episodes, len(ep_names))

    for i in range(total_episodes):
        ep_idx = (args.episode_idx + i) % len(ep_names)
        ep_name = ep_names[ep_idx]

        print(colored(f"\n{'='*50}", "cyan"))
        print(colored(f"Episode {i+1}/{total_episodes}: {ep_name}", "cyan"))
        print(colored(f"{'='*50}", "cyan"))

        replay_episode(env, hdf5_path, ep_name, env_args_obj)

    # Print summary
    print(colored(f"\n{'='*50}", "cyan"))


if __name__ == "__main__":
    main()
