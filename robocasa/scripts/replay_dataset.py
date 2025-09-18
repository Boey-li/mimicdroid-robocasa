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
import imageio
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


def replay_episode(env, hdf5_path, ep_name, env_args: EnvArgs, 
                   video_writer=None, camera_names=None, video_skip=5, 
                   camera_height=512, camera_width=512):
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
    
    video_count = 0
    write_video = video_writer is not None

    for i, action in enumerate(actions):
        if env_args.render:
            # Render the environment
            env.render()
            time.sleep(0.05)  # Small delay for visualization

        # Step the environment
        obs, reward, done, info = env.step(action)

        # Video recording from environment
        if write_video and video_count % video_skip == 0:
            video_img = []
            for cam_name in camera_names:
                try:
                    # Render camera view
                    im = env.sim.render(
                        height=camera_height, width=camera_width, camera_name=cam_name
                    )[::-1]  # Flip vertically to match standard image format
                    video_img.append(im)
                except Exception as e:
                    print(colored(f"Warning: Could not render camera {cam_name}: {e}", "yellow"))
                    # Add a black frame as placeholder
                    black_frame = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
                    video_img.append(black_frame)
            
            if video_img:
                # Concatenate images horizontally
                frame = np.concatenate(video_img, axis=1)
                video_writer.append_data(frame)

        video_count += 1

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

    if write_video:
        print(colored(f"Recorded {video_count} frames for episode {ep_name}", "green"))

    return success


###############################################################################
###                                 Main                                    ###
###############################################################################

def get_playback_args():
    parser = argparse.ArgumentParser(description="Replay dataset trajectories from HDF5 files"
                                     )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Path to HDF5 dataset dataset"
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
        choices=["DemoTwoHand", "GR1TwoHand"],
    )

    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Render the environment during replay"
    )

    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print verbose output"
    )

    parser.add_argument(
        "--max_episodes",
        type=int,
        default=1,
        help="Maximum number of episodes to replay",
    )

    parser.add_argument(
        "--use_camera_obs",
        action="store_true",
        default=False,
        help="Whether to return camera observations",
    )

    parser.add_argument("--reset_mode", type=str, default=None, help="Reset mode")

    # Added by Baoyu for image observation saving
    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use_obs",
        action="store_true",
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    # Camera names for video recording from environment
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_agentview_center",
            # "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type"
    )

    # Camera resolution for video recording
    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="height of camera images for video recording",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=512,
        help="width of camera images for video recording",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_playback_args()

    # some arg checking
    write_video = args.use_obs
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + '-' + args.robots + ".mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both
    
    # Validate inputs
    if not os.path.exists(args.dataset):
        print(colored(f"Error: HDF5 file not found: {args.dataset}", "red"))
        return

    # Load dataset information
    print(colored("Loading dataset information...", "yellow"))
    hdf5_path, ep_names, env_args, ep_meta = load_dataset_info(args.dataset)

    # Create keys_info
    keys_info = create_keys_info()

    env_args_obj = EnvArgs(
        robots=args.robots,
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

    # Setup video recording if requested
    video_writer = None
    if write_video:
        print(colored(f"Setting up video recording to: {args.video_path}", "green"))
        print(colored(f"Camera names: {args.camera_names}", "green"))
        print(colored(f"Camera resolution: {args.camera_width}x{args.camera_height}", "green"))
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # Replay episodes
    success_count = 0
    total_episodes = min(args.max_episodes, len(ep_names))

    for i in range(total_episodes):
        ep_idx = (args.episode_idx + i) % len(ep_names)
        ep_name = ep_names[ep_idx]

        print(colored(f"\n{'='*50}", "cyan"))
        print(colored(f"Episode {i+1}/{total_episodes}: {ep_name}", "cyan"))
        print(colored(f"{'='*50}", "cyan"))

        replay_episode(
            env, 
            hdf5_path, 
            ep_name, 
            env_args_obj,
            video_writer=video_writer,
            camera_names=args.camera_names,
            video_skip=args.video_skip,
            camera_height=args.camera_height,
            camera_width=args.camera_width
        )

    # Close video writer if it was opened
    if video_writer is not None:
        video_writer.close()
        print(colored(f"Video saved to: {args.video_path}", "green"))

    # Print summary
    print(colored(f"\n{'='*50}", "cyan"))


if __name__ == "__main__":
    main()
