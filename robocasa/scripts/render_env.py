import os
import cv2
import json
import argparse
import random
import numpy as np
import torch

import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.errors import RandomizationError
import robocasa.macros as macros

macros.SHOW_SITES = True


def control_seed(seed):
    """Set the random seeds (Python, NumPy, Torch) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_zero_action(env):
    """Return a zero action of the correct dimension for the first robot."""
    return np.zeros(env.robots[0].action_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="temp")
    parser.add_argument("-e", "--environment", type=str, default="SinkPlayEnv")
    parser.add_argument(
        "--robots", nargs="+", default=["DemoTwoHand"], help="Which robot(s) to use."
    )
    parser.add_argument(
        "--layout", nargs="+", type=int, default=[0], help="Layout ID(s)."
    )
    parser.add_argument(
        "--style", nargs="+", type=str, default=["0_l1"], help="Style ID(s)."
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller type, e.g. WHOLE_BODY_MINK_IK.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_ep", type=int, default=5, help="Number of episodes to run."
    )
    parser.add_argument(
        "--render", action="store_true", help="Render on-screen with mjviewer if True."
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="robot0_agentview_right",
        help="Camera name for rendering.",
    )
    args = parser.parse_args()

    # if style is string but an integer, convert to integer
    args.style = [int(s) if s.isdigit() else s for s in args.style]

    # If you have a special robot that needs a Mink IK, you can optionally assign a default controller:
    if "GR1" in args.robots:
        args.controller = "WHOLE_BODY_MINK_IK"

    # Load your controller configuration if provided
    controller_config = None
    if args.controller is not None:
        # Load composite config (handles multi-arm, etc.)
        controller_config = load_composite_controller_config(
            controller=args.controller,
            robot=args.robots[0] if isinstance(args.robots, list) else args.robots,
        )
        # Example changes if you want absolute control frames, etc.
        if "body_parts" in controller_config:
            if "right" in controller_config["body_parts"]:
                controller_config["body_parts"]["right"]["input_type"] = "absolute"
                controller_config["body_parts"]["right"]["input_ref_frame"] = "base"
            if "left" in controller_config["body_parts"]:
                controller_config["body_parts"]["left"]["input_type"] = "absolute"
                controller_config["body_parts"]["left"]["input_ref_frame"] = "base"

    # Create output directories
    env_name = args.environment
    img_dir = os.path.join(args.directory, f"{env_name}", "images")
    ep_meta_dir = os.path.join(args.directory, f"{env_name}", "ep_meta")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ep_meta_dir, exist_ok=True)

    # Create a single Robosuite environment
    env = robosuite.make(
        env_name=args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        layout_ids=args.layout,  # can be a list
        style_ids=args.style,  # can be a list
        translucent_robot=0.1,  # example parameter
        has_renderer=args.render,  # enable on-screen rendering if desired
        has_offscreen_renderer=not args.render,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",  # typical choices
    )

    # Simple loop for n_ep episodes
    for ep_idx in range(args.n_ep):
        # Seed for reproducibility
        current_seed = args.seed + ep_idx
        control_seed(current_seed)

        # Update environment metadata (if your env supports dynamic style/layout changes)
        env.set_ep_meta({"layout_ids": args.layout, "style_ids": args.style})

        for ind in range(10):
            try:
                # Reset the environment
                env.reset()
                break
            except RandomizationError as e:
                print(f"Reset failed: {e}")
                if ind == 9:
                    raise e
                pass
            except Exception as e:
                print(f"Reset failed: {e}")
                if ind == 9:
                    raise e
                pass

        # Optionally render once for your viewer
        if args.render:
            env.render()

        # Collect zero action
        zero_action = get_zero_action(env)

        for _ in range(10):
            # Take a single step with zero action (for demonstration)
            env.step(zero_action)

            # Optionally render again
            if args.render:
                env.render()

        # Get environment metadata
        ep_meta = env.get_ep_meta() if hasattr(env, "get_ep_meta") else {}
        ep_meta["episode_index"] = ep_idx
        ep_meta["seed"] = current_seed

        # Save episode metadata as JSON
        meta_filename = os.path.join(ep_meta_dir, f"ep_meta_{ep_idx:03d}.json")
        with open(meta_filename, "w") as f:
            json.dump(ep_meta, f, indent=4)
        print(f"Saved metadata to {meta_filename}")

        # Save a single rendered image from the camera
        # In robosuite, camera images come in the form (height, width, channels) with
        # top-to-bottom alignment. Flip if needed for typical top-left image alignment.
        if not args.render:
            img = env.sim.render(width=1024, height=1024, camera_name=args.camera)
            # Convert from OpenGL-style (top-left at (0,0)) to typical image style if needed
            img = img[::-1]  # vertical flip
            img = img[..., ::-1]  # convert RGB to BGR for OpenCV if needed

            img_filename = os.path.join(img_dir, f"ep_{ep_idx:03d}.png")
            cv2.imwrite(img_filename, img)
            print(f"Saved image to {img_filename}")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
