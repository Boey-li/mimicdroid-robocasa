# Description: Utility functions for loading and processing data from the MimicDroid dataset
# This module contains functions moved from casa_utils.py to make replay_dataset.py isolated

import os
import json
import h5py
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Union, List
from termcolor import colored

import robosuite
from robosuite.utils.binding_utils import MjSimState
from robosuite.controllers import load_composite_controller_config

import robocasa
from robocasa import PLAY_TASK_NAME_TO_CLASS


@dataclass
class EnvArgs:
    """Environment arguments for creating robosuite environments."""

    robots: Union[str, List[str]] = "DemoTwoHand"
    controller: Optional[str] = None
    task_name: Optional[str] = None
    reset_mode: Optional[str] = None
    render: bool = False
    control_freq: int = 20
    use_camera_obs: bool = False


def get_env_args_from_dataset(dataset_path):
    """Extract environment arguments from dataset."""
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_args = (
        json.loads(f["data"].attrs["env_args"])
        if "data" in f
        else json.loads(f.attrs["env_args"])
    )
    if isinstance(env_args, str):
        env_args = json.loads(env_args)  # double leads to dict type
    f.close()
    return env_args


def get_env_meta_from_dataset(dataset_path, index=0):
    """Extract environment metadata from dataset."""
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    data = f["data"] if "data" in f else f
    keys = list(data.keys())
    env_meta = data[keys[index]].attrs["ep_meta"]
    env_meta = json.loads(env_meta)
    assert isinstance(env_meta, dict), f"Expected dict type but got {type(env_meta)}"
    return env_meta


def load_controller_config(controller, robot, control_type, ref_frame):
    """Load and configure controller settings."""
    controller_config = load_composite_controller_config(
        controller=controller,
        robot=robot,
    )
    if "right" in controller_config["body_parts"]:
        controller_config["body_parts"]["right"]["input_type"] = control_type
        controller_config["body_parts"]["right"]["input_ref_frame"] = ref_frame
    if "left" in controller_config["body_parts"]:
        controller_config["body_parts"]["left"]["input_type"] = control_type
        controller_config["body_parts"]["left"]["input_ref_frame"] = ref_frame
    if "WHOLE_BODY" in controller_config["type"]:
        controller_config["composite_controller_specific_configs"][
            "ik_input_ref_frame"
        ] = ref_frame
        controller_config["composite_controller_specific_configs"][
            "ik_input_type"
        ] = control_type
    return controller_config


def replace_robot_tag(new_model_xml, old_model_xml):
    """
    Replace robot tags in the XML.

    Args:
        new_model_xml: str (XML content of the new model)
        old_model_xml: str (XML content of the old model)

    Returns:
        str: Modified XML content as a string.
    """
    # Parse XML content
    new_root = ET.fromstring(new_model_xml)
    old_root = ET.fromstring(old_model_xml)

    # Function to filter and transfer robot0 elements
    def transfer_robot0_elements(new_parent, old_parent):
        # Remove existing robot0 elements in new_parent
        if new_parent is not None:
            for elem in list(new_parent):
                if "robot0" in elem.attrib.get("name", ""):
                    new_parent.remove(elem)

        if old_parent is not None:
            if new_parent is None:
                # add a new parent element if it doesn't exist to the
                # Add robot0 elements from old_parent to new_parent as a child of <mujoco> tag
                new_parent = ET.SubElement(new_root, old_parent.tag)
            for elem in old_parent:
                if "robot0" in elem.attrib.get("name", ""):
                    new_parent.append(elem)

    # Update <actuator> elements
    new_actuator = new_root.find("actuator")
    old_actuator = old_root.find("actuator")
    transfer_robot0_elements(new_actuator, old_actuator)

    # Update <asset> elements
    new_asset = new_root.find("asset")
    old_asset = old_root.find("asset")
    transfer_robot0_elements(new_asset, old_asset)

    new_contact = new_root.find("contact")
    old_contact = old_root.find("contact")
    transfer_robot0_elements(new_contact, old_contact)

    new_robot0_base = None
    for body in new_root.findall(".//body"):
        if body.attrib.get("name") == "robot0_floating_base":
            new_robot0_base = body
            break

    # Locate "robot0_floating_base" in old model
    old_robot0_floating_base = None
    for body in old_root.findall(".//body"):
        if body.attrib.get("name") == "robot0_base":
            old_robot0_floating_base = body
            break

    assert new_robot0_base is not None, "robot0_floating_base must exist in new model"
    assert old_robot0_floating_base is not None, "robot0_base must exist in old model"
    # Replace the contents of "robot0_base" with "robot0_floating_base" if both exist
    if new_robot0_base is not None and old_robot0_floating_base is not None:
        # Clear the current children of new_robot0_base
        new_robot0_base.clear()

        # Copy all attributes from old_robot0_floating_base to new_robot0_base
        new_robot0_base.attrib = old_robot0_floating_base.attrib

        # Copy all child elements
        for elem in old_robot0_floating_base:
            new_robot0_base.append(elem)

    # copy the geom name from old model to new model named robot0_floor in the worldbody tag
    # we want to put it in the same location in the new model
    new_worldbody = new_root.find("worldbody")
    old_worldbody = old_root.find("worldbody")
    for geom in old_worldbody.findall(".//geom"):
        if geom.attrib.get("name") == "robot0_floor":
            index = list(old_worldbody).index(geom)
            new_worldbody.insert(index, geom)

    # Convert back to string
    return ET.tostring(new_root, encoding="unicode")


def get_zero_action(env):
    """Get zero action for the environment."""
    active_robot = env.robots[0]
    if env.action_dim == 14:
        arms = ["left", "right"]
        zero_action_dict = {}
        for arm in arms:
            # controller has absolute actions, so we need to set the initial action to be the current position
            zero_action = np.zeros(7)
            if active_robot.part_controllers[arm].input_type == "absolute":
                zero_action = robosuite.utils.control_utils.convert_delta_to_abs_action(
                    zero_action, active_robot, arm, env
                )
            zero_action_dict[f"{arm}"] = zero_action[: zero_action.shape[0] - 1]
            zero_action_dict[f"{arm}_gripper"] = zero_action[zero_action.shape[0] - 1 :]
        zero_action_dict["base_mode"] = -1
        zero_action_dict["base"] = np.zeros(3)
        zero_action = active_robot.create_action_vector(zero_action_dict)
    else:
        zero_action = np.zeros(env.action_dim)
        if active_robot.part_controllers[arm].input_type == "absolute":
            zero_action = robosuite.utils.control_utils.convert_delta_to_abs_action(
                zero_action, active_robot, arm, env
            )
    return zero_action


def make_env(file_name, env_args: EnvArgs):
    """Create environment from dataset file."""
    dataset_env_args = get_env_args_from_dataset(dataset_path=file_name)
    env_meta = get_env_meta_from_dataset(dataset_path=file_name, index=0)
    dataset_controller_config = dataset_env_args["env_kwargs"]["controller_configs"]

    if env_args.controller is not None:
        controller_config = load_controller_config(
            controller=env_args.controller,
            robot=env_args.robots
            if isinstance(env_args.robots, str)
            else env_args.robots[0],
            control_type=dataset_controller_config["body_parts"]["right"]["input_type"],
            ref_frame=dataset_controller_config["body_parts"]["right"][
                "input_ref_frame"
            ],
        )
    else:
        print("No controller specified. Using default controller from the dataset")
        controller_config = dataset_controller_config

    env_name = dataset_env_args["env_name"]
    print("Env name: ", env_name)
    if env_args.task_name is not None:
        env_name = PLAY_TASK_NAME_TO_CLASS[env_args.task_name]
    env_kwargs = dataset_env_args["env_kwargs"]

    env_kwargs["eval_mode"] = env_args.reset_mode
    print(f"Reset mode: {env_kwargs['eval_mode']}")
    env_kwargs["env_name"] = env_name
    env_kwargs[
        "ep_meta"
    ] = env_meta  # this should ideally reduce exploration for finding correct set of objects
    env_kwargs["robots"] = (
        env_args.robots if isinstance(env_args.robots, list) else [env_args.robots]
    )
    env_kwargs["controller_configs"] = controller_config
    env_kwargs.pop("has_renderer", None)
    env_kwargs.pop("use_camera_obs", None)
    env_kwargs.pop("renderer", None)
    env_kwargs.pop("camera_segmentations", None)
    print(f"Env args: {dataset_env_args}")

    print("Rendering the environment: ", env_args.render)
    print(f"Control frequency: {env_args.control_freq}")

    env = robosuite.make(
        has_renderer=env_args.render,
        use_camera_obs=env_args.use_camera_obs,
        renderer="mjviewer",
        camera_segmentations="segmentation_level",
        control_freq=env_args.control_freq,
        **env_kwargs,
    )
    return env, env_kwargs


def reset_to(env, state, replace_robot_joints=True, change_to_gr1=False):
    """
    Reset to a specific simulator state.

    Args:
        env: The environment to reset
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml
        replace_robot_joints (bool): whether to replace robot joints
        change_to_gr1 (bool): whether to change to GR1 robot

    Returns:
        None
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = state["ep_meta"]
            if isinstance(ep_meta, str):
                ep_meta = json.loads(ep_meta)
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        else:
            raise ValueError(
                "Environment does not have set_ep_meta or set_attrs_from_ep_meta function"
            )
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        # we need to first update state["model"] to replace the robot tag with the current robot
        curr_xml = env.sim.model.get_xml()

        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        if change_to_gr1:
            xml = replace_robot_tag(new_model_xml=xml, old_model_xml=curr_xml)
        
        # update xml absolute path
        xml = update_xml_absolute_path(xml)

        # update env
        env.reset_from_xml_string(xml)
        # env.sim.reset(): resets the robot back to some position which has collision with the table. Change the xml?
        env.sim.reset()
        env.robots[0].reset()
        env.sim.forward()

    if "states" in state:
        if replace_robot_joints:
            print("Replacing robot joints")
            robot_indices = env.robots[0]._ref_joint_pos_indexes
            other_indices = set(
                range(env.sim.get_state().qpos.flatten().shape[0])
            ) - set(robot_indices)
            other_indices = sorted(list(other_indices))

            non_robot_qpos_idx = state["non_robot_qpos_idx"]
            assert len(non_robot_qpos_idx) == len(
                other_indices
            ), f"Mismatch in non_robot_qpos_idx: {len(non_robot_qpos_idx)} != {len(other_indices)}"
            qpos_state = state["qpos_state"]
            time = env.sim.data.time
            qvel = env.sim.get_state().qvel.flatten().copy()
            qpos = env.sim.get_state().qpos.flatten().copy()
            # copy over everything except the robot state
            for curr_idx, state_idx in zip(other_indices, non_robot_qpos_idx):
                qpos[curr_idx] = qpos_state[state_idx]
            curr_state = MjSimState(qpos=qpos, qvel=qvel, time=time)
            env.sim.set_state_from_flattened(curr_state.flatten())
            env.sim.forward()
            should_ret = True
        else:
            env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True
        if replace_robot_joints:
            env.robots[0].reset()
            env.sim.forward()

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    return None


# Added by Baoyu
def update_xml_absolute_path(
        xml,
        old_path="/home/rutavms/research/gaze/robocasa/robocasa/models",
        new_path="/coc/flash7/bli678/projects/egowm/external/mimicdroid-robocasa/robocasa/models"
    ):
    """
    TODO: change new_path as input argument
    """
    print(colored(f"Update XML from {old_path} to {new_path}", "yellow"))
    return xml.replace(old_path, new_path)