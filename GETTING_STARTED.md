# Getting Started with MimicDroid Tasks

This document provides instructions for using the `replay_dataset.py` script to replay and visualize robotic manipulation trajectories from HDF5 dataset files.

## Quick Start

### Example Usage

Using the abstract (free-floating) embodiment
```bash
python robocasa/scripts/replay_dataset.py \
    --hdf5_path MimicDroidDataset/TaskDemos/CloseLeftCabinetDoor/003/demo_im128_notp.hdf5 \
    --robots DemoTwoHand \
    --render \
```
Using the GR1 (humanoid) embodiment
```bash
python robocasa/scripts/replay_dataset.py \
    --hdf5_path MimicDroidDataset/TaskDemos/TurnOnFaucet/003/demo_im128_notp.hdf5 \
    --robots GR1TwoHand \
    --render \
    --verbose
```

## Environment Initialization

The script automatically creates a RoboCasa environment from the HDF5 file using the `make_env` function:

```python
from robocasa.utils.mimicdroid_utils import make_env, EnvArgs

# Create environment arguments
env_args = EnvArgs(
    robots="DemoTwoHand",
    task_name=None,
    render=True,
    control_freq=20,
    controller=None,  # or "WHOLE_BODY_MINK_IK" for GR1
    use_camera_obs=False,
)

# Create environment from HDF5 file
env, env_kwargs = make_env(
    file_name="path/to/dataset.hdf5",
    env_args=env_args
)
```