import h5py
import argparse
import numpy as np
import cv2
import os
from tqdm import trange, tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import threading
from concurrent.futures import ThreadPoolExecutor
import time


class HDF5VideoDataset(Dataset):
    """
    PyTorch Dataset for loading video frames from HDF5 files with efficient preloading.
    """

    def __init__(
        self,
        hdf5_path: str,
        camera_names: List[str],
        chunk_size: int = 100,
        img_width: int = -1,
        img_height: int = -1,
    ):
        """
        Args:
            hdf5_path: Path to the HDF5 file
            camera_names: List of camera names to include
            chunk_size: Number of frames to load at once (for memory management)
        """
        self.hdf5_path = hdf5_path
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.img_width = img_width
        self.img_height = img_height

        # Get dataset info
        with h5py.File(hdf5_path, "r") as f:
            data_all = f["data"]
            data_keys = list(data_all.keys())
            if not data_keys:
                raise ValueError(f"No data keys found in {hdf5_path}")

            # Use first data key
            data = data_all[data_keys[0]]
            self.n_frames = len(data[f"obs/{camera_names[0]}"])
            self.data_key = data_keys[0]

        # Pre-load first chunk to get dimensions
        self._load_chunk(0)
        first_frame = self._get_frame(0)
        self.frame_height, self.frame_width = first_frame.shape[:2]

        print(
            f"Dataset initialized: {self.n_frames} frames, {self.frame_height}x{self.frame_width}"
        )

    def _load_chunk(self, chunk_idx: int):
        """Load a chunk of frames into memory."""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.n_frames)

        with h5py.File(self.hdf5_path, "r") as f:
            data = f["data"][self.data_key]

            # Load all camera data for this chunk
            chunk_data = {}
            for camera_name in self.camera_names:
                chunk_data[camera_name] = data[f"obs/{camera_name}"][start_idx:end_idx]
                if self.img_width != -1:
                    # resize the frame
                    chunk_data[camera_name] = [
                        cv2.resize(frame, (self.img_width, self.img_height))
                        for frame in chunk_data[camera_name]
                    ]

            # Store in thread-local storage
            if not hasattr(self, "_thread_local"):
                self._thread_local = threading.local()
            self._thread_local.chunk_data = chunk_data
            self._thread_local.chunk_start = start_idx

    def _get_frame(self, frame_idx: int) -> np.ndarray:
        """Get a single frame, loading chunk if necessary."""
        chunk_idx = frame_idx // self.chunk_size

        # Load chunk if not already loaded or if it's a different chunk
        if (
            not hasattr(self._thread_local, "chunk_data")
            or self._thread_local.chunk_start != chunk_idx * self.chunk_size
        ):
            self._load_chunk(chunk_idx)

        # Get frame from loaded chunk
        local_idx = frame_idx % self.chunk_size
        frame = np.concatenate(
            [
                self._thread_local.chunk_data[camera_name][local_idx]
                for camera_name in self.camera_names
            ],
            axis=1,
        )
        return frame

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._get_frame(idx)


def generate_video_from_hdf5_with_dataloader(
    hdf5_path: str,
    output_dir: Optional[str] = None,
    camera_names: List[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    chunk_size: int = 100,
    img_width: int = -1,
    img_height: int = -1,
) -> str:
    """
    Generate MP4 video from HDF5 file using PyTorch DataLoader for efficient loading.

    Args:
        hdf5_path: Path to the HDF5 file
        output_dir: Output directory. If None, saves in same directory as HDF5
        camera_names: List of camera names to include in video
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for DataLoader
        chunk_size: Number of frames to load at once per worker

    Returns:
        Path to the generated MP4 file
    """
    if camera_names is None:
        camera_names = ["robot0_agentview_center_image"]

    # Determine output path
    hdf5_path = Path(hdf5_path)
    if output_dir is None:
        output_file = hdf5_path.with_suffix(".mp4")
    else:
        parent_name = hdf5_path.parent.name
        parent_parent_name = hdf5_path.parent.parent.name
        output_filename = f"{parent_parent_name}_{parent_name}.mp4"
        output_file = Path(output_dir) / output_filename

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {hdf5_path} -> {output_file}")

    # Create dataset and dataloader
    dataset = HDF5VideoDataset(
        hdf5_path, camera_names, chunk_size, img_width, img_height
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Keep as numpy arrays
        persistent_workers=True if num_workers > 0 else False,
    )

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(output_file), fourcc, 30.0, (dataset.frame_width, dataset.frame_height)
    )

    print(
        f"Processing {len(dataset)} frames with batch_size={batch_size}, workers={num_workers}"
    )

    # Process frames
    frame_count = 0
    pbar = tqdm(len(dataloader), desc="Generating video", total=len(dataloader))
    for batch in dataloader:
        # Handle different batch structures
        if isinstance(batch, (list, tuple)):
            # If batch is a list/tuple of frames
            frames = batch
        elif isinstance(batch, torch.Tensor):
            # If batch is a tensor, convert to numpy
            frames = batch.numpy()
        else:
            # If batch is a single frame
            frames = [batch]

        # Write each frame
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy()
            # convert it to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
            frame_count += 1
        pbar.update(1)

    pbar.close()
    video_writer.release()
    print(f"✅ Video generated: {output_file} ({frame_count} frames)")
    return str(output_file)


def generate_video_from_hdf5(
    hdf5_path: str,
    output_dir: Optional[str] = None,
    camera_names: List[str] = None,
    use_dataloader: bool = True,
    img_width: int = -1,
    img_height: int = -1,
    **dataloader_kwargs,
) -> str:
    """
    Generate MP4 video from HDF5 file with option to use DataLoader.

    Args:
        hdf5_path: Path to the HDF5 file
        output_dir: Output directory. If None, saves in same directory as HDF5
        camera_names: List of camera names to include in video
        use_dataloader: Whether to use PyTorch DataLoader (recommended)
        **dataloader_kwargs: Additional arguments for DataLoader version

    Returns:
        Path to the generated MP4 file
    """
    if use_dataloader:
        return generate_video_from_hdf5_with_dataloader(
            hdf5_path,
            output_dir,
            camera_names,
            img_width=img_width,
            img_height=img_height,
            **dataloader_kwargs,
        )
    else:
        # Original implementation (keeping for backward compatibility)
        if camera_names is None:
            camera_names = ["robot0_agentview_center_image"]

        # Determine output path
        hdf5_path = Path(hdf5_path)
        if output_dir is None:
            # Save in same directory as HDF5, replacing .hdf5 with .mp4
            output_file = hdf5_path.with_suffix(".mp4")
        else:
            # Save in specified directory with name: parent_parent_name_parent_name.mp4
            parent_name = hdf5_path.parent.name
            parent_parent_name = hdf5_path.parent.parent.name
            output_filename = f"{parent_parent_name}_{parent_name}.mp4"
            output_file = Path(output_dir) / output_filename

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing {hdf5_path} -> {output_file}")

        with h5py.File(hdf5_path, "r") as output_hdf5:
            data_all = output_hdf5["data"]
            data_keys = list(data_all.keys())

            if not data_keys:
                print(f"Warning: No data keys found in {hdf5_path}")
                return str(output_file)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            print(f"First data keys: {data_keys}")
            # Get dimensions from first frame to set video dimensions
            first_data = data_all[data_keys[0]]
            first_frame = np.concatenate(
                [
                    first_data["obs/{}".format(camera_name)][0]
                    for camera_name in camera_names
                ],
                axis=1,
            )
            height, width = (
                first_frame.shape[:2]
                if img_width == -1
                else (img_height, img_width * len(camera_names))
            )
            print(f"Dimensions: {height}x{width}")

            video_writer = cv2.VideoWriter(
                str(output_file), fourcc, 30.0, (width, height)
            )

            # Process each data key
            for data_key in data_keys:
                data = data_all[data_key]

                print("Pre-loading all camera data into memory")
                # Pre-load all camera data at once to avoid repeated HDF5 reads
                camera_data = {}
                for camera_name in camera_names:
                    camera_data[camera_name] = data[f"obs/{camera_name}"][
                        :
                    ]  # Load entire dataset

                n_frames = len(camera_data[camera_names[0]])
                print(f"Processing {n_frames} frames...")

                for i in trange(n_frames):
                    # Concatenate camera images along width (axis=1)
                    reisze_func = (
                        lambda x: cv2.resize(x, (img_width, img_height))
                        if img_width != -1
                        else x
                    )
                    print(f"Resizing frame {i} to {img_width}x{img_height}")
                    frame = np.concatenate(
                        [
                            reisze_func(camera_data[camera_name][i])
                            for camera_name in camera_names
                        ],
                        axis=1,
                    )
                    video_writer.write(frame)
                break

            video_writer.release()
            print(f"✅ Video generated: {output_file}")
            return str(output_file)


def process_multiple_hdf5_files(
    hdf5_paths: List[str],
    output_dir: Optional[str] = None,
    camera_names: List[str] = None,
) -> List[str]:
    """
    Process multiple HDF5 files and generate MP4 videos.

    Args:
        hdf5_paths: List of paths to HDF5 files
        output_dir: Output directory. If None, saves in same directory as each HDF5
        camera_names: List of camera names to include in videos

    Returns:
        List of paths to generated MP4 files
    """
    generated_videos = []

    for hdf5_path in hdf5_paths:
        if os.path.exists(hdf5_path):
            output_file = generate_video_from_hdf5(hdf5_path, output_dir, camera_names)
            if output_file:
                generated_videos.append(output_file)
        else:
            print(f"❌ File not found: {hdf5_path}")

    return generated_videos


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width", type=int, required=False, default=256)
    parser.add_argument("--img_height", type=int, required=False, default=256)
    parser.add_argument("--output_dir", type=str, required=False, default=None)
    parser.add_argument(
        "--base_dir",
        type=str,
        required=False,
        default=None,
        help="Base directory for dataset (overrides CASAPLAY_DATAROOT env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Use provided base_dir or fall back to environment variable
    base_dir = args.base_dir or os.environ.get("CASAPLAY_DATAROOT", None)
    if base_dir is None:
        print(
            "ERROR: Base directory not specified. Use --base_dir or set CASAPLAY_DATAROOT environment variable."
        )
        sys.exit(1)

    # Example usage with DataLoader
    hdf5_paths = [
        # f"{base_dir}/MimicDroidDataset/TaskDemos/PnPSinkToMicrowaveTopL3/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/CloseLeftCabinetDoor/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/CloseLeftCabinetDoorL2/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/CloseLeftCabinetDoorL3/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/CloseRightCabinetDoorL2/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/PnPSinkToCabinet/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/PnPSinkToCabinetL2/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/PnPSinkToMicrowaveTopL3/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/PnPSinkToRightCounterPlate/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/PnPSinkToRightCounterPlateL2/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/PnPSinkToRightCounterPlateL3/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/TurnOnFaucet/003/demo_im128_notp.hdf5",
        # f"{base_dir}/MimicDroidDataset/TaskDemos/TurnOnFaucetL3/003/demo_im128_notp.hdf5",
    ]

    output_directory = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(base_dir, "final_prompt_videos")
    )
    os.makedirs(output_directory, exist_ok=True)

    # Use DataLoader with optimized settings
    generated_videos = []
    for hdf5_path in hdf5_paths:
        if os.path.exists(hdf5_path):
            output_file = generate_video_from_hdf5(
                hdf5_path,
                output_directory,
                use_dataloader=True,
                batch_size=16,  # Adjust based on your GPU/memory
                num_workers=16,  # Adjust based on your CPU cores
                chunk_size=50,  # Adjust based on your memory
                img_width=args.img_width,
                img_height=args.img_height,
            )
            if output_file:
                generated_videos.append(output_file)
        else:
            print(f"❌ File not found: {hdf5_path}")

    print(f"\n🎬 Generated {len(generated_videos)} videos:")
    for video in generated_videos:
        print(f"  - {video}")
