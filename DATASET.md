# MimicDroid Dataset Download Instructions

This document provides instructions for downloading the MimicDroid datasets used in the RoboCasa environment.

## Dataset Overview

The MimicDroid dataset consists of two main components:

1. **Task Demos Dataset**: Contains demonstration data for few-shot learning (~250MB)
2. **Play Data**: Contains the full dataset with all task demonstrations (~8GB)

## Download Options

### Option 1: Using Hugging Face CLI (Recommended)

Use the Hugging Face CLI to download the datasets:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the complete dataset
huggingface-cli download Rutav/MimicDroidDataset --repo-type dataset --local-dir /home/rutavms/research/gaze/MimicDroidDataset
```

This will:
- Download the complete MimicDroid dataset from Hugging Face
- Store it in the specified local directory
- Handle authentication and progress indicators automatically
- Allow you to resume interrupted downloads

### Option 2: Using Python Script

Use the provided Python script to download the datasets:

```bash
python robocasa/scripts/download_mimicdroid_dataset.py
```

This script will:
- Download the complete MimicDroid dataset
- Handle extraction and organization automatically
- Provide progress indicators during download
- Allow you to choose which components to download

### Option 3: Manual Download via Hugging Face Web Interface

If you prefer to download manually, you can access the dataset directly:

#### MimicDroid Dataset
- **Repository**: https://huggingface.co/datasets/Rutav/MimicDroidDataset
- **Description**: Contains the complete MimicDroid dataset with all task demonstrations
- **Size**: ~8GB total

To download via web interface:
1. Visit the repository URL above
2. Click "Files and versions" tab
3. Download individual files or use the "Download repository" button

## File Structure

After downloading, the datasets will be organized as follows:

### PlayDataset
The main dataset containing comprehensive robotic manipulation demonstrations:

```
MimicDroidDataset/
└── training/                               # Training data and metadata
    ├── 003/                                # Training episode directories
    │   ├── demo_im128_notp.hdf5
    │   └── demo.hdf5
    ├── 004/
    ├── 005/
    ├── 006/
    ├── 007/
    ├── 008/
    ├── 009/
    ├── 010/
    ├── 011/
    ├── 012/
    ├── 013/
    ├── 015/
    ├── 017/
    ├── 018/
    ├── 019/
    ├── 020/
  TaskDemos/
  ├── CloseLeftCabinetDoor/
  │   └── 003/
  │       ├── demo_im128_notp.hdf5
  │       └── demo.hdf5
  ├── CloseLeftCabinetDoorL2/
  ├── CloseLeftCabinetDoorL3/
  ├── CloseRightCabinetDoorL2/
  ├── PnPSinkToCabinet/
  ├── PnPSinkToCabinetL2/
  ├── PnPSinkToMicrowaveTopL3/
  ├── PnPSinkToRightCounterPlate/
  ├── PnPSinkToRightCounterPlateL2/
  ├── PnPSinkToRightCounterPlateL3/
  ├── TurnOnFaucet/
  └── TurnOnFaucetL3/
```

## Requirements

- Python 3.7+
- `huggingface_hub` package (`pip install huggingface_hub`)
- Sufficient disk space (at least 10GB recommended)
- Internet connection for download

## Authentication

If the dataset requires authentication:
1. Create a Hugging Face account at https://huggingface.co
2. Generate an access token in your account settings
3. Login using: `huggingface-cli login`
4. Enter your access token when prompted

## Troubleshooting

If you encounter issues during download:

1. Ensure you have sufficient disk space
2. Check your internet connection
3. Verify you have the `huggingface_hub` package installed
4. Try logging in again with `huggingface-cli login`
5. Use `--resume-download` flag to resume interrupted downloads
6. Check the dataset repository for any access restrictions

## Support

For issues related to dataset download or usage, please create an issue in the repository.
