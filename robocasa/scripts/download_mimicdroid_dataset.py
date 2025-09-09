#!/usr/bin/env python3
"""
MimicDroid dataset downloader (Hugging Face only)

- Single implementation path using Hugging Face CLI
- Supports downloading: all | task_demos | play_data
- No interactive prompts; use --overwrite to remove existing target
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import robocasa  # to keep the same default target convention as before

HF_REPO = "Rutav/MimicDroidDataset"
DATASET_SUBSETS = {
    "task_demos": ["TaskDemos/**"],
    "play_data": ["training/**"],
    "all": [],  # no include pattern -> full repo
}


def _find_hf_cli() -> str:
    """
    Prefer 'hf'; fall back to 'huggingface-cli' if needed.
    """
    for cmd in ("hf", "huggingface-cli"):
        try:
            subprocess.run(
                [cmd, "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return cmd
        except Exception:
            continue
    print("ERROR: Hugging Face CLI not found. Install via:")
    print("  pip install -U 'huggingface_hub[cli]'")
    sys.exit(1)


def _check_hf_transfer():
    """
    Check if hf_transfer is available for fast downloads.
    """
    try:
        import hf_transfer

        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download MimicDroid datasets from Hugging Face"
    )
    parser.add_argument(
        "--component",
        choices=list(DATASET_SUBSETS.keys()),
        default="all",
        help="Which part to download: all | task_demos | play_data (default: all)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Local directory to place the dataset (default: ./MimicDroidDataset)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove target directory before downloading",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress CLI output (passes --quiet to huggingface-cli)",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Resume interrupted downloads",
    )
    parser.add_argument(
        "--disable-fast-transfer",
        action="store_true",
        help="Disable fast transfer (hf_transfer) even if available",
    )
    args = parser.parse_args()

    # Set default target directory if not provided
    if args.target_dir is None:
        target_dir = Path.cwd() / "MimicDroidDataset"
    else:
        target_dir = Path(args.target_dir).expanduser().resolve()

    if args.overwrite and target_dir.exists():
        print(f"[info] Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    hf_cmd = _find_hf_cli()

    # Base CLI args
    cli = [
        hf_cmd,
        "download",
        HF_REPO,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(target_dir),
    ]

    # Only add symlinks option for older huggingface-cli versions
    if hf_cmd == "huggingface-cli":
        cli.extend(["--local-dir-use-symlinks", "False"])

    if args.quiet:
        cli.append("--quiet")
    if args.resume_download:
        cli.append("--resume-download")

    include_patterns = DATASET_SUBSETS[args.component]
    if include_patterns:
        cli.extend(include_patterns)

    print(f"[info] Downloading {args.component} from {HF_REPO}")
    print(f"[info] Target directory: {target_dir}")
    print(f"[info] Using CLI: {hf_cmd}")

    # Check for fast transfer availability
    has_hf_transfer = _check_hf_transfer()
    if has_hf_transfer and not args.disable_fast_transfer:
        print("[info] Fast transfer (hf_transfer) is available and enabled")
        env = os.environ.copy()
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    else:
        if not has_hf_transfer:
            print("[info] Fast transfer not available, using standard download")
            print("[info] To enable fast downloads, install: pip install hf_transfer")
        else:
            print("[info] Fast transfer disabled by user")
        env = os.environ.copy()
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    print("[info] Executing:", " ".join(cli))

    # Run download
    try:
        subprocess.run(cli, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[error] Download failed with exit code {e.returncode}")
        print("[error] Make sure you have access to the dataset and are logged in:")
        print("  huggingface-cli login")
        print("\n[error] If you continue to have issues, try:")
        print("  1. Install hf_transfer for faster downloads: pip install hf_transfer")
        print("  2. Use the --disable-fast-transfer flag to disable fast transfer")
        print("  3. Check your internet connection")
        sys.exit(e.returncode)

    print(f"[success] Download complete → {target_dir}")
    print(f"[info] Dataset structure:")
    print(f"  - Task demos: {target_dir}/TaskDemos/")
    print(f"  - Training data: {target_dir}/training/")


if __name__ == "__main__":
    main()
