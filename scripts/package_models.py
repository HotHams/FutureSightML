#!/usr/bin/env python3
"""Package model checkpoints and pool data for CI/CD.

Creates a tar.gz archive of data/checkpoints/ and data/pools/ for upload
to a pinned GitHub Release (tag: model-data).

Usage:
    # Package for upload
    python scripts/package_models.py

    # Download and extract (CI use)
    python scripts/package_models.py --download

    # Upload to GitHub Release (requires gh CLI)
    python scripts/package_models.py --upload
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_NAME = "model-data.tar.gz"
ARCHIVE_PATH = PROJECT_ROOT / ARCHIVE_NAME

DATA_DIRS = [
    ("data/checkpoints", "Model checkpoints (.pt, .joblib, .json)"),
    ("data/pools", "Pre-exported pool data (.json)"),
]


def package():
    """Create model-data.tar.gz from data/checkpoints/ and data/pools/."""
    print(f"Packaging model data into {ARCHIVE_NAME}...")

    total_files = 0
    total_size = 0

    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        for rel_dir, description in DATA_DIRS:
            dir_path = PROJECT_ROOT / rel_dir
            if not dir_path.exists():
                print(f"  WARNING: {rel_dir}/ does not exist, skipping")
                continue

            files = [f for f in dir_path.rglob("*") if f.is_file()]
            if not files:
                print(f"  WARNING: {rel_dir}/ is empty, skipping")
                continue

            dir_size = sum(f.stat().st_size for f in files)
            print(f"  Adding {rel_dir}/ ({len(files)} files, {dir_size / 1024 / 1024:.1f} MB)")
            total_files += len(files)
            total_size += dir_size

            for file_path in files:
                arcname = str(file_path.relative_to(PROJECT_ROOT))
                tar.add(file_path, arcname=arcname)

    if total_files == 0:
        print("\nERROR: No files found to package!")
        print("  Make sure data/checkpoints/ and data/pools/ contain files.")
        ARCHIVE_PATH.unlink(missing_ok=True)
        sys.exit(1)

    archive_size = ARCHIVE_PATH.stat().st_size / (1024 * 1024)
    print(f"\nCreated {ARCHIVE_NAME}: {archive_size:.1f} MB ({total_files} files)")
    print(f"  Uncompressed: {total_size / 1024 / 1024:.1f} MB")
    print(f"\nTo upload to GitHub Release:")
    print(f"  gh release create model-data {ARCHIVE_NAME} --title 'Model Data' \\")
    print(f"    --notes 'Model checkpoints and pool data for CI builds. Re-upload when models are retrained.'")
    print(f"\nTo update existing release:")
    print(f"  gh release upload model-data {ARCHIVE_NAME} --clobber")


def download():
    """Download and extract model-data.tar.gz from GitHub Release."""
    print(f"Downloading {ARCHIVE_NAME} from model-data release...")

    result = subprocess.run(
        ["gh", "release", "download", "model-data", "-p", ARCHIVE_NAME],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("ERROR: Failed to download. Is the model-data release available?")
        sys.exit(1)

    print(f"Extracting {ARCHIVE_NAME}...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=PROJECT_ROOT)

    ARCHIVE_PATH.unlink()

    for rel_dir, _ in DATA_DIRS:
        dir_path = PROJECT_ROOT / rel_dir
        if dir_path.exists():
            files = list(dir_path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            print(f"  Extracted {rel_dir}/: {file_count} files")
        else:
            print(f"  WARNING: {rel_dir}/ not found after extraction")

    print("Done.")


def upload():
    """Upload model-data.tar.gz to GitHub Release."""
    if not ARCHIVE_PATH.exists():
        print(f"ERROR: {ARCHIVE_NAME} not found. Run without flags first to create it.")
        sys.exit(1)

    archive_size = ARCHIVE_PATH.stat().st_size / (1024 * 1024)
    print(f"Uploading {ARCHIVE_NAME} ({archive_size:.1f} MB)...")

    # Try to create release first (may already exist)
    subprocess.run(
        ["gh", "release", "create", "model-data",
         "--title", "Model Data",
         "--notes", "Model checkpoints and pool data for CI builds."],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )

    # Upload (clobber existing)
    result = subprocess.run(
        ["gh", "release", "upload", "model-data", str(ARCHIVE_PATH), "--clobber"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("ERROR: Upload failed.")
        sys.exit(1)

    print("Upload complete.")


def main():
    parser = argparse.ArgumentParser(description="Package model data for CI/CD")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true",
                       help="Download and extract from GitHub Release")
    group.add_argument("--upload", action="store_true",
                       help="Upload archive to GitHub Release")
    args = parser.parse_args()

    if args.download:
        download()
    elif args.upload:
        package()
        upload()
    else:
        package()


if __name__ == "__main__":
    main()
