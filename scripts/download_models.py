#!/usr/bin/env python3
"""Download pre-trained models and pool data from HuggingFace.

Usage:
    python scripts/download_models.py
"""

import io
import sys
import tarfile
import urllib.request
from pathlib import Path

HF_REPO = "HotHams/FutureSightML"
ASSET_NAME = "model-data.tar.gz"


def get_download_url() -> str:
    """Get the download URL for model-data.tar.gz.

    Primary: HuggingFace model repo (no size limits, fast CDN).
    Fallback: GitHub release.
    """
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{ASSET_NAME}"


def main():
    project_root = Path(__file__).resolve().parent.parent
    checkpoints_dir = project_root / "data" / "checkpoints"
    pools_dir = project_root / "data" / "pools"

    # Check if models already exist
    existing = list(checkpoints_dir.glob("neural_*_best.pt"))
    if existing:
        print(f"Found {len(existing)} existing models in data/checkpoints/")
        resp = input("Re-download and overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Skipped.")
            return

    url = get_download_url()
    print(f"Downloading models from {url} ...")
    print("(This is ~310 MB, may take a few minutes)")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunks = []
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    mb = downloaded // (1024 * 1024)
                    total_mb = total // (1024 * 1024)
                    print(f"\r  {mb}/{total_mb} MB ({pct}%)", end="", flush=True)
            print()
            data = b"".join(chunks)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"You can manually download from: https://huggingface.co/{HF_REPO}")
        sys.exit(1)

    print("Extracting...")
    buf = io.BytesIO(data)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(path=project_root)

    n_neural = len(list(checkpoints_dir.glob("neural_*_best.pt")))
    n_pools = len(list(pools_dir.glob("*.json")))
    print(f"Done! {n_neural} models, {n_pools} pool files extracted.")
    print("Run 'python scripts/run_server.py' to start.")


if __name__ == "__main__":
    main()
