#!/usr/bin/env python3
"""Launch the FutureSightML API server.

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --port 8080
    python scripts/run_server.py --host 0.0.0.0 --port 5000
"""

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run FutureSightML server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    # Check for pre-trained models — auto-download if missing
    project_root = Path(__file__).resolve().parent.parent
    # Support PyInstaller frozen exe — data lives in _MEIPASS/_internal
    if getattr(sys, '_MEIPASS', None):
        project_root = Path(sys._MEIPASS)
    checkpoints_dir = project_root / "data" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = list(checkpoints_dir.glob("neural_*_best.pt"))
    if not checkpoints:
        print("\n  No pre-trained models found. Downloading (~310 MB)...")
        try:
            def get_download_url():
                return "https://huggingface.co/HotHams/FutureSightML/resolve/main/model-data.tar.gz"
            import io, tarfile, urllib.request
            url = get_download_url()
            print(f"  Fetching from {url}")
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunks = []
                while True:
                    chunk = resp.read(1024 * 1024)
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
            print("  Extracting...")
            buf = io.BytesIO(data)
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                tar.extractall(path=project_root)
            n = len(list(checkpoints_dir.glob("neural_*_best.pt")))
            print(f"  Done! {n} models downloaded.\n")
        except Exception as e:
            print(f"\n  Auto-download failed: {e}")
            print("  Run manually: python scripts/download_models.py")
            print()
            sys.exit(1)

    if not args.no_browser:
        import threading
        def open_browser():
            import time
            time.sleep(2)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n  FutureSightML starting at http://{args.host}:{args.port}")
    print(f"  API docs at http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        "showdown.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
