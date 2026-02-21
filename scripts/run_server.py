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
