#!/usr/bin/env python3
"""Build orchestrator for FutureSightML Windows installer.

Steps:
1. Generate Master Ball icon (gui/static/icon.ico)
2. Export meta pools from DB (data/pools/*.json) — gen-validated
3. Bundle Python backend with PyInstaller
4. Build Electron app with electron-builder

Output: gui/dist/win-unpacked/FutureSightML.exe (portable)

Usage:
    python scripts/build_exe.py              # Full build
    python scripts/build_exe.py --skip-pools # Skip pool export (use existing)
    python scripts/build_exe.py --skip-pyinstaller # Skip PyInstaller step
    python scripts/build_exe.py --skip-electron    # Skip electron-builder step
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, env=None) -> int:
    """Run a command, streaming output."""
    print(f"\n{'=' * 60}")
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    if cwd:
        print(f"  CWD: {cwd}")
    print(f"{'=' * 60}\n")
    import os
    run_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=run_env)
    if check and result.returncode != 0:
        print(f"\nERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def step_icon():
    """Step 1: Generate Master Ball icon."""
    print("\n[1/4] Generating Master Ball icon...")
    ico_path = PROJECT_ROOT / "gui" / "static" / "icon.ico"
    if ico_path.exists():
        print(f"  Icon already exists: {ico_path} ({ico_path.stat().st_size} bytes)")
        return
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "Pillow"])
    run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_icon.py")])
    if not ico_path.exists():
        print("ERROR: icon.ico was not created!")
        sys.exit(1)
    print(f"  Icon created: {ico_path} ({ico_path.stat().st_size} bytes)")


def step_export_pools():
    """Step 2: Export meta pools from database (gen-validated)."""
    print("\n[2/4] Exporting meta pools from database...")
    run([sys.executable, str(PROJECT_ROOT / "scripts" / "export_pools.py")])
    pools_dir = PROJECT_ROOT / "data" / "pools"
    if pools_dir.exists():
        pool_files = list(pools_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in pool_files) / (1024 * 1024)
        print(f"  Exported {len(pool_files)} pool files ({total_size:.1f} MB)")
    else:
        print("WARNING: No pool files were exported")


def step_pyinstaller():
    """Step 3: Bundle Python backend with PyInstaller."""
    print("\n[3/4] Bundling Python backend with PyInstaller...")

    # Ensure pyinstaller is installed
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    spec_path = PROJECT_ROOT / "FutureSightML.spec"
    if not spec_path.exists():
        print(f"ERROR: {spec_path} not found!")
        sys.exit(1)

    # Clean previous build (handle locked dirs gracefully)
    dist_path = PROJECT_ROOT / "dist" / "run_server"
    if dist_path.exists():
        try:
            print(f"  Cleaning {dist_path}...")
            shutil.rmtree(dist_path)
        except PermissionError:
            # Try renaming instead
            alt = PROJECT_ROOT / "dist" / "run_server_old"
            print(f"  Cannot remove (locked), renaming to {alt.name}...")
            if alt.exists():
                shutil.rmtree(alt, ignore_errors=True)
            dist_path.rename(alt)

    run([sys.executable, "-m", "PyInstaller", str(spec_path), "--noconfirm"])

    exe_path = PROJECT_ROOT / "dist" / "run_server" / "run_server.exe"
    if exe_path.exists():
        dist_size = sum(
            f.stat().st_size for f in (PROJECT_ROOT / "dist" / "run_server").rglob("*") if f.is_file()
        ) / (1024 * 1024)
        print(f"  PyInstaller output: {exe_path} (dist: {dist_size:.0f} MB)")
    else:
        print("ERROR: run_server.exe was not created!")
        sys.exit(1)


def step_electron():
    """Step 4: Build Electron app (portable directory)."""
    print("\n[4/4] Building Electron app...")

    gui_dir = PROJECT_ROOT / "gui"

    # Check npm is available
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    try:
        subprocess.run([npm_cmd, "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: npm not found! Install Node.js first.")
        sys.exit(1)

    # Install electron dependencies
    run([npm_cmd, "install"], cwd=gui_dir)

    # Build Windows app (dir target = no code signing needed)
    run(
        [npm_cmd, "run", "build-win"],
        cwd=gui_dir,
        env={"CSC_IDENTITY_AUTO_DISCOVERY": "false"},
    )

    # Check for output
    unpacked = gui_dir / "dist" / "win-unpacked"
    if unpacked.exists():
        exe = unpacked / "FutureSightML.exe"
        if exe.exists():
            total_size = sum(
                f.stat().st_size for f in unpacked.rglob("*") if f.is_file()
            ) / (1024 * 1024 * 1024)
            print(f"  Electron app: {unpacked}")
            print(f"  Total size: {total_size:.1f} GB")
            print(f"\n  Launch with: {exe}")
    else:
        print("WARNING: win-unpacked directory not found in gui/dist/")


def main():
    parser = argparse.ArgumentParser(description="Build FutureSightML Windows app")
    parser.add_argument("--skip-pools", action="store_true", help="Skip pool export step")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="Skip PyInstaller step")
    parser.add_argument("--skip-electron", action="store_true", help="Skip electron-builder step")
    args = parser.parse_args()

    print("=" * 60)
    print("  FutureSightML Build Orchestrator")
    print("=" * 60)

    # Step 1: Always generate icon
    step_icon()

    # Step 2: Export pools
    if not args.skip_pools:
        step_export_pools()
    else:
        print("\n[2/4] Skipping pool export (--skip-pools)")

    # Step 3: PyInstaller
    if not args.skip_pyinstaller:
        step_pyinstaller()
    else:
        print("\n[3/4] Skipping PyInstaller (--skip-pyinstaller)")

    # Step 4: Electron
    if not args.skip_electron:
        step_electron()
    else:
        print("\n[4/4] Skipping electron-builder (--skip-electron)")

    print("\n" + "=" * 60)
    print("  Build complete!")
    print("  Output: gui/dist/win-unpacked/FutureSightML.exe")
    print("=" * 60)


if __name__ == "__main__":
    main()
