#!/usr/bin/env python3
"""Cross-platform build orchestrator for FutureSightML.

Steps:
1. Generate Master Ball icons (icon.ico + icon.png)
2. Export meta pools from DB (data/pools/*.json) — gen-validated
3. Bundle Python backend with PyInstaller
4. Build Electron app with electron-builder

Outputs per platform:
  Windows: gui/dist/win-unpacked/FutureSightML.exe (portable)
  macOS:   gui/dist/mac-universal/FutureSightML.app (universal binary)
  Linux:   gui/dist/FutureSightML-*.AppImage

Usage:
    python scripts/build_exe.py              # Full build for current platform
    python scripts/build_exe.py --skip-pools # Skip pool export (use existing)
    python scripts/build_exe.py --skip-pyinstaller # Skip PyInstaller step
    python scripts/build_exe.py --skip-electron    # Skip electron-builder step
    python scripts/build_exe.py --ci               # CI mode (skip icon gen, assume icons exist)
    python scripts/build_exe.py --electron-arch x64  # Force electron-builder arch (macOS)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Platform detection
PLATFORM = sys.platform  # 'win32', 'darwin', 'linux'
IS_WINDOWS = PLATFORM == 'win32'
IS_MACOS = PLATFORM == 'darwin'
IS_LINUX = PLATFORM.startswith('linux')

PLATFORM_LABEL = 'Windows' if IS_WINDOWS else 'macOS' if IS_MACOS else 'Linux'


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
    """Step 1: Generate Master Ball icons (ICO + PNG)."""
    print("\n[1/4] Generating Master Ball icons...")
    ico_path = PROJECT_ROOT / "gui" / "static" / "icon.ico"
    png_path = PROJECT_ROOT / "gui" / "static" / "icon.png"
    if ico_path.exists() and png_path.exists():
        print(f"  Icons already exist: {ico_path.name} ({ico_path.stat().st_size} bytes), "
              f"{png_path.name} ({png_path.stat().st_size} bytes)")
        return
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "Pillow"])
    run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_icon.py")])
    for p in [ico_path, png_path]:
        if p.exists():
            print(f"  Created: {p.name} ({p.stat().st_size} bytes)")
        else:
            print(f"  WARNING: {p.name} was not created")


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
    print(f"\n[3/4] Bundling Python backend with PyInstaller ({PLATFORM_LABEL})...")

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

    # Check for output — platform-specific exe name
    exe_name = "run_server.exe" if IS_WINDOWS else "run_server"
    exe_path = PROJECT_ROOT / "dist" / "run_server" / exe_name
    if exe_path.exists():
        dist_size = sum(
            f.stat().st_size for f in (PROJECT_ROOT / "dist" / "run_server").rglob("*") if f.is_file()
        ) / (1024 * 1024)
        print(f"  PyInstaller output: {exe_path} (dist: {dist_size:.0f} MB)")
    else:
        print(f"ERROR: {exe_name} was not created!")
        sys.exit(1)


def step_electron(electron_arch: str | None = None):
    """Step 4: Build Electron app for current platform.

    Args:
        electron_arch: Override architecture for electron-builder on macOS
                       ('x64' or 'arm64'). None = use default (universal).
    """
    arch_label = f" [{electron_arch}]" if electron_arch else ""
    print(f"\n[4/4] Building Electron app ({PLATFORM_LABEL}{arch_label})...")

    gui_dir = PROJECT_ROOT / "gui"

    # Check npm is available
    npm_cmd = "npm.cmd" if IS_WINDOWS else "npm"
    try:
        subprocess.run([npm_cmd, "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: npm not found! Install Node.js first.")
        sys.exit(1)

    # Install electron dependencies
    run([npm_cmd, "install"], cwd=gui_dir)

    # Platform-specific build command
    if IS_MACOS:
        if electron_arch in ("x64", "arm64"):
            build_script = f"build-mac-{electron_arch}"
        else:
            build_script = "build-mac"
    elif IS_LINUX:
        build_script = "build-linux"
    else:
        build_script = "build-win"

    run(
        [npm_cmd, "run", build_script],
        cwd=gui_dir,
        env={"CSC_IDENTITY_AUTO_DISCOVERY": "false"},
    )

    # Check for platform-specific output
    if IS_WINDOWS:
        output_dir = gui_dir / "dist" / "win-unpacked"
        app_path = output_dir / "FutureSightML.exe"
    elif IS_MACOS:
        # electron-builder output dir depends on arch
        if electron_arch:
            output_dir = gui_dir / "dist" / f"mac-{electron_arch}"
        else:
            output_dir = gui_dir / "dist" / "mac-universal"
        if not output_dir.exists():
            output_dir = gui_dir / "dist" / "mac"
        app_path = output_dir / "FutureSightML.app"
    else:
        output_dir = gui_dir / "dist"
        # AppImage files match FutureSightML-*.AppImage
        appimages = list(output_dir.glob("FutureSightML-*.AppImage"))
        app_path = appimages[0] if appimages else output_dir / "FutureSightML.AppImage"

    if app_path.exists():
        if app_path.is_dir():
            total_size = sum(
                f.stat().st_size for f in app_path.rglob("*") if f.is_file()
            ) / (1024 * 1024 * 1024)
        else:
            total_size = app_path.stat().st_size / (1024 * 1024 * 1024)
        print(f"  Electron app: {app_path}")
        print(f"  Total size: {total_size:.1f} GB")
        print(f"\n  Launch with: {app_path}")
    else:
        print(f"WARNING: Expected output not found at {app_path}")
        print(f"  Check gui/dist/ for build output")


def main():
    parser = argparse.ArgumentParser(description=f"Build FutureSightML app ({PLATFORM_LABEL})")
    parser.add_argument("--skip-pools", action="store_true", help="Skip pool export step")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="Skip PyInstaller step")
    parser.add_argument("--skip-electron", action="store_true", help="Skip electron-builder step")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: skip icon generation (assume icons exist)")
    parser.add_argument("--electron-arch", choices=["x64", "arm64"],
                        help="Override electron-builder arch on macOS (x64 or arm64)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FutureSightML Build Orchestrator ({PLATFORM_LABEL})")
    if args.ci:
        print("  Mode: CI")
    print("=" * 60)

    # Step 1: Generate icons (skip in CI mode — icons should already exist)
    if args.ci:
        print("\n[1/4] Skipping icon generation (--ci)")
        ico_path = PROJECT_ROOT / "gui" / "static" / "icon.ico"
        png_path = PROJECT_ROOT / "gui" / "static" / "icon.png"
        if not ico_path.exists() and not png_path.exists():
            print("  WARNING: No icons found. Electron app may lack an icon.")
    else:
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
        step_electron(electron_arch=args.electron_arch)
    else:
        print("\n[4/4] Skipping electron-builder (--skip-electron)")

    # Final output summary
    print("\n" + "=" * 60)
    print(f"  Build complete! ({PLATFORM_LABEL})")
    if IS_WINDOWS:
        print("  Output: gui/dist/win-unpacked/FutureSightML.exe")
    elif IS_MACOS:
        arch = args.electron_arch or "universal"
        print(f"  Output: gui/dist/mac-{arch}/FutureSightML.app")
    else:
        print("  Output: gui/dist/FutureSightML-*.AppImage")
    print("=" * 60)


if __name__ == "__main__":
    main()
