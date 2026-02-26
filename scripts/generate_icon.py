#!/usr/bin/env python3
"""Generate an 8-bit pixel-art Master Ball icon for FutureSightML.

Creates a 16x16 base pixel art, then nearest-neighbor upscales to
multi-size ICO (16, 32, 48, 64, 256) at gui/static/icon.ico.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


# 16x16 Master Ball pixel art
# Color palette
T = (0, 0, 0, 0)        # Transparent
K = (20, 20, 30, 255)    # Black outline
P = (120, 40, 160, 255)  # Purple (top shell)
D = (80, 25, 110, 255)   # Dark purple (shading)
L = (160, 70, 200, 255)  # Light purple (highlight)
W = (240, 240, 245, 255) # White (bottom shell)
G = (200, 200, 210, 255) # Light gray (bottom shading)
B = (50, 50, 60, 255)    # Band (dark gray)
M = (200, 50, 80, 255)   # Magenta/pink (M detail)
C = (220, 220, 230, 255) # Center button highlight
Y = (255, 255, 255, 255) # Pure white (button shine)

# fmt: off
PIXELS = [
    # Row 0  (top edge)
    [T, T, T, T, T, K, K, K, K, K, K, T, T, T, T, T],
    # Row 1
    [T, T, T, K, K, P, P, P, P, P, P, K, K, T, T, T],
    # Row 2
    [T, T, K, D, P, P, L, L, P, P, P, P, D, K, T, T],
    # Row 3
    [T, K, D, P, P, L, L, P, P, P, P, P, P, D, K, T],
    # Row 4 (M starts)
    [T, K, P, P, M, P, P, M, P, P, M, P, P, P, K, T],
    # Row 5
    [K, P, P, M, M, M, M, M, M, M, M, M, P, P, P, K],
    # Row 6
    [K, P, P, M, P, M, P, P, M, P, M, P, P, P, P, K],
    # Row 7  (band)
    [K, B, B, B, B, B, K, C, C, K, B, B, B, B, B, K],
    # Row 8  (band with button)
    [K, B, B, B, B, K, C, Y, Y, C, K, B, B, B, B, K],
    # Row 9  (band bottom)
    [K, B, B, B, B, B, K, C, C, K, B, B, B, B, B, K],
    # Row 10
    [K, W, W, W, W, W, W, W, W, W, W, W, W, W, W, K],
    # Row 11
    [K, W, W, W, W, W, W, W, W, W, W, W, W, G, W, K],
    # Row 12
    [T, K, W, W, W, W, W, W, W, W, W, W, G, G, K, T],
    # Row 13
    [T, K, G, W, W, W, W, W, W, W, W, G, G, G, K, T],
    # Row 14
    [T, T, K, G, G, W, W, W, W, G, G, G, G, K, T, T],
    # Row 15 (bottom edge)
    [T, T, T, K, K, K, K, K, K, K, K, K, K, T, T, T],
]
# fmt: on


def _build_base_image() -> Image.Image:
    """Build the 16x16 base pixel art image."""
    base = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for y, row in enumerate(PIXELS):
        for x, color in enumerate(row):
            base.putpixel((x, y), color)
    return base


def create_icon(output_path: Path) -> None:
    """Create multi-size ICO from 16x16 pixel art."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = _build_base_image()

    # Generate sizes via nearest-neighbor (keeps chunky pixels)
    sizes = [16, 32, 48, 64, 256]
    frames = []
    for sz in sizes:
        resized = base.resize((sz, sz), Image.NEAREST)
        frames.append(resized)

    # Save as ICO — Pillow expects the sizes parameter to match the images
    # The first image is the base, append_images are additional sizes
    frames[-1].save(
        str(output_path),
        format="ICO",
        append_images=frames[:-1],
        sizes=[(sz, sz) for sz in sizes],
    )
    print(f"Icon saved: {output_path} ({', '.join(f'{s}x{s}' for s in sizes)})")


def create_png(output_path: Path, size: int = 512) -> None:
    """Create a high-res PNG icon (for macOS/Linux and electron-builder)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = _build_base_image()
    large = base.resize((size, size), Image.NEAREST)
    large.save(str(output_path), format="PNG")
    print(f"PNG icon saved: {output_path} ({size}x{size})")


def main():
    project_root = Path(__file__).resolve().parent.parent
    static_dir = project_root / "gui" / "static"

    # Always generate both formats
    create_icon(static_dir / "icon.ico")
    create_png(static_dir / "icon.png", size=512)


if __name__ == "__main__":
    main()
