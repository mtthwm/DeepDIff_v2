#!/usr/bin/env python3
"""
make_supplement_spectra_figures_nature.py
Create high-quality spectra plots and labeled RGB overlays (Nature-style)
with normalized intensity and upscale visualization.
"""

import os, csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colors


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
ROOT_FOLDERS = [
    r"Z:\Supplement_Figures\produce",
    r"Z:\Supplement_Figures\banknotes",
    r"Z:\Supplement_Figures\fossils_flora",
    r"Z:\Supplement_Figures\invertebrates",
]

NUM_BANDS = 106
WAVELENGTHS = np.linspace(450, 850, NUM_BANDS)
DPI = 450
UPSCALE = 2.5  # upscale factor for RGB visualization
FONT_SIZE = 36
MARKER_RADIUS = 10

# Nature-style color palette (muted, non-default)
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#F0E442",  # yellow
    "#56B4E9",  # light blue
    "#E69F00",  # gold
    "#000000",  # black
]

# -------------------------------------------------------------------
def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def find_font():
    try:
        return ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        return ImageFont.load_default()

def upscale_image(img, scale):
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def overlay_points_on_rgb(rgb_path, csv_path, out_path):
    """Draw labeled filled dots (P1–Pn) on upscaled RGB image."""
    img = Image.open(rgb_path).convert("RGB")
    img = upscale_image(img, UPSCALE)
    draw = ImageDraw.Draw(img)
    font = find_font()

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        coords = [row[:2] for row in reader]

    for i, (x, y) in enumerate(coords):
        x = float(x) * UPSCALE
        y = float(y) * UPSCALE
        color = tuple(int(c * 255) for c in colors.to_rgb(PALETTE[i % len(PALETTE)]))
        r = MARKER_RADIUS
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="white", width=2)

    img.save(out_path, dpi=(DPI, DPI))
    print(f"Saved labeled RGB: {out_path}")

def make_spectra_plot(csv_path, out_png):
    """Generate normalized, Nature-style spectra plot."""
    basename = Path(csv_path).stem.replace("_spectra", "")
    folder = Path(csv_path).parent

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]

    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Intensity (a.u.)", fontsize=11)
    ax.set_xlim(450, 850)
    ax.set_ylim(0, 1.05)

    for i, row in enumerate(data):
        x, y = map(float, row[:2])
        mu_vals = normalize([float(v) for v in row[2:]])

        color = PALETTE[i % len(PALETTE)]
        label_rs = f"P{i+1} RS"
        label_gt = f"P{i+1} GT"

        # RS (dashed)
        ax.plot(WAVELENGTHS, mu_vals, linestyle=(0, (3, 2)), color=color, lw=2.2, label=label_rs)

        # GT (solid)
        gt_tif = folder / f"{basename}_gt.tif"
        if gt_tif.exists():
            import tifffile as tiff
            gt = tiff.imread(gt_tif)
            gt_spec = normalize(gt[:, int(y), int(x)])
            ax.plot(WAVELENGTHS, gt_spec, color=color, lw=2.2, label=label_gt)

    ax.legend(
        fontsize=8,
        frameon=False,
        ncol=2,
        loc="upper right",           # anchor corner of the legend
        bbox_to_anchor=(0.88, 1.0)   # x < 1.0 moves it left; y=1.0 keeps vertical alignment
    )

    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"✅ Saved spectra plot: {out_png}")

# -------------------------------------------------------------------
def process_dataset(folder):
    folder = Path(folder)
    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        csv_path = next(sub.glob("*_spectra.csv"), None)
        rgb_path = next(sub.glob("rgb_mu.png"), None)
        if csv_path and rgb_path:
            overlay_points_on_rgb(rgb_path, csv_path, sub / "rgb_labeled_nature.png")
            make_spectra_plot(csv_path, sub / "spectra_plot_nature.png")

# -------------------------------------------------------------------
if __name__ == "__main__":
    for root in ROOT_FOLDERS:
        process_dataset(root)
    print("All enhanced Nature-style figures generated.")
