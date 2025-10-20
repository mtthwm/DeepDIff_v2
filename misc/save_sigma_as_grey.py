#!/usr/bin/env python3
"""
Convert all 'sigma' figures from hot colormap to grayscale.

For each sigma image:
 - Renames the original (hot) image to *_hot.ext
 - Saves a grayscale version using the original filename
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import shutil

# --- Define root folders ---
ROOT_FOLDERS = [
    r"Z:\Supplement_Figures\produce",
    r"Z:\Supplement_Figures\banknotes",
    r"Z:\Supplement_Figures\fossils_flora",
    r"Z:\Supplement_Figures\invertebrates",
]

# --- File types to process ---
EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

for root in ROOT_FOLDERS:
    root_path = Path(root)
    if not root_path.exists():
        print(f"[WARN] Folder not found: {root_path}")
        continue

    for f in root_path.rglob("*sigma*"):
        if f.suffix.lower() not in EXTS:
            continue

        try:
            # --- Load image ---
            if f.suffix.lower() in (".tif", ".tiff"):
                img = tifffile.imread(f)
            else:
                img = np.array(Image.open(f))

            # --- Normalize and convert to grayscale ---
            img = img.astype(np.float32)
            img -= img.min()
            if img.max() > 0:
                img /= img.max()

            # If RGB, convert to grayscale
            if img.ndim == 3:
                img = img.mean(axis=-1)

            # --- Move the original hot file to *_hot.ext ---
            hot_path = f.with_name(f.stem + "_hot" + f.suffix)
            if not hot_path.exists():
                shutil.move(str(f), str(hot_path))
                print(f"[MOVE] Renamed original to: {hot_path.name}")
            else:
                print(f"[SKIP] Hot version already exists: {hot_path.name}")

            # --- Save grayscale to original filename ---
            plt.imsave(f, img, cmap="gray", vmin=0, vmax=1)
            print(f"[OK] Overwrote {f.name} with grayscale")

        except Exception as e:
            print(f"[ERR] Failed on {f}: {e}")
