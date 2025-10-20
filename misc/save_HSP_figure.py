#!/usr/bin/env python3
"""
Export spectral–polarimetric frames and uncertainty maps.

Each gen .tif = (212, 128, 128):
  - first 106 bands = reconstruction
  - last 106 bands  = uncertainty (σ)
Each GT .tif = (106, 128, 128).

Output:
 - gen/:      reconstructed frames & Stokes
 - gt/:       ground truth frames & Stokes
 - uncertainty/: S0-style uncertainty maps
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SETS = [
    dict(
        gen=[
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol0\validation_latest\images\tl_gen_image_24_thorlabs_thorlabs_5.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol45\validation_latest\images\tl_gen_image_24_thorlabs_thorlabs_5.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol90\validation_latest\images\tl_gen_image_24_thorlabs_thorlabs_5.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol135\validation_latest\images\tl_gen_image_24_thorlabs_thorlabs_5.tif",
        ],
        gt=[
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol0\validation_latest\images\cb_raw_image_24_thorlabs_thorlabs_5.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol45\validation_latest\images\cb_raw_image_24_thorlabs_thorlabs_5.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol90\validation_latest\images\cb_raw_image_24_thorlabs_thorlabs_5.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol135\validation_latest\images\cb_raw_image_24_thorlabs_thorlabs_5.tif",
        ],
    ),
    dict(
        gen=[
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol0\validation_latest\images\tl_gen_image_20_thorlabs_thorlabs_3.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol45\validation_latest\images\tl_gen_image_20_thorlabs_thorlabs_3.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol90\validation_latest\images\tl_gen_image_20_thorlabs_thorlabs_3.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol135\validation_latest\images\tl_gen_image_20_thorlabs_thorlabs_3.tif",
        ],
        gt=[
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol0\validation_latest\images\cb_raw_image_20_thorlabs_thorlabs_3.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol45\validation_latest\images\cb_raw_image_20_thorlabs_thorlabs_3.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol90\validation_latest\images\cb_raw_image_20_thorlabs_thorlabs_3.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol135\validation_latest\images\cb_raw_image_20_thorlabs_thorlabs_3.tif",
        ],
    ),
    dict(
        gen=[
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol0\validation_latest\images\tl_gen_image_58_thorlabs_thorlabs_11.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol45\validation_latest\images\tl_gen_image_58_thorlabs_thorlabs_11.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol90\validation_latest\images\tl_gen_image_58_thorlabs_thorlabs_11.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol135\validation_latest\images\tl_gen_image_58_thorlabs_thorlabs_11.tif",
        ],
        gt=[
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol0\validation_latest\images\cb_raw_image_58_thorlabs_thorlabs_11.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol45\validation_latest\images\cb_raw_image_58_thorlabs_thorlabs_11.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol90\validation_latest\images\cb_raw_image_58_thorlabs_thorlabs_11.tif",
            r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol135\validation_latest\images\cb_raw_image_58_thorlabs_thorlabs_11.tif",
        ],
    ),
]

OUTROOT = Path(r"Z:\Probabalistic_UNET\exports_spectral_polarimetric_uncertainty")
OUTROOT.mkdir(parents=True, exist_ok=True)

WLS = np.linspace(450, 850, 212)
TARGET_WLS = [450,475,500,525, 550, 575, 600, 650]
NUM_BANDS = 106
CROP = 120

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def center_crop(arr, size=120):
    _, h, w = arr.shape
    sy, sx = h//2 - size//2, w//2 - size//2
    return arr[:, sy:sy+size, sx:sx+size]

def normalize(img):
    img = img.astype(float)
    img -= img.min()
    img /= img.max() if img.max() > 0 else 1
    return img

def save_img(img, path):
    plt.imsave(path, normalize(img), cmap='gray', dpi=300)

def compute_stokes(cubes):
    I0, I45, I90, I135 = cubes
    S0 = 0.5 * (I0 + I45 + I90 + I135)
    S1 = I0 - I90
    S2 = I45 - I135
    return dict(S0=S0, S1=S1, S2=S2)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
for si, entry in enumerate(SETS, 1):
    print(f"\nProcessing set {si}...")
    setdir = OUTROOT / f"set{si}"
    (setdir / "gen").mkdir(parents=True, exist_ok=True)
    (setdir / "gt").mkdir(parents=True, exist_ok=True)
    (setdir / "uncertainty").mkdir(parents=True, exist_ok=True)

    # ----- load generated (split into mean + sigma) -----
    gen_cubes = []
    unc_cubes = []
    for path in entry["gen"]:
        cube = tifffile.imread(path)
        gen_cubes.append(center_crop(cube[:NUM_BANDS]))
        unc_cubes.append(center_crop(cube[NUM_BANDS:]))
    gen_cubes = np.stack(gen_cubes, axis=0)
    unc_cubes = np.stack(unc_cubes, axis=0)

    # ----- load ground truth -----
    gt_cubes = [center_crop(tifffile.imread(p)) for p in entry["gt"]]
    gt_cubes = np.stack(gt_cubes, axis=0)

    # ----- compute stokes -----
    stokes_gen = compute_stokes(gen_cubes)
    stokes_gt = compute_stokes(gt_cubes)
    stokes_unc = compute_stokes(unc_cubes)

    for wl in TARGET_WLS:
        idx = np.argmin(np.abs(WLS[:NUM_BANDS] - wl))
        wl_tag = f"{int(wl)}nm"

        for theta, (gen, gt, unc) in zip([0,45,90,135], zip(gen_cubes, gt_cubes, unc_cubes)):
            save_img(gen[idx], setdir / "gen" / f"gen_theta{theta}_{wl_tag}.png")
            save_img(gt[idx], setdir / "gt" / f"gt_theta{theta}_{wl_tag}.png")
            save_img(unc[idx], setdir / "uncertainty" / f"unc_theta{theta}_{wl_tag}.png")

        for key in ["S0", "S1", "S2"]:
            save_img(stokes_gen[key][idx], setdir / "gen" / f"gen_{key}_{wl_tag}.png")
            save_img(stokes_gt[key][idx], setdir / "gt" / f"gt_{key}_{wl_tag}.png")
            save_img(stokes_unc[key][idx], setdir / "uncertainty" / f"unc_{key}_{wl_tag}.png")

print("\n✅ All exports complete.")
