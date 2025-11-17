#!/usr/bin/env python3
import os
import numpy as np
import tifffile as tiff
import csv
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import re

# ------------------------------------------------------------
# --- CONFIGURATION ------------------------------------------
# ------------------------------------------------------------
# Hardcode your image directory here:
IMAGE_DIR = r"/home/matthew-morales/Documents/EndoDBV1 Results 11-15-2025/bronch_lcd_pol-1/validation_latest/images"
NUM_IMAGES = 173
METRICS_CSV = os.path.join(IMAGE_DIR, "../metrics.csv")
CROP_SIZE = 352

# ------------------------------------------------------------
# --- HELPER FUNCTIONS ---------------------------------------
# ------------------------------------------------------------
def center_crop(img, target_h, target_w):
    """Center-crop [C, H, W] or [H, W]."""
    if img.ndim == 3:
        _, h, w = img.shape
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return img[:, top:top+target_h, left:left+target_w]
    elif img.ndim == 2:
        h, w = img.shape
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return img[top:top+target_h, left:left+target_w]
    else:
        raise ValueError("Input must have 2 or 3 dims")

def compute_rase(gt, pred, eps=1e-12):
    C = gt.shape[0]
    mse_per_band = np.mean((gt - pred) ** 2, axis=(1, 2))
    mu = float(np.mean(gt))
    return float((100.0 / (mu + eps)) * np.sqrt(np.mean(mse_per_band)))

def spectral_fidelity(gt, pred):
    n_bands, height, width = gt.shape
    pixel_fidelity = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            gt_spectrum = gt[:, y, x]
            pred_spectrum = pred[:, y, x]
            if np.sum(gt_spectrum**2) == 0 or np.sum(pred_spectrum**2) == 0:
                pixel_fidelity[y, x] = 0
                continue
            dot = np.sum(gt_spectrum * pred_spectrum)
            pixel_fidelity[y, x] = dot / (np.linalg.norm(gt_spectrum) * np.linalg.norm(pred_spectrum))
    return np.mean(pixel_fidelity)

def relative_spectral_error_l1(gt, pred, eps=1e-6):
    C, H, W = gt.shape
    gt_flat = gt.reshape(C, -1)
    pred_flat = pred.reshape(C, -1)
    denom = np.sum(np.abs(gt_flat), axis=0)
    num = np.sum(np.abs(gt_flat - pred_flat), axis=0)
    valid = denom > eps
    return float(np.mean(num[valid] / denom[valid])) if np.any(valid) else np.nan

def spectral_reconstruction_error_l1(gt, pred):
    return np.mean(np.abs(gt - pred))

def spectral_reconstruction_error_l2(gt, pred):
    return np.sqrt(np.mean((gt - pred) ** 2))

def relative_spectral_reconstruction_error(gt, pred, norm='l1'):
    if norm == 'l1':
        sre = spectral_reconstruction_error_l1(gt, pred)
        gt_mag = np.mean(np.abs(gt))
    else:
        sre = spectral_reconstruction_error_l2(gt, pred)
        gt_mag = np.sqrt(np.mean(gt ** 2))
    return (sre / gt_mag) * 100 if gt_mag != 0 else np.nan

def analyze_sigma(sigma):
    return {
        "sigma_mean": float(np.mean(sigma)),
        "sigma_std": float(np.std(sigma)),
        "sigma_min": float(np.min(sigma)),
        "sigma_max": float(np.max(sigma)),
        "sigma_median": float(np.median(sigma)),
    }

# ------------------------------------------------------------
# --- MAIN METRIC ANALYSIS -----------------------------------
# ------------------------------------------------------------
def main():
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".tif")])
    gt_files = sorted([f for f in image_files if f.startswith("cb_raw_")])
    pred_files = sorted([f for f in image_files if f.startswith("tl_gen_")])

    ssim_3d_list, ssim_2d_list, mse_list, mae_list, rase_list, fidelity_list = [], [], [], [], [], []
    RSE_list, sre_l1_list, sre_l2_list, rsre_l1_list, rsre_l2_list = [], [], [], [], []
    sigma_means, sigma_stds, sigma_mins, sigma_maxs, sigma_medians = [], [], [], [], []

    for idx in range(min(NUM_IMAGES, len(gt_files), len(pred_files))):
        gt = tiff.imread(os.path.join(IMAGE_DIR, gt_files[idx]))      # [106, H, W]
        pred_full = tiff.imread(os.path.join(IMAGE_DIR, pred_files[idx]))  # [212, 256, 256]
        # Split into μ and σ and crop
        n = gt.shape[0]
        mu = center_crop(pred_full[:n], CROP_SIZE, CROP_SIZE)
        sigma = center_crop(pred_full[n:], CROP_SIZE, CROP_SIZE)
        gt = center_crop(gt, CROP_SIZE, CROP_SIZE)

        # Compute metrics
        ssim_3d = ssim(gt, mu, data_range=gt.max() - gt.min(), channel_axis=0)
        ssim_2d = ssim(gt[0], mu[0], data_range=gt[0].max() - gt[0].min())
        mse = np.mean((gt - mu) ** 2)
        mae = np.mean(np.abs(gt - mu))
        rase = compute_rase(gt, mu)
        fidelity = spectral_fidelity(gt, mu)
        RSE = relative_spectral_error_l1(gt, mu)
        sre_l1 = spectral_reconstruction_error_l1(gt, mu)
        sre_l2 = spectral_reconstruction_error_l2(gt, mu)
        rsre_l1 = relative_spectral_reconstruction_error(gt, mu, norm='l1')
        rsre_l2 = relative_spectral_reconstruction_error(gt, mu, norm='l2')

        # Aggregate
        ssim_3d_list.append(ssim_3d)
        ssim_2d_list.append(ssim_2d)
        mse_list.append(mse)
        mae_list.append(mae)
        rase_list.append(rase)
        fidelity_list.append(fidelity)
        RSE_list.append(RSE)
        sre_l1_list.append(sre_l1)
        sre_l2_list.append(sre_l2)
        rsre_l1_list.append(rsre_l1)
        rsre_l2_list.append(rsre_l2)

        stats = analyze_sigma(sigma)
        sigma_means.append(stats["sigma_mean"])
        sigma_stds.append(stats["sigma_std"])
        sigma_mins.append(stats["sigma_min"])
        sigma_maxs.append(stats["sigma_max"])
        sigma_medians.append(stats["sigma_median"])

    metrics = {
        "avg_ssim_3d": np.mean(ssim_3d_list),
        "avg_ssim_2d": np.mean(ssim_2d_list),
        "avg_mse": np.mean(mse_list),
        "avg_mae": np.mean(mae_list),
        "avg_rase": np.mean(rase_list),
        "avg_fidelity": np.mean(fidelity_list),
        "avg_RSE": np.mean(RSE_list),
        "avg_sre_l1": np.mean(sre_l1_list),
        "avg_sre_l2": np.mean(sre_l2_list),
        "avg_rsre_l1": np.mean(rsre_l1_list),
        "avg_rsre_l2": np.mean(rsre_l2_list),
        "avg_sigma_mean": np.mean(sigma_means),
        "avg_sigma_std": np.mean(sigma_stds),
        "avg_sigma_min": np.mean(sigma_mins),
        "avg_sigma_max": np.mean(sigma_maxs),
        "avg_sigma_median": np.mean(sigma_medians),
        "num_images": len(ssim_3d_list)
    }

    print(f"\nAverage Metrics Over {len(ssim_3d_list)} Valid Image Pairs:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(METRICS_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics written to {METRICS_CSV}")

if __name__ == "__main__":
    main()
