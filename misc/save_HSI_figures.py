#!/usr/bin/env python3
"""
hsi_supplement_builder.py
Integrated GUI for browsing, analyzing, and exporting hyperspectral reconstructions.
 - Browse image pairs (GT / Reconstruction)
 - Click to plot spectra (GT vs μ ± σ)
 - Clear selections
 - Save all chosen spectra + high-res PNGs + TIFs for supplementary figures
"""

import os, csv
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
IMAGE_DIR = Path(r"Z:\Probabalistic_UNET\results\invertebrates_augmented_pol0\validation_latest\images")
OUT_DIR   = Path(r"Z:\Supplement_Figures\invertebrates")
OUT_DIR.mkdir(exist_ok=True, parents=True)

NUM_BANDS = 106
WAVELENGTHS = np.linspace(450, 850, NUM_BANDS)
CROP_SIZE = 120
SELECTED_WLS = np.linspace(450, 850, 5, dtype=int)

# --------------------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------------------
def center_crop(img, target_h, target_w):
    _, h, w = img.shape
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return img[:, top:top+target_h, left:left+target_w]

def HSI2RGB(hsi):
    wl = np.linspace(450, 850, hsi.shape[0])
    R = hsi[np.argmin(abs(wl - 650))]
    G = hsi[np.argmin(abs(wl - 550))]
    B = hsi[np.argmin(abs(wl - 460))]
    rgb = np.stack([R, G, B], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb

def save_gray(data, out_path, cmap="gray", dpi=300):
    plt.imsave(out_path, data, cmap=cmap, dpi=dpi)

# --------------------------------------------------------------------------
# MAIN CLASS
# --------------------------------------------------------------------------
class HSISupplementBuilder:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.gt_files   = sorted([f for f in os.listdir(image_dir) if f.startswith("cb_raw_")])
        self.pred_files = sorted([f for f in os.listdir(image_dir) if f.startswith("tl_gen_")])
        if not self.gt_files:
            raise FileNotFoundError("No cb_raw_*.tif found in folder!")

        self.index = 0
        self.selected_points = []
        self.colors = plt.cm.tab10.colors
        self.color_index = 0

        self.load_pair()

        # ---- GUI layout ----
        self.fig = plt.figure(figsize=(14, 7))
        gs = self.fig.add_gridspec(2, 4, height_ratios=[1, 0.15])

        self.ax_img = [self.fig.add_subplot(gs[0, i]) for i in range(3)]
        self.ax_spec = self.fig.add_subplot(gs[0, 3])
        self.ax_slider = self.fig.add_subplot(gs[1, 1:3])

        # ---- Slider ----
        self.slider = Slider(self.ax_slider, 'Band', 0, NUM_BANDS - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_band)

        # ---- Buttons ----
        self.btn_prev  = Button(plt.axes([0.05, 0.03, 0.1, 0.05]), 'Prev')
        self.btn_next  = Button(plt.axes([0.17, 0.03, 0.1, 0.05]), 'Next')
        self.btn_clear = Button(plt.axes([0.45, 0.03, 0.1, 0.05]), 'Clear')
        self.btn_save  = Button(plt.axes([0.85, 0.03, 0.1, 0.05]), 'Save Export')

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_clear.on_clicked(self.clear_points)
        self.btn_save.on_clicked(self.save_export)

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.band = 0
        self.update_display()
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()

    # ----------------------------------------------------------
    def load_pair(self):
        gt_path = self.image_dir / self.gt_files[self.index]
        pred_path = self.image_dir / self.pred_files[self.index]

        gt = tiff.imread(gt_path)
        pred_full = tiff.imread(pred_path)
        mu, sigma = pred_full[:NUM_BANDS], pred_full[NUM_BANDS:]

        self.gt = center_crop(gt, CROP_SIZE, CROP_SIZE)
        self.mu = center_crop(mu, CROP_SIZE, CROP_SIZE)
        self.sigma = center_crop(sigma, CROP_SIZE, CROP_SIZE)
        self.err = np.abs(self.gt - self.mu)

        self.basename = gt_path.stem.replace("cb_raw_", "")
        print(f"Loaded image {self.basename}")

    # ----------------------------------------------------------
    def update_display(self):
        band = int(self.band)
        wl = WAVELENGTHS[band]
        panels = [
            (self.ax_img[0], self.gt[band], f"GT ({wl:.0f} nm)"),
            (self.ax_img[1], self.mu[band], f"Recon ({wl:.0f} nm)"),
            (self.ax_img[2], self.err[band], f"|Error|"),
        ]
        for ax, data, title in panels:
            ax.cla()
            ax.imshow(data, cmap='gray' if 'Error' not in title else 'hot')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            for (x, y, c) in self.selected_points:
                ax.plot(x, y, "o", color=c, markersize=6, markeredgecolor='white')
        self.update_spectra_plot()
        self.fig.suptitle(f"{self.basename} ({self.index+1}/{len(self.gt_files)})", fontsize=13)
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------
    def update_spectra_plot(self):
        self.ax_spec.cla()
        self.ax_spec.set_title("Spectra (GT vs μ ± σ)")
        self.ax_spec.set_xlabel("Wavelength (nm)")
        self.ax_spec.set_ylabel("Intensity (norm.)")
        if not self.selected_points:
            self.ax_spec.text(0.5, 0.5, "Click pixels to plot spectra",
                              ha='center', va='center', color='gray', fontsize=10)
        else:
            for (x, y, c) in self.selected_points:
                gt_spec = self.gt[:, y, x]
                mu_spec = self.mu[:, y, x]
                sigma_spec = self.sigma[:, y, x]
                gt_spec = gt_spec / (gt_spec.max() + 1e-8)
                mu_spec = mu_spec / (mu_spec.max() + 1e-8)
                self.ax_spec.plot(WAVELENGTHS, gt_spec, color=c, lw=2, label=f"GT ({x},{y})")
                self.ax_spec.plot(WAVELENGTHS, mu_spec, "--", color=c, lw=1.5)
                self.ax_spec.fill_between(WAVELENGTHS, mu_spec - sigma_spec, mu_spec + sigma_spec,
                                          color=c, alpha=0.25)
            self.ax_spec.legend(fontsize=7)

    # ----------------------------------------------------------
    def onclick(self, event):
        if event.inaxes in self.ax_img[:3] and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1
            self.selected_points.append((x, y, color))
            self.update_display()

    def update_band(self, val):
        self.band = int(val)
        self.update_display()

    def clear_points(self, event=None):
        self.selected_points = []
        self.color_index = 0
        self.update_display()

    def next_image(self, event=None):
        self.index = (self.index + 1) % len(self.gt_files)
        self.clear_points()
        self.load_pair()
        self.update_display()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % len(self.gt_files)
        self.clear_points()
        self.load_pair()
        self.update_display()

    # ----------------------------------------------------------
    def save_export(self, event=None):
        out_dir = OUT_DIR / self.basename
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{self.basename}_spectra.csv"

        # --- Save spectra to CSV ---
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", *[f"{wl:.1f}nm" for wl in WAVELENGTHS]])
            for (x, y, _) in self.selected_points:
                writer.writerow([x, y, *self.mu[:, y, x]])
        print(f"✅ Saved spectra CSV: {csv_path}")

        # --- Save high-res images (GT, μ, |Err|, σ, RGB) ---
        for wl in SELECTED_WLS:
            idx = np.argmin(abs(WAVELENGTHS - wl))
            save_gray(self.gt[idx], out_dir / f"band_{wl}_gt.png")
            save_gray(self.mu[idx], out_dir / f"band_{wl}_mu.png")
            save_gray(self.err[idx], out_dir / f"band_{wl}_err.png", cmap="hot")
            save_gray(self.sigma[idx], out_dir / f"band_{wl}_sigma.png")

        rgb_gt = HSI2RGB(self.gt)
        rgb_mu = HSI2RGB(self.mu)
        plt.imsave(out_dir / "rgb_gt.png", rgb_gt, dpi=300)
        plt.imsave(out_dir / "rgb_mu.png", rgb_mu, dpi=300)

        # --- Save cropped TIFs ---
        tiff.imwrite(out_dir / f"{self.basename}_gt.tif", self.gt.astype(np.float32))
        tiff.imwrite(out_dir / f"{self.basename}_mu.tif", self.mu.astype(np.float32))
        tiff.imwrite(out_dir / f"{self.basename}_sigma.tif", self.sigma.astype(np.float32))
        print(f"✅ Export complete for {self.basename} -> {out_dir}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    HSISupplementBuilder(IMAGE_DIR)
