#!/usr/bin/env python3
"""
Simply displays each image (Cubert + Thorlabs, cropped + uncropped)
so you can save manually. No labels, no axes.
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
IMAGES = [
    r"D:\banknotes_4-15\processed\cumulative\cubert\image_24_cubert_cubert.tif",
    r"D:\banknotes_4-15\processed\cumulative\thorlabs\image_24_thorlabs_thorlabs.tif",
    r"D:\banknotes_4-15\original\cubert\image_24_cubert.tif",
    r"D:\banknotes_4-15\original\thorlabs\image_24_thorlabs.tif",
]

# ---------------- Helper ----------------
def normalize(img):
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img

# ---------------- Show Each ----------------
for path in IMAGES:
    data = tifffile.imread(path)
    if "thorlabs" in path.lower():
        img = data[0] if data.ndim == 3 else data
    else:  # Cubert
        img = np.mean(data, axis=0)
    
    img = normalize(img)
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")

plt.show()
