#!/usr/bin/env python3
# python3.15 visualize_data.py
# ----------------------------------------------------------------------
# Title    : Data Visualization
# Authors  : Jadon Duff, Claude
# Date     : 2026-11-05
# Version  : v0.0.1
# Project  : Thermal Vision AI
# Software : Python 3.15.2
# ----------------------------------------------------------------------
# Description:
#   Displays a random sample of N processed thermal images from each class
#   in two color palettes (same images, so the palettes are comparable).
#
#   Input : data/<Class>_<Trial>/<image>.npy   (from process_raw_data.py)
#
# References:
#   None
# ----------------------------------------------------------------------

# ---- Libraries -------------------------------------------------------
lazy import matplotlib.pyplot as plt
lazy import numpy as np

from pathlib import Path
import random


# ---- Setup -----------------------------------------------------------
DATA_DIR = Path("data")
N = 10                           # images shown per class
PALETTES = ("inferno", "turbo")  # one row per palette, per class
UNIT = "F"                       # label only; must match process_raw_data.py
SEED = None                      # set an int for a repeatable sample


# ---- Load ------------------------------------------------------------
random.seed(SEED)
classes = sorted({p.name.split("_")[0] for p in DATA_DIR.glob("[0-9]*_[0-9]*")})

samples = {}  # class -> [(trial folder name, 2D array), ...]
for c in classes:
    files = sorted(DATA_DIR.glob(f"{c}_*/*.npy"))
    picked = random.sample(files, min(N, len(files)))
    samples[c] = [(f.parent.name, np.load(f)) for f in picked]

# One shared color scale so classes are compared honestly
all_temps = [a for imgs in samples.values() for _, a in imgs]
vmin, vmax = min(a.min() for a in all_temps), max(a.max() for a in all_temps)


# ---- Plot ------------------------------------------------------------
n_rows = len(classes) * len(PALETTES)
fig, axes = plt.subplots(n_rows, N, figsize=(1.6 * N, 2.2 * n_rows), squeeze=False)

for ax in axes.ravel():
    ax.axis("off")

for ci, c in enumerate(classes):
    for pi, cmap in enumerate(PALETTES):
        row = ci * len(PALETTES) + pi
        for col, (trial, temps) in enumerate(samples[c]):
            ax = axes[row, col]
            ax.imshow(temps, cmap=cmap, vmin=vmin, vmax=vmax)
            if pi == 0:
                ax.set_title(trial, fontsize=7)
        axes[row, 0].text(-0.08, 0.5, f"Class {c}\n{cmap}", transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=9)

fig.suptitle(f"Random sample of {N} per class  |  shared scale {vmin:.1f}–{vmax:.1f} °{UNIT}")
fig.tight_layout()
plt.show()
