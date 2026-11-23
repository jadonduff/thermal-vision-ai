#!/usr/bin/env python3
# python3.15 augmentation.py
# ----------------------------------------------------------------------
# Title    : Data Visualization
# Authors  : Jacob Lebkuecher, Claude
# Date     : 2026-11-11
# Version  : v0.0.1
# Project  : Thermal Vision AI
# Software : Python 3.15.2
# ----------------------------------------------------------------------
# Description:
#   Augments the npy images from the data folder. Each is rotated 4 times,
#   then applied a random shift, and finally cropped. After execution,
#   there should be 2000+ images.
#
# References:
#   None
# ----------------------------------------------------------------------

# ---- Libraries -------------------------------------------------------
from pathlib import Path
import numpy as np
from skimage.transform import resize

# ---- Setup -----------------------------------------------------------
rng = np.random.default_rng(0)
padding = 0.08
crop_range = (0.85, 1.0)
variants = 2
data_folder = Path("data")
augmented_data_folder = Path("data_augmented")

# ---- Functions ------------------------------------------------------------
def random_shift(npy_data):
    height, width = npy_data.shape

    pad = int(min(height, width) * padding)

    dy, dx = (int(v) for v in rng.integers(-pad, pad + 1, size = 2))
    padded = np.pad(npy_data, pad, mode="edge")

    return padded[pad + dy:pad + dy + height, pad + dx:pad + dx + width], dy, dx

def random_crop(npy_data):
    height, width = npy_data.shape

    fraction = round(float(rng.uniform(*crop_range)), 2)

    crop_height, crop_width = int(height * fraction), int(width * fraction)
    crop_top = int(rng.integers(0, height - crop_height + 1))
    crop_left = int(rng.integers(0, width - crop_width + 1))

    crop = npy_data[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]

    cropped_image = resize(crop, npy_data.shape, order = 1, preserve_range = True).astype(npy_data.dtype)

    return cropped_image, fraction

# ---- Main Loop ------------------------------------------------------------
# Iterate through folders in data folder
for folder in data_folder.iterdir():
    # Format destination dir name and create it (if it doesn't exist yet)
    to_dir = augmented_data_folder / folder.name
    to_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through images in current folder.
    for file in sorted(folder.glob("*.npy")):
        # Load the npy data from the image
        npy_data = np.load(file)

        # Run 4 augmentations (1 for each rotation)
        for k in range(4):
            # Rotate the npy data
            deg = k*90
            rotated = np.rot90(npy_data, k=k)

            # Save it
            np.save(to_dir / f"{file.stem}_r{deg}.npy", np.ascontiguousarray(rotated))

            # Add it to a set of seen images so no duplicates are stored
            seen = set()

            # Run for all variants
            while len(seen) < variants:
                # Randomly shift and crop the rotated npy data
                shifted, dy, dx = random_shift(rotated)
                cropped, fraction = random_crop(shifted)

                # Generate the augmented image name
                name = f"{file.stem}_r{deg}_s{dy}x{dy}_m{fraction}.npy"

                # Check if name has been already seen
                if name in seen:
                    continue

                # Save the augmented image and add it to set of seen images
                seen.add(name)
                np.save(to_dir / name, np.ascontiguousarray(cropped))