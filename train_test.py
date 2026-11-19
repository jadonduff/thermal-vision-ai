#!/usr/bin/env python3
"""
Train a binary footprint classifier on trial-separated thermal arrays.

Dependencies: NumPy and PyTorch
Run from the repository root: python train_test.py
The CNN starts from random initialization; no pretrained weights are used.
Class 1 is the positive (footprint) class. No generated files are written.
"""

import array
import ast
import hashlib
import math
from pathlib import Path
import random
import re
import struct
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("data")
SEED = 42
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 16
IMAGE_SIZE = (80, 60)
CONFIRMED_GROUPS = (
    ("1_3/20261105T214430.npy", "1_3/20261105T214433.npy"),
)


def load_image(path):
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not an NPY file: {path}")
        version = stream.read(2)
        if version == b"\x01\x00":
            length_size, encoding = 2, "latin1"
        elif version in (b"\x02\x00", b"\x03\x00"):
            length_size = 4
            encoding = "utf-8" if version == b"\x03\x00" else "latin1"
        else:
            raise ValueError(f"Unsupported NPY version in {path}: {version!r}")
        length = int.from_bytes(stream.read(length_size), "little")
        header = ast.literal_eval(stream.read(length).decode(encoding).strip())
        if (header["descr"] not in ("<f4", ">f4", "=f4") or
                header["fortran_order"] or len(header["shape"]) != 2):
            raise ValueError(f"Expected a C-order 2D float32 array: {path}")
        height, width = header["shape"]
        payload = stream.read()
    if len(payload) != height * width * 4 or min(height, width) == 0:
        raise ValueError(f"Invalid array size: {path}")
    values = array.array("f")
    values.frombytes(payload)
    byteorder = header["descr"][0]
    if (byteorder == ">" and sys.byteorder == "little") or (byteorder == "<" and sys.byteorder == "big"):
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite temperatures in {path}")
    return (height, width), values


def load_dataset():
    records = []
    for path in sorted(DATA_DIR.glob("*/*.npy")):
        match = re.fullmatch(r"([01])_\d+", path.parent.name)
        if not match:
            raise ValueError(f"Unexpected trial folder: {path.parent}")
        shape, values = load_image(path)
        # Hash the complete numeric contents and shape, independent of NPY headers.
        canonical = array.array("f", values)
        if sys.byteorder == "big":
            canonical.byteswap()
        digest = hashlib.sha256(struct.pack("<QQ", *shape) + canonical.tobytes()).digest()
        records.append((path, int(match.group(1)), path.parent.name, digest, np.asarray(values, dtype=np.float32).reshape(shape)))
    if not records:
        raise ValueError(f"No images found under {DATA_DIR}")
    return records


def split_trials(records):
    """Link duplicate trials, then find an exact, class-balanced 80/20 split."""
    trials = sorted({r[2] for r in records})
    parent = {trial: trial for trial in trials}

    def find(trial):
        while parent[trial] != trial:
            parent[trial] = parent[parent[trial]]
            trial = parent[trial]
        return trial

    def union(first, second):
        parent[find(first)] = find(second)

    paths = {r[0].relative_to(DATA_DIR).as_posix(): r for r in records}
    by_hash = {}
    for record in records:
        by_hash.setdefault(record[3], []).append(record)
    groups = [group for group in by_hash.values() if len(group) > 1]
    for group in groups:
        for record in group[1:]:
            union(group[0][2], record[2])
    for names in CONFIRMED_GROUPS:
        if any(name not in paths for name in names):
            raise ValueError(f"Confirmed duplicate group missing from dataset: {names}")
        group = [paths[name] for name in names]
        groups.append(group)
        for record in group[1:]:
            union(group[0][2], record[2])
    print(f"Duplicate audit: {sum(len(g) > 1 for g in by_hash.values())} exact-content groups; "
          f"{len(CONFIRMED_GROUPS)} manually confirmed groups.")
    for group in groups:
        print("  linked:", ", ".join(str(record[0]) for record in group))
    candidates = [str(r[0]) for r in records if re.search(r" \(\d+\)$", r[0].stem)]
    print(f"Timestamp-suffix candidates: {len(candidates)} (full contents compared).")

    totals = [sum(r[1] == label for r in records) for label in (0, 1)]
    if any(total == 0 or total % 5 for total in totals):
        raise ValueError(f"Exact 80/20 per-class split impossible for counts {totals}")
    targets = tuple(total // 5 for total in totals)
    components = {}
    for trial in trials:
        components.setdefault(find(trial), set()).add(trial)
    choices = list(components.values())
    random.Random(SEED).shuffle(choices)
    # Dynamic programming stores one set of connected trials for each count pair.
    possible = {(0, 0): set()}
    for component in choices:
        counts = tuple(sum(r[2] in component and r[1] == label for r in records) for label in (0, 1))
        for count, selected in list(possible.items()):
            new = tuple(count[i] + counts[i] for i in (0, 1))
            if all(new[i] <= targets[i] for i in (0, 1)) and new not in possible:
                possible[new] = selected | component
    if targets not in possible:
        raise ValueError(f"Exact 80/20 split impossible with linked trials; target test counts {targets}")
    test_trials = possible[targets]
    train = [r for r in records if r[2] not in test_trials]
    test = [r for r in records if r[2] in test_trials]
    assert not ({r[2] for r in train} & {r[2] for r in test})
    assert all(len({r[2] in test_trials for r in group}) == 1 for group in groups)
    print(f"Train: {len(train)} images; test: {len(test)} images; test trials: {', '.join(sorted(test_trials))}")
    for label in (0, 1):
        print(f"  class {label}: train {sum(r[1] == label for r in train)}, test {sum(r[1] == label for r in test)}")
    return train, test


class FootprintCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 3)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 3, 1))

    def forward(self, images):
        return self.head(self.convolutions(images)).squeeze(1)


def train_and_evaluate(train, test):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Global pixel statistics are computed from original training images only.
    pixel_count = sum(r[4].size for r in train)
    mean = sum(r[4].sum(dtype=np.float64) for r in train) / pixel_count
    variance = sum(np.square(r[4].astype(np.float64) - mean).sum() for r in train) / pixel_count
    scale = max(math.sqrt(variance), 1e-6)

    def tensors(records):
        # Area downsampling preserves the image layout while making CPU training practical.
        images = torch.from_numpy(np.stack([r[4] for r in records])).unsqueeze(1)
        images = F.interpolate(images, size=IMAGE_SIZE, mode="area")
        images = (images - mean) / scale
        labels = torch.tensor([r[1] for r in records], dtype=torch.float32)
        return images, labels

    train_images, train_labels = tensors(train)
    test_images, test_labels = tensors(test)
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model = FootprintCNN().to(device)
    initial_weights = model.convolutions[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"CNN: random initialization; device: {device}; fixed epochs: {EPOCHS}; no augmentation")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        print(f"Epoch {epoch + 1:2d}/{EPOCHS}: training loss {total_loss / len(train):.4f}")
    changed = not torch.equal(initial_weights, model.convolutions[0].weight.detach())
    if not changed:
        raise RuntimeError("Convolution weights did not update during training")
    print("Convolution weights updated: yes")

    model.eval()
    with torch.inference_mode():
        predicted = (model(test_images.to(device)) >= 0).cpu()
    truth = test_labels.bool()
    tp = int((predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    assert tp + tn + fp + fn == len(test)
    accuracy = (tp + tn) / len(test)
    print(f"Test results (class 1 positive): TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Accuracy: {accuracy:.3f}; strictly above 80% target: {'yes' if accuracy > 0.8 else 'no'}")
    print(f"Precision: {tp / (tp + fp):.3f}" if tp + fp else "Precision: undefined (no positive predictions)")
    print(f"Recall: {tp / (tp + fn):.3f}" if tp + fn else "Recall: undefined (no positive examples)")
    # Wilson interval is descriptive under independent Bernoulli observations.
    n = len(test)
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (accuracy + z * z / (2 * n)) / denominator
    half = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / denominator
    print(f"95% Wilson interval for image-level accuracy: [{center - half:.3f}, {center + half:.3f}]")
    print("This interval assumes independent images; images within the four held-out trials may be correlated.")
    print("Four held-out trials are insufficient for a reliable trial-level confidence interval; no statistical significance claim is made.")

def main():
    try:
        train, test = split_trials(load_dataset())
        train_and_evaluate(train, test)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())