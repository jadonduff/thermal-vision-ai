#!/usr/bin/env python3
# python3.15 process_raw_data.py
# ----------------------------------------------------------------------
# Title    : Raw Data Preprocessing
# Authors  : Jadon Duff, Claude
# Date     : 2026-11-05
# Version  : v0.0.2
# Project  : Thermal Vision AI
# Software : Python 3.15.2
# ----------------------------------------------------------------------
# Description:
#   Preprocesses the raw thermal imagery into temperature data.
#
#   Input  : raw_data/<Class>_<Trial>/<image>.jpg   (10 images per trial)
#   Output : data/<Class>_<Trial>/<image>.npy       (same layout, 2D temps)
#
#   The trial folder structure is preserved so the train/test split can be
#   done at the trial level (never split a trial across train and test).
#
# References:
#   None
# ----------------------------------------------------------------------

# ---- Libraries -------------------------------------------------------
lazy import numpy as np
from PIL import Image

from pathlib import Path
import subprocess
import json
import io


# ---- Setup -----------------------------------------------------------
RAW_DIR = Path("raw_data")
OUT_DIR = Path("data")
UNIT = "F"  # "C", "K", or "F"


# ---- Helpers ---------------------------------------------------------
def get_temperatures(path, unit="C"):
    meta = json.loads(subprocess.run(
        ["exiftool", "-j", "-Emissivity", "-ObjectDistance", "-ReflectedApparentTemperature",
         "-AtmosphericTemperature", "-IRWindowTemperature", "-IRWindowTransmission",
         "-RelativeHumidity", "-PlanckR1", "-PlanckB", "-PlanckF", "-PlanckO", "-PlanckR2",
         "-AtmosphericTransAlpha1", "-AtmosphericTransAlpha2", "-AtmosphericTransBeta1",
         "-AtmosphericTransBeta2", "-AtmosphericTransX", "-RawValueRange", str(path)],
        capture_output=True, check=True).stdout)[0]
    num = lambda k: float(str(meta[k]).split()[0].rstrip("%"))

    png = subprocess.run(["exiftool", "-b", "-RawThermalImage", str(path)],
                         capture_output=True, check=True).stdout
    raw = np.array(Image.open(io.BytesIO(png))).astype(np.uint16)
    swapped = raw.byteswap()  # pick the byte order matching exiftool's RawValueRange
    span = lambda a: int(a.max()) - int(a.min())
    if abs(span(swapped) - num("RawValueRange")) < abs(span(raw) - num("RawValueRange")):
        raw = swapped
    raw = raw.astype(np.float64)

    E, OD, IRT, RH = num("Emissivity"), num("ObjectDistance"), num("IRWindowTransmission"), num("RelativeHumidity")
    RT, AT, WT = num("ReflectedApparentTemperature"), num("AtmosphericTemperature"), num("IRWindowTemperature")
    R1, B, F, O, R2 = (num(k) for k in ("PlanckR1", "PlanckB", "PlanckF", "PlanckO", "PlanckR2"))
    a1, a2, b1, b2, X = (num(k) for k in ("AtmosphericTransAlpha1", "AtmosphericTransAlpha2",
                                          "AtmosphericTransBeta1", "AtmosphericTransBeta2", "AtmosphericTransX"))

    h2o = RH / 100 * np.exp(1.5587 + 0.06939 * AT - 0.00027816 * AT**2 + 0.00000068455 * AT**3)
    tau = X * np.exp(-np.sqrt(OD / 2) * (a1 + b1 * np.sqrt(h2o))) + \
          (1 - X) * np.exp(-np.sqrt(OD / 2) * (a2 + b2 * np.sqrt(h2o)))
    rad = lambda T: R1 / (R2 * (np.exp(B / (T + 273.15)) - F)) - O  # temp(C) -> raw counts

    raw_obj = (raw / E / tau / IRT / tau
               - 2 * (1 - tau) / E / tau * rad(AT)   # atmosphere (both path halves)
               - (1 - IRT) / E / tau / IRT * rad(WT)  # IR window
               - (1 - E) / E * rad(RT))               # reflection
    celsius = B / np.log(R1 / (R2 * (raw_obj + O)) + F) - 273.15

    return {"C": celsius, "K": celsius + 273.15, "F": celsius * 9 / 5 + 32}[unit.upper()]


# ---- Main ------------------------------------------------------------
images = sorted(RAW_DIR.glob("[0-9]*_[0-9]*/*.jpg"))  # raw_data/Class_Trial/*.jpg

for i, path in enumerate(images, 1):
    out_path = OUT_DIR / path.parent.name / f"{path.stem}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temps = get_temperatures(path, UNIT).astype(np.float32)
    np.save(out_path, temps)
    print(f"[{i}/{len(images)}] {path} -> {out_path}  "
          f"{temps.shape} min {temps.min():.1f} max {temps.max():.1f} {UNIT}")
