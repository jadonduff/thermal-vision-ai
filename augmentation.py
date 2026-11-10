from pathlib import Path
import numpy as np

def rotate_file(npy_data, destination_path, axes=(0, 1)):
    rot = 0

    for _ in range(4):
        data = np.rot90(npy_data, axes=axes)
        name = destination_path + "_" + str(rot)
        np.save(f"data_augmented/{name}", data)
        rot += 90

data_folder = Path("data")
augmented_data_folder = Path("data_augmented")

for folder in data_folder.iterdir():
    for file in folder.iterdir():
        npy_data = np.load(file)
        rotate_file(npy_data, file.name)