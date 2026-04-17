import numpy as np
import sys

filename = sys.argv[1]
data = np.load(filename)
with open("npz_info.txt", "w") as f:
    f.write(f"Keys: {list(data.keys())}\n")
    for key in data.keys():
        f.write(f"Key: {key}, Shape: {data[key].shape}, Dtype: {data[key].dtype}, Min: {np.min(data[key])}, Max: {np.max(data[key])}\n")
