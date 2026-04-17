import numpy as np
import sys

def check_bands(file_path):
    with open("band_stats.txt", "w") as f:
        f.write(f"Checking {file_path}\n")
        data = np.load(file_path)
        keys = list(data.keys())
        key = 'image' if 'image' in keys else keys[0]
        img = data[key]
        
        names = ["R", "G", "B", "DSM", "DTM", "SVF", "Pos_Op", "Neg_Op", "LRM", "Slope", "nDSM", "TPI"]
        
        for i in range(img.shape[0]):
            band = img[i]
            name = names[i] if i < len(names) else f"B{i+1}"
            f.write(f"Band {i:2d} ({name:8s}): "
                  f"min={band.min():.4f}, "
                  f"max={band.max():.4f}, "
                  f"mean={band.mean():.4f}, "
                  f"std={band.std():.4f}, "
                  f"zeros={np.sum(band == 0.0)}\n")

if __name__ == "__main__":
    check_bands(sys.argv[1])
