import rasterio
import numpy as np

tif_path = r"C:\d_surucusu\arkeolojik_alan_tespit\archaeo_detect_base\karlik_rgb_dsm_dtm_5band.tif"

with rasterio.open(tif_path) as src:
    with open("tiff_stats.txt", "w") as f:
        f.write(f"Bands: {src.count}\n")
        f.write(f"Shape: {src.width}x{src.height}\n")
        f.write(f"Dtypes: {src.dtypes}\n")
        f.write(f"NoData: {src.nodatavals}\n")
        
        window = rasterio.windows.Window(13056, 7936, 256, 256)
        data = src.read(window=window)
        for i in range(src.count):
            f.write(f"Band {i+1} stats -> min: {data[i].min():.4f}, max: {data[i].max():.4f}, mean: {data[i].mean():.4f}, std: {data[i].std():.4f}\n")
