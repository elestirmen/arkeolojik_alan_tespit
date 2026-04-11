import rasterio

files = [
    r"C:\d_surucusu\arkeolojik_alan_tespit\archaeo_detect_base\gizem_hocanin_veriler_5band_rgb_dsm_dtm_0p5m_bilinear.tif",
    r"C:\d_surucusu\arkeolojik_alan_tespit\archaeo_detect_base\karlik_vadi_rgb_dsm_dtm.tif"
]

for f in files:
    try:
        with rasterio.open(f) as src:
            print(f"File: {f}")
            print(f"Bands: {src.count}")
            print(f"Dtypes: {src.dtypes}")
            print("---")
    except Exception as e:
        print(f"Error on {f}: {e}")
