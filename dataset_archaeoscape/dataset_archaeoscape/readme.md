# Archaeoscape dataset instructions

- Archaeoscape dataset consists of 23 de-georefenced parcels with fictional names, covering 888 km2 in total. See `./data/parcels.gpkg` file.
- Two modalities are available per each parcel, orthophoto RGB and LiDAR DTM. See `./rasters/rgb` and `./rasters/dtm` folders.
- The vector annotations for 4 features classes (hydrology, mound, temple, void) can be found in `./data/features.gpkg`.

## Overview

- For a quick overview of the features please see the `./doc/dataset_overview.pdf` file.
- To inspect both features and RGB/DTM rasters, please install the open-source QGIS tool and open the `./doc/dataset_overview_qgis/dataset_overview.qgz` project.


## License
The Archaeoscape dataset is distributed under a custom license, which prohibits commercial use, redistribution and attempts at localizing the data. If you have received this dataset from someone, you have to submit a dataset request at our website before legally allowed to use it. Please do so via this link: <https://archaeoscape.ai/data/2024/>

We provide the full text of the license below:

***

The École française d'Extrême-Orient (EFEO) makes the Archaeoscape dataset (the “DATASET”) available for research and educational purposes to individuals or entities ("USER") that agree to the terms and conditions stated in this License.

1. The USER may access, view, and use the DATASET without charge for lawful non-commercial research purposes only. Any commercial use, sale, or other monetization is prohibited. The USER may not use the DATASET for any unlawful activities, including but not limited to looting, vandalism, and disturbance of archaeological sites.
2. The USER may not attempt to identify the location of any part of the DATASET and must exercise all reasonable and prudent care to avoid the disclosure of the locations referenced in the DATASET in any publication or other communication.
3. The USER may not share access to the DATASET with anyone else. This includes distributing the download link or any portion of the DATASET. Other users must register separately and comply with all the terms of this License.
4. The USER must use the DATASET in a manner that respects the cultural heritage of Cambodia and its people, and in compliance with the relevant Cambodian authorities. Any use of the DATASET that could harm or exploit these cultural sites or their environment is strictly prohibited.
5. The USER must properly attribute the EFEO as the source of the data in any publications, presentations, or other forms of dissemination that make use of the DATASET.
6. This agreement may be terminated by either party at any time, but the USER's obligations with respect to the DATASET shall continue after termination. If the USER fails to comply with any of the above terms and conditions, their rights under this License shall terminate automatically and without notice.

THE DATASET IS PROVIDED "AS IS," AND THE EFEO DOES NOT MAKE ANY WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE EFEO OR ITS COLLABORATORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THE DATASET.

***

## Dataset splits

The dataset is split into training/validation/test sets in the following way:

- Training:

    ```
    01_olinda
    02_irene
    03_diomira
    04_isidora
    05_tamara
    06_dorothea
    07_anastasia
    08_isaura
    09_zora
    10_gold_road
    11_silver_road
    12_ginger_road
    13_saffron_road
    14_lavender_road
    15_thyme_road
    16_cinnamon_road
    ```
- Validation:

    ```
    17_fusang
    18_yaochi
    19_penglai
    ```
- Validation set:

    ```
    20_hali
    21_carcosa
    22_ulthar
    23_kadath
    ```

## Citation (BibTeX)

If you use this dataset in your research, please cite this publication:

```
@inproceedings{
perron2024archaeoscape,
title={Archaeoscape: Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era},
author={Yohann Perron and Vladyslav Sydorov and Adam P. Wijker and Damian Evans and Christophe Pottier and Loic Landrieu},
booktitle={NeurIPS},
year={2024},
doi={https://doi.org/10.48550/arXiv.2412.05203},
}
```
