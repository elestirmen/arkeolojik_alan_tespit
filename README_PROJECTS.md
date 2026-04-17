# Proje Yapisi

Bu repository artik tek bir ana proje olarak duzenlenmistir.

## Klasorler

```text
arkeolojik_alan_tespit/
├── archaeo_detect.py                  # Ana tespit akisi
├── training.py                        # Model egitimi
├── egitim_verisi_olusturma.py         # Eslestirilmis tile veri hazirligi
├── prepare_tile_classification_dataset.py
├── ground_truth_kare_etiketleme_qt.py
├── archeo_shared/                     # Ortak kanal/model yardimcilari
├── configs/                           # Ornek konfigler
├── docs/                              # Ek dokumantasyon
├── tests/                             # Testler
├── tools/                             # Yardimci inceleme/bakim scriptleri
└── workspace/                         # Repo-ici veri/model/cache/cikti alani
```

## Workspace

`workspace/` koddan ayrilmis, fakat repo icinde kalan calisma alanidir. Su tur dosyalar burada tutulur:

- ham ve turetilmis raster verileri
- egitim datasetleri
- checkpoint dosyalari
- derivative cache dosyalari
- inference ciktilari
- buyuk model veya indirme artefact'lari

Bu dizin `.gitignore` ile izlenmez; yalnizca [workspace/README.md](workspace/README.md) repoda kalir.
