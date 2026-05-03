# LM Studio VLM Taramasi

Bu entegrasyon, LM Studio uzerinden OpenAI uyumlu bir vision-language model ile
GeoTIFF tile taramasi yapar. VLM sonucu mevcut DL, classic, YOLO veya fusion
maskelerine karistirilmaz; ayri aday dosyalari olarak yazilir.

## LM Studio Hazirligi

1. LM Studio'da vision/multimodal destekli bir model indirin ve yukleyin.
   Onerilen baslangic modeli: vision destekli bir Qwen/Gemma/LLaVA turevi model.
2. LM Studio icinde **Local Server** bolumunu acin.
3. Server'i baslatin ve OpenAI compatible endpoint'in su adreste calistigini
   kontrol edin:

```text
http://localhost:1234/v1
```

Model vision desteklemiyorsa VLM taramasi goruntu girdisi hatasi kaydeder ve
kullaniciya log ile bildirir.

## Ornek Config

`config.local.yaml` varsa `python archaeo_detect.py` onu kullanir; yoksa
`config.yaml` kullanilir. VLM varsayilan olarak kapali gelir.

```yaml
enable_vlm: true
vlm_base_url: "http://localhost:1234/v1"
vlm_api_key: "lm-studio"
vlm_model: "auto"
vlm_tile: 1024
vlm_overlap: 256
vlm_views: "auto"
vlm_gsd_m: 0.30
vlm_confidence_threshold: 0.60
vlm_max_tiles: 30
vlm_export_every: 50
vlm_resume: true
vlm_timeout: 120
vlm_temperature: 0.0
```

Kucuk deneme icin `vlm_max_tiles: 30` kullanin. Tum raster icin `0` sinirsiz
anlamina gelir.

## Ornek CLI

```bash
python archaeo_detect.py \
  --enable-vlm \
  --vlm-model auto \
  --vlm-views auto \
  --vlm-max-tiles 30 \
  --vlm-export-every 50 \
  --vlm-resume
```

CLI argumanlari YAML degerlerini override eder. CLI argumani vermek zorunlu
degildir; YAML icinde `enable_vlm: true` yeterlidir.

`vlm_model: "auto"` LM Studio `/v1/models` listesinden o an yuklu modeli alir.
Tek model yukluyse dogrudan onu kullanir. Birden fazla model yukluyse ilk modeli
secer ve log'a uyari yazar. Belirli bir modele sabitlemek icin model adini acik
yazin:

```yaml
vlm_model: "qwen2.5-vl-7b-instruct"
```

LM Studio arayuzundeki adres `http://127.0.0.1:8081` gibi `/v1` olmadan
gorunuyorsa config'e bu adresi yazabilirsiniz; entegrasyon OpenAI base URL icin
gerekirse `/v1` ekler.

## `vlm_views: auto`

Auto mod rasterda kullanilabilir bantlara gore view secer:

| Bant yapisi | Kullanilan view'lar |
| --- | --- |
| Sadece RGB | `rgb` |
| RGB + DSM + DTM | `rgb`, `hillshade`, `ndsm`, `slope` |
| RGB + DTM | `rgb`, `hillshade`, `slope` |
| RGB + DSM | `rgb`, `dsm` |

Manuel view listesi de verilebilir:

```yaml
vlm_views: "rgb,hillshade,ndsm,slope"
```

Eksik bant gerektiren view'lar warning log ile atlanir. VLM icin RGB zorunludur;
RGB yoksa tarama aciklayici hata verir.

Not: 4 bantli RGB+DTM dosyalarinda band aciklamasi DTM/terrain olarak
etiketlenmemisse `bands: "1,2,3,0,4"` kullanarak dorduncu bandin DTM oldugunu
belirtin. 4 bantli RGB+DSM icin varsayilan `bands: "1,2,3,4,0"` veya mevcut
`1,2,3,4,5` ayari DSM bandini kullanir.

## RGB-only ve Topografik Davranis

RGB-only rasterlarda prompt modele bunun yalnizca ortofoto oldugunu soyler.
Model renk farki, bitki izi, toprak izi, duzenli geometri, yapi lekesi ve
yuzey anomalileri uzerinden yorum yapar; mikro-topografik yorumlarda temkinli
olmasi istenir.

RGB + topo turevleri olan rasterlarda prompt hem yuzey izlerini hem de
mikro-topografik anomalileri degerlendirmesini ister. Hoyuk, tumulus, hendek,
sur, teras, temel izi, eski yol izi ve duzenli topo anomalileri ozellikle aranir.

Onerilen baslangic ayari:

```yaml
vlm_views: "auto"
```

## Ciktilar

VLM ciktilari mevcut session output klasorune ayri dosyalar olarak yazilir:

```text
*_vlm_candidates.jsonl
*_vlm_candidates.csv
*_vlm_candidates.xlsx
*_vlm_candidates.geojson
*_vlm_candidates.gpkg
```

Excel listesi, CSV ile ayni aday satirlarini `vlm_candidates` sayfasinda filtreli
tablo olarak verir. Excel/CSV icinde raster CRS merkezi (`center_x`, `center_y`),
WGS84 koordinatlar (`gps_lon`, `gps_lat`) ve dogrudan acilabilir
`google_maps_url` kolonu bulunur.

GPKG icinde genel `vlm_candidates` katmani yazilir. Ayrica bulunan aday tipleri
icin tur bazli ek katmanlar olusturulur; ornegin `vlm_mound`,
`vlm_tumulus`, `vlm_ring_ditch`, `vlm_wall_trace`.

Bozuk JSON veya API hatalari tile bazinda JSONL hata kaydi olarak tutulur; tum
pipeline bu yuzden dusmez. Bozuk ham cevaplar ayrica
`*_vlm_raw_errors.jsonl` dosyasina yazilir.

Uzun calismalarda `vlm_export_every: 50` ile her 50 yeni tile sonunda Excel,
CSV, GeoJSON ve GPKG ara ciktilari guncellenir. JSONL her tile'dan hemen sonra
flush edilir; bilgisayar kapanirsa o ana kadar yazilan tile kayitlari korunur.

`vlm_resume: true` acikken yeni calisma basladiginda onceki session
klasorlerinde ayni cikti adina sahip `*_vlm_candidates.jsonl` aranir. Bulunursa
islenmis tile'lar okunur ve ayni tile/overlap/view planiyla uyusanlar atlanarak
tarama kaldigi yerden devam eder. Tile boyutu, overlap veya view ayarlari
degistiyse eski kayitlar guvenli tarafta kalmak icin kullanilmaz.

`vlm_confidence_threshold` Excel/CSV/GPKG/GeoJSON'a yazilacak aday esigidir.
Modelin aday dedigi tum tile cevaplari JSONL icinde kalir; tablo ve GIS
ciktilarina yalnizca bu esigi gecen, geometriye sahip adaylar aktarilir.

`vlm_gsd_m` prompt'a metre/piksel olcegini ekler. Ornegin `0.30` degeri modele
girdinin yaklasik 30 cm GSD'li nadir ortofoto oldugunu soyler; 1024 piksel tile
yaklasik 307 m x 307 m alan kaplar. `0` veya `null` verilirse raster transform
metre birimindeyse GSD otomatik tahmin edilmeye calisilir.

## Onemli Uyari

`bbox_xyxy` tile piksel koordinatindadir. Cikti ayrica global piksel ve raster
CRS koordinatlarina cevrilir. Bu bbox yaklasiktir; nihai arkeolojik sinir olarak
kullanilmamalidir. Uzman GIS incelemesi ve arazi kontrolu gereklidir.
