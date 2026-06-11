# LM Studio / llama-server VLM Taramasi

Bu entegrasyon, LM Studio veya standalone llama.cpp llama-server uzerinden OpenAI uyumlu
bir vision-language model ile
GeoTIFF tile taramasi yapar. VLM sonucu mevcut DL, classic, YOLO veya fusion
maskelerine karistirilmaz; ayri aday dosyalari olarak yazilir.

LM Studio ve standalone llama-server'i ayni anda aktif tutmayin. Ikisi de ayni
GPU/VRAM icin yaristiginda hiz ve stabilite bozulabilir. Hangi backend'in
kullanilacagini `config_vlm.yaml` icindeki `backend: lmstudio` veya
`backend: llama` satiri belirler. Secilen backend'i gercekten acmak/kapatmak
icin tek giris noktasi:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/use_vlm_backend.ps1 -Backend status
powershell -ExecutionPolicy Bypass -File scripts/use_vlm_backend.ps1 -Backend lmstudio
powershell -ExecutionPolicy Bypass -File scripts/use_vlm_backend.ps1 -Backend llama
```

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

## VLM-only Config

`vlm_detect.py` IDE'den argumansiz calistirilirsa once `config_vlm.local.yaml`,
yoksa `config_vlm.yaml` dosyasini okur. Bu repo icin onerilen VLM-only akista
backend secimi tek dosyadadir:

```yaml
backend: lmstudio   # lmstudio | llama

backends:
  lmstudio:
    base_url: "http://127.0.0.1:1234"
    api_key: "lm-studio"
    tile: 1024
    overlap: 256
  llama:
    base_url: "http://127.0.0.1:18080"
    api_key: "llama-server"
    image_tokens: 1120
    auto_start_backend: true
    backend_startup_timeout_seconds: 180
    tile: 768
    overlap: 192
```

Kucuk deneme icin `max_tiles: 30` kullanin. Tum raster icin `0` sinirsiz
anlamina gelir. `--backend llama` CLI argumani config'teki `backend` secimini
gecici olarak ezebilir.

## Ornek CLI

```bash
python vlm_detect.py --config config_vlm.yaml --max-tiles 30
```

CLI argumanlari YAML degerlerini override eder. CLI argumani vermek zorunlu
degildir; YAML icinde `backend` ve ilgili profil yeterlidir.

`model: "auto"` OpenAI uyumlu `/v1/models` listesinden uygun modeli alir.
Tek model yukluyse dogrudan onu kullanir. Birden fazla model yukluyse ilk modeli
secer ve log'a uyari yazar. Belirli bir modele sabitlemek icin model adini acik
yazin:

```yaml
backends:
  lmstudio:
    model: "qwen2.5-vl-7b-instruct"
```

LM Studio arayuzundeki adres `http://127.0.0.1:8081` gibi `/v1` olmadan
gorunuyorsa config'e bu adresi yazabilirsiniz; entegrasyon OpenAI base URL icin
gerekirse `/v1` ekler.

## llama.cpp llama-server alternatifi

llama-server kullanmak icin once CUDA destekli llama.cpp paketini hazirlayin.
Windows NVIDIA icin resmi release paketlerinde Windows x64 CUDA 13 ve CUDA
13.3 DLL paketleri bulunur. Kurulumdan sonra su kontrol kritik:

```powershell
cd C:\llama
.\llama-server.exe --help | findstr image
```

Yardim ciktisinda `--image-min-tokens` ve `--image-max-tokens` gorunmelidir.
Bu repo icin hazir baslatma scripti:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_llama_server_gemma4.ps1
```

Script varsayilan olarak LM Studio model klasorundeki Gemma 4 GGUF ve mmproj
dosyalarina isaret eder; gerekirse `-ModelPath` ve `-MmprojPath` ile ezebilirsiniz.
`use_vlm_backend.ps1 -Backend llama`, `config_vlm.yaml` icindeki
`backends.llama.image_tokens` degerini okuyup baslatma scriptine `-ImageTokens`
olarak gecirir. llama-server'i `http://127.0.0.1:18080/v1` adresinde baslatir
ve goruntu token sayisini su flag'lere yazar:

```text
--image-min-tokens 1120
--image-max-tokens 1120
```

`image_tokens` degerini `1536` veya `2048` gibi daha yuksek bir degere almak
daha ayrintili gorsel tokenization ister; view sayisi, context ve VRAM maliyeti
artar. `auto_start_backend: true` iken `vlm_detect.py`, `backend: llama`
seciliyse ve `http://127.0.0.1:18080/v1/models` kapaliysa llama-server'i ayri
bir PowerShell penceresinde otomatik baslatir. Sonra VLM taramasini calistirin:

```powershell
python vlm_detect.py --config config_vlm.yaml
```

Not: LM Studio'da model yukluyse VRAM dolu olabilir. llama-server'i baslatmadan
once LM Studio modelini unload etmek veya LM Studio'yu kapatmak gerekebilir.
`config_vlm.yaml` icindeki `backends.llama.reload_every_tiles: 0` tutulur;
periyodik unload/load yalnizca LM Studio native API icin desteklenir.

## `vlm_views: auto`

Auto mod rasterda kullanilabilir bantlara gore view secer:

| Bant yapisi | Kullanilan view'lar |
| --- | --- |
| Sadece RGB | `rgb` |
| RGB + DSM + DTM | `rgb`, `hillshade`, `ndsm`, `slope` |
| RGB + DTM | `rgb`, `hillshade`, `slope` |
| RGB + DSM | `rgb`, `dsm` |
| Klasor: RGB + hillshade/SLRM/SVF/slope | `rgb` + bulunan topo turevleri |
| Klasor: sadece hillshade/SLRM/SVF/slope | tek bant referans + bulunan topo turevleri |

Manuel view listesi de verilebilir:

```yaml
vlm_views: "rgb,hillshade,ndsm,slope"
```

Eksik bant gerektiren view'lar warning log ile atlanir. VLM icin RGB zorunludur;
RGB yoksa tarama aciklayici hata verir.

`input` bir GeoTIFF dosyasi yerine klasor de olabilir. Bu durumda kod klasordeki
`.tif/.tiff` dosyalarini tarar; RGB/ortho goruntu varsa referans raster olarak
onu secer, `hillshade`, `slrm`, `svf`/`svm`, `slope`, `dsm`, `dtm` adlarini
dosya adi veya band aciklamasindan yakalayip ayni tile icinde ayri VLM view'lari
olarak gonderir. Ek rasterlar referans grid ile birebir ayni degilse rasterio
WarpedVRT ile referans CRS/transform/extent uzerine hizalanir.

Ornek:

```yaml
input: "workspace/on_veri/karlik_vadi_set/topo_haritalar"
vlm_views: "auto"
vlm_source_kind: "auto"
```

Klasor icinde birden fazla ayni tip view varsa ilk bulunan kullanilir ve digeri
warning ile atlanir. En iyi sonuc icin dosya adlarini acik tutun:
`alan_rgb.tif`, `alan_hillshade.tif`, `alan_slrm.tif`, `alan_svf.tif`,
`alan_slope.tif`.

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
