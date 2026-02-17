#!/usr/bin/env python3
"""
VisDrone formatını YOLO formatına çeviren script.

VisDrone format:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

YOLO format:
    <class_id> <x_center> <y_center> <width> <height>
    (normalized 0-1)
"""

from pathlib import Path
import shutil
from tqdm import tqdm

# VisDrone class mapping (0-11)
VISDRONE_CLASSES = {
    0: "ignored",      # Ignored regions (kullanma)
    1: "pedestrian",   # Yaya
    2: "people",       # İnsan grubu
    3: "bicycle",      # Bisiklet
    4: "car",          # Araba
    5: "van",          # Minibüs
    6: "truck",        # Kamyon
    7: "tricycle",     # Üç tekerlekli
    8: "awning-tricycle", # Tenteli üç tekerlekli
    9: "bus",          # Otobüs
    10: "motor",       # Motorsiklet
    11: "others",      # Diğerleri
}

def convert_visdrone_to_yolo(
    visdrone_dir: Path,
    output_dir: Path,
    image_ext: str = ".jpg",
    ignore_classes: set = {0, 11}  # ignored ve others
):
    """
    VisDrone annotation'larını YOLO formatına çevir.
    
    Args:
        visdrone_dir: VisDrone veri seti kök dizini
        output_dir: YOLO formatında çıktı dizini
        image_ext: Görüntü uzantısı
        ignore_classes: Göz ardı edilecek sınıflar
    """
    
    # Dizinleri oluştur
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Train ve val setlerini işle
    for split in ["train", "val"]:
        print(f"\n{split.upper()} seti işleniyor...")
        
        # Annotation dizini
        anno_dir = visdrone_dir / "annotations" / split
        if not anno_dir.exists():
            anno_dir = visdrone_dir / split / "annotations"
        
        # Image dizini
        img_dir = visdrone_dir / "images" / split
        if not img_dir.exists():
            img_dir = visdrone_dir / split / "images"
        
        if not anno_dir.exists() or not img_dir.exists():
            print(f"  ⚠️ {split} dizinleri bulunamadı, atlanıyor...")
            continue
        
        # Tüm annotation dosyalarını işle
        anno_files = list(anno_dir.glob("*.txt"))
        
        for anno_file in tqdm(anno_files, desc=f"  Converting {split}"):
            # Görüntü dosyasını bul
            img_name = anno_file.stem + image_ext
            img_file = img_dir / img_name
            
            if not img_file.exists():
                continue
            
            # Görüntü boyutunu al
            try:
                from PIL import Image
                with Image.open(img_file) as img:
                    img_width, img_height = img.size
            except Exception:
                print(f"    ⚠️ Görüntü okunamadı: {img_file}")
                continue
            
            # Annotation'ları oku
            yolo_lines = []
            with open(anno_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    
                    bbox_left = int(parts[0])
                    bbox_top = int(parts[1])
                    bbox_width = int(parts[2])
                    bbox_height = int(parts[3])
                    # score = int(parts[4])  # Kullanılmıyor
                    class_id = int(parts[5])
                    # truncation = int(parts[6])
                    # occlusion = int(parts[7])
                    
                    # Geçersiz/göz ardı edilecek sınıfları atla
                    if class_id in ignore_classes or bbox_width <= 0 or bbox_height <= 0:
                        continue
                    
                    # YOLO formatına çevir (normalize edilmiş merkez + boyut)
                    x_center = (bbox_left + bbox_width / 2) / img_width
                    y_center = (bbox_top + bbox_height / 2) / img_height
                    norm_width = bbox_width / img_width
                    norm_height = bbox_height / img_height
                    
                    # Sınırları kontrol et [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    norm_width = max(0, min(1, norm_width))
                    norm_height = max(0, min(1, norm_height))
                    
                    # YOLO class ID (ignored ve others'ı çıkardık, yeniden mapla)
                    # VisDrone 1-10 -> YOLO 0-9
                    yolo_class_id = class_id - 1  # 1->0, 2->1, ..., 10->9
                    
                    yolo_lines.append(
                        f"{yolo_class_id} {x_center:.6f} {y_center:.6f} "
                        f"{norm_width:.6f} {norm_height:.6f}\n"
                    )
            
            # Boş annotation'ları atla
            if not yolo_lines:
                continue
            
            # YOLO label dosyasını yaz
            yolo_label_file = output_dir / "labels" / split / (anno_file.stem + ".txt")
            with open(yolo_label_file, 'w') as f:
                f.writelines(yolo_lines)
            
            # Görüntüyü kopyala
            yolo_img_file = output_dir / "images" / split / img_name
            shutil.copy(img_file, yolo_img_file)
    
    # data.yaml oluştur
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, 'w') as f:
        f.write("# VisDrone → YOLO dönüştürülmüş veri seti\n\n")
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/val  # Test seti yok, val kullan\n\n")
        f.write("# Sınıflar (VisDrone classes 1-10)\n")
        f.write("names:\n")
        # 0 ve 11'i çıkardık, kalan 1-10 -> 0-9
        for class_id in range(1, 11):
            yolo_id = class_id - 1
            class_name = VISDRONE_CLASSES[class_id]
            f.write(f"  {yolo_id}: {class_name}\n")
        f.write("\n# Toplam sınıf sayısı\n")
        f.write("nc: 10\n")
    
    print("\n✅ Dönüştürme tamamlandı!")
    print(f"   YOLO veri seti: {output_dir}")
    print(f"   data.yaml: {data_yaml}")
    print("\n🚀 Eğitim başlatmak için:")
    print(f"   yolo segment train data={data_yaml} model=yolo11s-seg.pt epochs=100")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VisDrone → YOLO format converter")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("VisDrone2019-DET"),
        help="VisDrone veri seti dizini"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visdrone_yolo"),
        help="YOLO çıktı dizini"
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        help="Görüntü dosya uzantısı"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VisDrone → YOLO Format Dönüştürücü")
    print("=" * 70)
    print(f"Girdi: {args.input}")
    print(f"Çıktı: {args.output}")
    print("=" * 70)
    
    convert_visdrone_to_yolo(
        visdrone_dir=args.input,
        output_dir=args.output,
        image_ext=args.image_ext
    )

