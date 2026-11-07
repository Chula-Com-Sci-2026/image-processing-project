# dataset_zips: ข้อมูลภาพสำหรับเทรนโมเดล

dataset ชุดใหม่ (เริ่ม 2026-11-05) ใช้แทน `data_zips/` เดิมทั้งหมด **อย่าใช้ zip ใน `data_zips/` อีก** เพราะเป็นคลาสและ split ของ process เก่า

มี 4 dataset ทุก zip มีภาพชุดเดียวกัน 9,630 ภาพ ใน split เดียวกัน และชื่อไฟล์ตรงกันทุกชุด ต่างกันแค่ preprocessing

## ข้อมูลโดยย่อ

- **10 คลาส** (คลาสที่ภาพมืดที่สุดใน Food-101): bibimbap, miso_soup, mussels, prime_rib, edamame, baby_back_ribs, paella, chicken_wings, guacamole, nachos
- **คลาสละ 963 ภาพ** แบ่ง 60 / 20 / 20

| split | ต่อคลาส | รวม |
|---|---:|---:|
| train | 578 | 5,780 |
| val | 193 | 1,930 |
| test | 192 | 1,920 |

- ตัด noise ออกแล้ว 118 ภาพ (label ผิด, ไม่ใช่อาหาร, ภาพซ้ำ) รายการอยู่ใน `meta/removed.csv`
- ภาพเก็บที่ด้านสั้น 256 px, JPEG quality 95

## ต้องใช้ไฟล์ไหน

| dataset | ไฟล์ | ขนาด | preprocessing |
|---|---|---:|---|
| 1 | `01_original.zip` | 333 MB | ภาพต้นฉบับ (baseline) |
| 2 | `02_denoise_clahe.zip` | 393 MB | dataset 1 + ลด noise ด้วย wavelet แล้วทำ CLAHE |
| 3 | `03_denoise_clahe_gsam.zip` | 254 MB | dataset 2 + Grounded-SAM: เก็บเฉพาะอาหาร พื้นหลังสีเทา crop ให้เหลือแต่อาหาร |
| 4 | `04_denoise_clahe_dino.zip` | 346 MB | dataset 2 + crop ตามกรอบอาหารของ Grounding DINO เก็บพิกเซลจริงไว้ ไม่ใช้ mask |

ต้องเทรนครบทั้ง 4 dataset ในทุกโมเดล คำถามวิจัยตอบด้วยการเทียบ 2 กับ 3 ส่วน dataset 4 ใช้แยกว่าผลของ segmentation มาจากการ crop หรือจาก mask

## แต่ละ zip มีอะไร

ไม่ต้องใช้ git ทุกอย่างที่ต้องใช้อยู่ใน zip แล้ว

```text
README_<dataset>.md     อธิบาย dataset, จำนวนภาพ, hash, กติกา
meta/manifest.csv       relpath, label, split, source ของทุกภาพ
meta/class_counts.csv   จำนวนภาพที่เหลือต่อคลาสหลังตัด noise
meta/removed.csv        ภาพที่ตัดออก พร้อมเหตุผล
meta/gsam_stats.csv     (เฉพาะ dataset 3) mask area และ fallback ของแต่ละภาพ
meta/dino_crop_stats.csv (เฉพาะ dataset 4) ขนาดกรอบและ fallback ของแต่ละภาพ
<dataset>/{train,val,test}/<class>/<id>.jpg
```

## ถ้าใช้ในเครื่อง

โหลด zip ที่ต้องการ แล้วแตก**ทุก zip ไว้ที่โฟลเดอร์เดียวกัน** `data/dataset/` (ไฟล์ใน `meta/` ของทุก zip เหมือนกัน เขียนทับได้) จะได้

```text
data/dataset/
├── meta/
├── 01_original/{train,val,test}/<class>/<id>.jpg
├── 02_denoise_clahe/
├── 03_denoise_clahe_gsam/
└── 04_denoise_clahe_dino/
```

อ่านภาพด้วย `torchvision.datasets.ImageFolder("data/dataset/01_original/train")` ได้เลย ชื่อโฟลเดอร์คือชื่อคลาส

## ถ้าเทรนบน Colab: ไม่ต้องโหลดลงเครื่อง

แตก zip จาก Drive ลง `/content` แล้วอ่านภาพจาก `/content` (อ่านภาพจาก Drive ตรง ๆ จะช้ามาก)

```python
from google.colab import drive
drive.mount("/content/drive")
ZIPS = "/content/drive/MyDrive/dip-project/dataset_zips"   # ดูตารางด้านล่าง
DATASET = "02_denoise_clahe"                                # เปลี่ยนแค่บรรทัดนี้
!unzip -q -o {ZIPS}/{DATASET}.zip -d /content/data/dataset
DATA_ROOT = f"/content/data/dataset/{DATASET}"              # มี train/ val/ test/
```

| คนใช้ | `ZIPS` |
|---|---|
| เพื่อนในทีม (ผ่าน shortcut ของ `dip-project`) | `/content/drive/MyDrive/dip-project/dataset_zips` |
| เจ้าของโฟลเดอร์ | `/content/drive/MyDrive/2110651-digital-images-processing/2110651-digital-images-project/workspace/dip-project/dataset_zips` |

ถ้ายังไม่มี shortcut: เปิดโฟลเดอร์ `dip-project` ที่แชร์ไว้ แล้วกด **Organize → Add shortcut** ไปที่ My Drive

## ตรวจว่าได้ข้อมูลชุดเดียวกับทุกคน

```text
sha256sum data/dataset/meta/manifest.csv
79eb1d1cf8a475b8deaa2a301acc64a1c3a334fd4a289c8d45af91c359c3c2a4
```

บน Colab ใช้ `!sha256sum /content/data/dataset/meta/manifest.csv` ถ้าได้ค่าไม่ตรง ให้โหลด zip ใหม่

รายละเอียดทั้งหมดอยู่ใน repo: `doc/pipeline/pipeline_02.md`
