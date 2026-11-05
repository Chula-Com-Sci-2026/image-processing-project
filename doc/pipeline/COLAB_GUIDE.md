# คู่มือสร้าง dataset 3 (Grounded-SAM) บน Google Colab

คู่มือนี้รัน step `gsam` ของ `pipeline_02.py` บน GPU ของ Colab เพื่อสร้าง `03_denoise_clahe_gsam` จาก dataset 2 แล้วเก็บผลไว้บน Google Drive

- **ส่วนที่ 1–5:** สำหรับ data engineer ที่สร้าง dataset 3
- **ส่วนที่ 6:** สำหรับเพื่อนในทีมที่นำ dataset ไปเทรนโมเดล

ทำไมต้องใช้ Colab: step `gsam` รัน Grounding DINO และ SAM ต่อภาพ บน CPU ใช้เวลาประมาณ 20 วินาทีต่อภาพ (ทดสอบแล้ว) ถ้าทำ 9,630 ภาพจะใช้เวลาหลายวัน ส่วน GTX 1050 ในเครื่องรุ่นเก่าเกินไปสำหรับ PyTorch ปัจจุบัน

## หลักการที่ต้องรู้ก่อนเริ่ม

| ที่เก็บ | มีอะไร | อยู่นานแค่ไหน |
|---|---|---|
| `/content/repo/` (เครื่อง Colab) | โค้ดจาก GitHub, `manifest.csv` และภาพที่ unzip หรือสร้างขึ้น | หายเมื่อ session จบหรือหลุด |
| `dip-project/dataset_zips/` (Drive) | zip ของแต่ละ dataset, `gsam_stats.csv`, `gsam_boxes.json` | ถาวร |
| `dip-project/dataset_zips/checkpoints_gsam/` (Drive) | ความคืบหน้าของ step `gsam` ทุก 500 ภาพ | ถาวร จนกว่าจะลบ |

- **อย่าอ่านหรือเขียนภาพบน Drive โดยตรง** เพราะไฟล์เล็กจำนวนมากจะช้ามาก ให้ทำงานใน `/content` แล้วเก็บผลเป็น zip
- **step `gsam` เซฟ checkpoint ขึ้น Drive เอง** ถ้า session หลุด จะเสียงานไม่เกิน 500 ภาพ
- **ใช้โฟลเดอร์ `dataset_zips/` ไม่ใช่ `data_zips/` ของ process เก่า** checkpoint ของ process เก่าใช้ชื่อไฟล์ `gsam_00000.zip` เหมือนกัน ถ้าชี้ไปโฟลเดอร์เก่า pipeline จะดึงภาพของ dataset เก่ามาใช้แทน

## ส่วนที่ 1: เตรียมไฟล์บนเครื่องตัวเอง

### 1. Push โค้ดล่าสุดขึ้น GitHub

Colab จะ clone repo จาก `main` ต้องมี step `gsam` ใน `pipeline_02.py` และ `data/dataset/meta/manifest.csv` อยู่แล้ว

### 2. อัปโหลด zip ขึ้น Drive

zip ถูกสร้างไว้แล้วที่ `data/dataset/zips/` ในเครื่อง

| ไฟล์ | ขนาด | ใช้ทำอะไร |
|---|---|---|
| `02_denoise_clahe.zip` | 393 MB | input ของ step `gsam` (**ต้องมี**) |
| `01_original.zip` | 332 MB | ให้เพื่อนในทีมใช้ ไม่ได้ใช้ใน Colab |

1. สร้างโฟลเดอร์ `dataset_zips` ใน `MyDrive/2110651-digital-images-processing/2110651-digital-images-project/workspace/dip-project/`
2. อัปโหลดทั้งสองไฟล์ลงในโฟลเดอร์นั้น

ถ้าต้องการสร้าง zip ใหม่ (เช่น หลัง rebuild dataset) ให้รันคำสั่งนี้ที่ root ของ repo:

```bash
.venv/bin/python -c "
import zipfile, csv; from pathlib import Path
D = Path('data/dataset'); rels = [r['relpath'] for r in csv.DictReader(open(D / 'meta/manifest.csv'))]
for name in ['01_original', '02_denoise_clahe']:
    with zipfile.ZipFile(D / 'zips' / f'{name}.zip', 'w', zipfile.ZIP_STORED) as z:
        for r in rels: z.write(D / name / r, f'{name}/{r}')"
```

## ส่วนที่ 2: สร้าง notebook

### 3. สร้างไฟล์

สร้าง `pipeline_02_gsam_colab.ipynb` ไว้ในโฟลเดอร์ `dip-project`

### 4. เลือก GPU

ไปที่ **Runtime → Change runtime type → T4 GPU → Save**

ไม่ต้องใช้ Kaggle token แล้ว เพราะ input มาจาก zip บน Drive

## ส่วนที่ 3: วางโค้ดทีละ cell แล้วรันตามลำดับ

### Cell 1: setup

ระบบจะถามสิทธิ์เข้าถึง Drive ให้กดอนุญาต

```python
from google.colab import drive
import os
drive.mount("/content/drive")
SAVE = "/content/drive/MyDrive/2110651-digital-images-processing/2110651-digital-images-project/workspace/dip-project/dataset_zips"
DATA = "/content/repo/data/dataset"
os.environ.update(FOOD_DATA_DIR="/content/repo/data",
                  FOOD_CHECKPOINT_DIR=f"{SAVE}/checkpoints_gsam")
!nvidia-smi -L
!ls -lh {SAVE}
```

**เช็ก:**
- ต้องเห็นชื่อ GPU ถ้าไม่เห็น ให้กลับไปทำข้อ 4
- ต้องเห็น `02_denoise_clahe.zip` ใน `SAVE` ถ้าไม่เห็น ให้กลับไปทำข้อ 2

### Cell 2: โหลดโค้ดและ dataset 2

```python
!git clone -q https://github.com/Chula-Com-Sci-2026/image-processing-project.git /content/repo
%cd /content/repo
!sha256sum data/dataset/meta/manifest.csv
!unzip -q -o {SAVE}/02_denoise_clahe.zip -d {DATA}
!find {DATA}/02_denoise_clahe -name "*.jpg" | wc -l
!pip freeze > {SAVE}/pip_freeze_gsam.txt
```

**เช็ก:**
- hash ต้องเป็น `79eb1d1cf8a475b8deaa2a301acc64a1c3a334fd4a289c8d45af91c359c3c2a4`
- จำนวนภาพต้องเป็น **9630**

### Cell 3: รัน Grounded-SAM

```python
!python pipeline_preprocess/pipeline_02.py --steps gsam
```

- log `[gsam] n/9630` ขึ้นทุก 500 ภาพ ใช้ประมาณเวลาที่เหลือได้
- เมื่อจบจะขึ้นบรรทัด `[gsam] done: 9630 images, fallback rate ...` **จดตัวเลข fallback rate ไว้** เพราะต้องใช้ในรายงาน
- **ถ้า session หลุด:** ทำตาม [ส่วนที่ 5](#ส่วนที่-5-ถ้า-colab-หลุดกลางทาง)

### Cell 4: เก็บผลขึ้น Drive

รันทันทีที่ Cell 3 จบ ไม่อย่างนั้นถ้า session หลุด ภาพจะหาย

```python
!cd {DATA} && zip -0 -qr /content/03_denoise_clahe_gsam.zip 03_denoise_clahe_gsam   # -0: JPEG is already compressed
!cp /content/03_denoise_clahe_gsam.zip {DATA}/meta/gsam_stats.csv {DATA}/meta/gsam_boxes.json {SAVE}/
!ls -lh {SAVE}
```

**เช็ก:** ต้องเห็น `03_denoise_clahe_gsam.zip`, `gsam_stats.csv` และ `gsam_boxes.json` ใน `SAVE`

## ส่วนที่ 4: ตรวจผลและปิดงาน

### 5. ตรวจ fallback rate แยกตาม split

```python
import pandas as pd
st = pd.read_csv(f"{DATA}/meta/gsam_stats.csv")
print(st.groupby("split").fallback.mean().mul(100).round(1))
print(st.mask_area.describe().round(3))
```

ค่า fallback ของ train, val และ test ควรใกล้เคียงกัน ถ้า split ไหนสูงกว่าที่อื่นมาก ให้บันทึกไว้ในรายงาน

### 6. ปิด session

ไปที่ **Runtime → Disconnect and delete runtime** เพื่อไม่ให้เสียโควตา GPU

### 7. นำผลกลับมาที่เครื่องตัวเองแล้ว commit

1. โหลด `gsam_stats.csv` และ `gsam_boxes.json` จาก `dataset_zips/` บน Drive มาไว้ที่ `data/dataset/meta/`
2. โหลด `03_denoise_clahe_gsam.zip` แล้วแตกที่ `data/dataset/` จะได้ `data/dataset/03_denoise_clahe_gsam/`
3. Commit `gsam_stats.csv` และ `gsam_boxes.json` (git ignore ภาพไว้อยู่แล้ว)
4. บันทึก fallback rate ลงใน section 4 ของ `doc/pipeline/pipeline_02.md`

### 8. ลบ checkpoint

เมื่อมีไฟล์ใน Drive ครบแล้ว ลบโฟลเดอร์ `dataset_zips/checkpoints_gsam/` ได้ ถ้าจะรันใหม่ด้วย prompt, threshold หรือโมเดลอื่น **ต้องลบก่อน** ไม่อย่างนั้น pipeline จะนำผลเก่ากลับมาใช้

## ส่วนที่ 5: ถ้า Colab หลุดกลางทาง

1. เปิด notebook ใหม่ เลือก T4 GPU (ข้อ 4)
2. รัน Cell 1 และ Cell 2 อีกครั้ง (`/content` ว่างทุกครั้งที่เปิด session ใหม่)
3. รัน Cell 3 อีกครั้ง log จะขึ้น `[gsam] resumed at n/9630` แล้วทำต่อจาก checkpoint ล่าสุด
4. รัน Cell 4 เมื่อจบ

ทดสอบแล้วบนเครื่องตัวเอง (2026-11-05): หยุดกลางทางหลังภาพที่ 4 แล้วรันใหม่ pipeline ดึงภาพที่ทำแล้วจาก checkpoint และทำต่อจนครบ

## ส่วนที่ 6: สำหรับเพื่อนในทีม

### 1. เอา dataset มาใช้

1. `git pull` repo จะได้ `data/dataset/meta/` (`manifest.csv`, `gsam_stats.csv` และอื่น ๆ)
2. โหลด zip ของ dataset ที่ต้องการจาก `dip-project/dataset_zips/` บน Drive แล้วแตกที่ `data/dataset/`

| dataset | zip |
|---|---|
| 1 | `01_original.zip` |
| 2 | `02_denoise_clahe.zip` |
| 3 | `03_denoise_clahe_gsam.zip` |

3. ตรวจด้วย `sha256sum data/dataset/meta/manifest.csv` ต้องได้ `79eb1d1c…` (ค่าเต็มอยู่ใน Cell 2)

บน Colab ให้แตก zip ลงใน `/content` แทนการอ่านจาก Drive โดยตรง เช่น `!unzip -q -o <path ใน Drive>/02_denoise_clahe.zip -d /content/data/dataset`

### 2. path บน Drive

| คนใช้ | path ของ `dataset_zips` |
|---|---|
| เพื่อนในทีม (ผ่าน shortcut ของ `dip-project`) | `/content/drive/MyDrive/dip-project/dataset_zips` |
| เจ้าของโฟลเดอร์ | `/content/drive/MyDrive/2110651-digital-images-processing/2110651-digital-images-project/workspace/dip-project/dataset_zips` |

### 3. กติกา

- ใช้ train/val/test ตามโฟลเดอร์ ห้ามแบ่งใหม่
- ภายในโมเดลเดียวกัน ทุก dataset ต้องใช้ค่าตั้งเดียวกัน
- เลือก epoch และ hyper-parameter ด้วย val macro-F1 แล้วประเมิน test ครั้งเดียว

## แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `nvidia-smi` ไม่เจอ GPU | เลือก T4 GPU ในข้อ 4 แล้วรัน Cell 1 ใหม่ |
| `[gsam] no CUDA GPU` | เหมือนข้อบน pipeline ข้าม step นี้โดยไม่ขึ้น error |
| `[gsam] STOP: ... dataset-2 images missing` | unzip ใน Cell 2 ไม่ครบ ให้รัน Cell 2 ใหม่ แล้วเช็กว่าได้ 9630 ภาพ |
| `[gsam] gsam_stats.csv exists, skipping` | step นี้จบไปแล้ว ถ้าจะรันใหม่ ให้ลบ `{DATA}/meta/gsam_stats.csv` และโฟลเดอร์ checkpoint บน Drive |
| hash ของ manifest ไม่ตรง | clone repo ผิด branch หรือ repo ยังไม่ได้ pull commit ล่าสุด ให้ลบ `/content/repo` แล้วรัน Cell 2 ใหม่ |
| `ModuleNotFoundError` หรือหา `pipeline_preprocess` ไม่เจอ | รัน `%cd /content/repo` แล้วลองใหม่ |
| `resumed at` ขึ้นเลขแปลก ๆ หรือได้ภาพจาก dataset เก่า | `FOOD_CHECKPOINT_DIR` ชี้ไปโฟลเดอร์เก่า ให้เช็ก Cell 1 ว่าใช้ `dataset_zips/checkpoints_gsam` |
