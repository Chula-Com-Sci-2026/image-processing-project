# คู่มือรัน pipeline_01 บน Google Colab

คู่มือนี้สร้างข้อมูลทุก variant บน Colab แล้วเก็บไว้บน Google Drive ให้ทั้งทีมใช้ร่วมกัน

- **ส่วนที่ 1–5:** สำหรับ data engineer ที่รัน pipeline ถ้า `original_data.zip` และ `manifest.csv` อยู่บน Drive แล้ว ให้เริ่มจากข้อ 4–5, Cell 1 และส่วนที่ 5 แล้วไป Cell 5
- **ส่วนที่ 6:** สำหรับ data scientist ที่นำข้อมูลไปเทรนโมเดล

## หลักการที่ต้องรู้ก่อนเริ่ม

| ที่เก็บ | มีอะไร | อยู่นานแค่ไหน |
|---|---|---|
| `/content/data/` (เครื่อง Colab) | ภาพที่ pipeline สร้างหรือ unzip มา | หายเมื่อ session จบหรือหลุด |
| `dip-project/data_zips/` (Drive) | zip ราย variant, `manifest.csv`, `boxes.json`, `*_stats.csv` | ถาวร |
| `dip-project/data_zips/checkpoints/` (Drive) | ความคืบหน้าของ step segment ทุก 500 รูป | ถาวร จนกว่าจะลบ |

- **อย่าเขียนหรืออ่านภาพจาก Drive ตรง ๆ** เพราะไฟล์เล็กจำนวนมากจะช้ามาก ให้ทำงานบน `/content` แล้วเก็บเป็น zip
- **รัน `save(...)` ทันทีที่แต่ละ step จบ** ไม่อย่างนั้นถ้า session หลุด ภาพชุดนั้นจะหาย
- **Step segment เซฟ checkpoint ขึ้น Drive เอง** ถ้าหลุดจะเสียงานไม่เกิน 500 รูป

## ส่วนที่ 1: เตรียมก่อนเปิด notebook

### 1. ขอ Kaggle API token

1. ไปที่ kaggle.com แล้วกดรูปโปรไฟล์ → **Settings** → หัวข้อ **API**
2. สร้าง token แล้วคัดลอกค่าที่ได้ (มักขึ้นต้นด้วย `KGAT_`)

ค่านี้ใช้ค่าเดียว ไม่ต้องมี username อย่าวางลงใน notebook หรือไฟล์ที่แชร์กัน

### 2. เตรียมโฟลเดอร์บน Drive

1. โฟลเดอร์งานอยู่ที่ `MyDrive/2110651-digital-images-processing/2110651-digital-images-project/workspace/dip-project/`
2. แชร์โฟลเดอร์ `dip-project` ให้เพื่อนในทีม
3. ให้เพื่อนกด **Organize → Add shortcut** ไปที่ My Drive ของตัวเองโดยตรง

## ส่วนที่ 2: สร้าง notebook

### 3. สร้างไฟล์

สร้าง `pipeline_01_colab.ipynb` ไว้ในโฟลเดอร์ `dip-project`

### 4. เลือก GPU

ไปที่ **Runtime → Change runtime type → T4 GPU → Save**

### 5. ใส่ token ใน Colab Secrets

1. กดไอคอนกุญแจ (Secrets) ที่แถบซ้าย
2. เพิ่ม `KAGGLE_API_TOKEN` แล้ววางค่า token
3. เปิดสวิตช์ให้ notebook เข้าถึง

## ส่วนที่ 3: วางโค้ดทีละ cell แล้วรันตามลำดับ

### Cell 1: setup

ระบบจะถามสิทธิ์เข้าถึง Drive ให้กดอนุญาต

```python
from google.colab import drive, userdata
import os
drive.mount("/content/drive")
SAVE = "/content/drive/MyDrive/2110651-digital-images-processing/2110651-digital-images-project/workspace/dip-project/data_zips"
OUT = "/content/data/preprocessing_data"
os.makedirs(SAVE, exist_ok=True)
os.environ.update(FOOD_DATA_DIR="/content/data",
                  FOOD_CHECKPOINT_DIR=f"{SAVE}/checkpoints",
                  KAGGLE_API_TOKEN=userdata.get("KAGGLE_API_TOKEN"))

def save(*variants):
    for v in variants:                      # -0: JPEG is already compressed
        !cd {OUT} && zip -0 -qr /content/{v}_data.zip {v}_data && cp /content/{v}_data.zip {SAVE}/
    !cd {OUT} && cp manifest.csv boxes.json *_stats.csv {SAVE}/ 2>/dev/null

!nvidia-smi -L
```

**เช็ก:** ต้องเห็นชื่อ GPU ถ้าไม่เห็น ให้กลับไปทำข้อ 4

### Cell 2: โหลดโค้ด

```python
!git clone https://github.com/Chula-Com-Sci-2026/image-processing-project.git /content/repo
%cd /content/repo
!pip -q install -U kaggle
!pip freeze > {SAVE}/pip_freeze.txt
!kaggle datasets files kmader/food41 | head
```

**เช็ก:** ต้องเห็นรายชื่อไฟล์ของ dataset ถ้าขึ้น 401 Unauthorized ดูหัวข้อ [แก้ปัญหา](#แก้ปัญหา)

### Cell 3: โหลด dataset

```python
!kaggle datasets download -d kmader/food41 -p /content/data/raw_data --unzip
!ls /content/data/raw_data
```

**เช็ก:** ต้องเห็นโฟลเดอร์ `images` และ `meta` ขนาดรวมหลังแตกไฟล์น่าจะราว 11–12 GB

### Cell 4: prepare + light

```python
!python pipeline_preprocess/pipeline_01.py --steps prepare light
save("original", "clahe", "histeq")
!sha256sum {OUT}/manifest.csv
```

**เช็ก:** บรรทัด `[prepare] done:` ต้องแสดง train 6750, val 750, test 2500

### Cell 5: segment ชุดแรก (priority 1)

```python
!python pipeline_preprocess/pipeline_01.py --steps segment --seg-modes gsam
save("gsam", "gsam_crop")
```

- **ประมาณเวลา:** ดูจาก log `[boxes] n/10000` และ `[gsam] n/10000` ที่ขึ้นทุก 500 รูป
- **Fallback rate:** จดเลขในบรรทัด `[gsam] done, fallback rate` เพราะต้องใช้ในรายงาน
- **ถ้าหลุด:** ทำตาม [ส่วนที่ 5](#ส่วนที่-5-ถ้า-colab-หลุดกลางทาง) แล้วรัน Cell นี้ซ้ำ log จะขึ้น `resumed at n/10000`

### Cell 6: segment ที่เหลือ (priority 2)

แยกรันวันอื่นได้ แต่ต้องทำขั้นตอนใน [ส่วนที่ 5](#ส่วนที่-5-ถ้า-colab-หลุดกลางทาง) ก่อน

```python
!python pipeline_preprocess/pipeline_01.py --steps segment --seg-modes gsam_light
save("gsam_light", "gsam_light_crop")
!python pipeline_preprocess/pipeline_01.py --steps segment --seg-modes sam
save("sam", "sam_crop")
```

## ส่วนที่ 4: ตรวจผลและปิดงาน

### 7. เช็กไฟล์บน Drive

```python
!ls -lh {SAVE}
```

ต้องมีครบ:

- zip 9 ไฟล์: `original`, `clahe`, `histeq`, `gsam`, `gsam_crop`, `gsam_light`, `gsam_light_crop`, `sam`, `sam_crop`
- `manifest.csv`, `boxes.json`, `pip_freeze.txt`
- `gsam_stats.csv`, `gsam_light_stats.csv`, `sam_stats.csv`

เมื่อครบแล้วลบโฟลเดอร์ `data_zips/checkpoints/` บน Drive ได้ ถ้าจะรันใหม่ด้วย prompt, threshold หรือโมเดลอื่น **ต้องลบก่อน** ไม่อย่างนั้น pipeline จะนำผลเก่ากลับมาใช้

### 8. Commit manifest ลง repo

1. โหลด `manifest.csv` จาก `data_zips/` บน Drive
2. วางไว้ที่ `data/preprocessing_data/manifest.csv` ในเครื่อง
3. Commit และ push แล้วส่งค่า `sha256sum` ให้ทีมเทียบ

### 9. ปิด session

ไปที่ **Runtime → Disconnect and delete runtime** เพื่อไม่ให้เสียโควตา GPU

## ส่วนที่ 5: ถ้า Colab หลุดกลางทาง

ใช้ส่วนนี้เมื่อ `original_data.zip` และ `manifest.csv` อยู่บน Drive แล้ว ให้รัน Cell 1 แล้วรัน cell นี้แทน Cell 2–4

```python
import glob
!git clone https://github.com/Chula-Com-Sci-2026/image-processing-project.git /content/repo
%cd /content/repo
!mkdir -p {OUT}
for z in glob.glob(f"{SAVE}/*_data.zip"):
    !unzip -q -o "{z}" -d {OUT}
!cp {SAVE}/manifest.csv {SAVE}/boxes.json {SAVE}/*_stats.csv {OUT}/ 2>/dev/null
```

จากนั้นรัน Cell 5 หรือ 6 ต่อจากจุดที่ค้าง โหมดที่ save แล้วจะถูกข้ามเอง ส่วนโหมดที่ค้างจะอ่าน checkpoint จาก Drive แล้วทำต่อจาก 500 รูปล่าสุด ทั้ง `boxes` และ mask

## ส่วนที่ 6: สำหรับ data scientist

ต้องทำทุกครั้งที่เปิด session ใหม่ เพราะ `/content` เริ่มต้นว่างเสมอ

### 1. เพิ่ม cell นี้ระหว่าง cell ที่ตั้ง `VARIANT` กับ cell paths

```python
from google.colab import drive
drive.mount("/content/drive")
ROOT = "/content/drive/MyDrive/dip-project"   # ใช้ค่าตามตารางด้านล่าง
!unzip -q -o {ROOT}/data_zips/{VARIANT}_data.zip -d /content/data/preprocessing_data
```

### 2. แก้ cell paths

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/dip-project")   # ใช้ค่าตามตารางด้านล่าง
DATA_ROOT = Path("/content/data/preprocessing_data")
```

| คนใช้ | `ROOT` และ `PROJECT_ROOT` |
|---|---|
| เพื่อนในทีม (ผ่าน shortcut) | `/content/drive/MyDrive/dip-project` |
| เจ้าของโฟลเดอร์ | `/content/drive/MyDrive/2110651-digital-images-processing/2110651-digital-images-project/workspace/dip-project` |

### 3. เปลี่ยนแค่ `VARIANT` แล้ว Run All

ผลลัพธ์จะไปอยู่ที่ `results/` ใต้ `PROJECT_ROOT` บน Drive

## แก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `nvidia-smi` ไม่เจอ GPU | เลือก T4 GPU ในข้อ 4 แล้วรัน Cell 1 ใหม่ |
| `[segment] no CUDA GPU` | เหมือนข้อบน pipeline ข้าม step นี้โดยไม่ขึ้น error |
| Kaggle ขึ้น 401 Unauthorized | สร้าง token ใหม่ ถ้ายังไม่ได้ให้ใช้ legacy API key ตามด้านล่าง |
| `SecretNotFoundError` หรือเข้าถึง Secret ไม่ได้ | เช็กชื่อ Secret และเปิดสวิตช์ในข้อ 5 |
| `[prepare] STOP: expected ...` | ดูบรรทัด `skip unreadable` แล้วลบ `manifest.csv` กับ `original_data/` ก่อนรันใหม่ |
| `ModuleNotFoundError` หรือหา `pipeline_preprocess` ไม่เจอ | รัน `%cd /content/repo` แล้วลองใหม่ |
| หา zip ไม่เจอตอน unzip | เช็กค่า `ROOT` และ shortcut ของ `dip-project` |

**ถ้าต้องใช้ legacy API key:** สร้างจากหน้า Kaggle Settings จะได้ไฟล์ `kaggle.json` ที่มี `username` และ `key` ให้เพิ่ม Secrets ชื่อ `KAGGLE_USERNAME` และ `KAGGLE_KEY` แล้วเปลี่ยน `os.environ.update(...)` ใน Cell 1 เป็น

```python
os.environ.update(FOOD_DATA_DIR="/content/data",
                  FOOD_CHECKPOINT_DIR=f"{SAVE}/checkpoints",
                  KAGGLE_USERNAME=userdata.get("KAGGLE_USERNAME"),
                  KAGGLE_KEY=userdata.get("KAGGLE_KEY"))
```
