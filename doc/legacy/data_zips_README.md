# data_zips: ข้อมูลภาพสำหรับเทรนโมเดล

มี 9 variant ทุก zip มีภาพชุดเดียวกัน 10,000 รูป (train 6,750 · val 750 · test 2,500) และชื่อไฟล์ตรงกันทุกชุด

## ต้องใช้ไฟล์ไหน

เริ่มจากลำดับ 1 ให้ครบทุกโมเดลก่อน

| ลำดับ | ไฟล์ | ขนาด |
|---|---|---:|
| 1 | `original_data.zip` | 328 MB |
| 1 | `gsam_data.zip` | 215 MB |
| 1 | `gsam_crop_data.zip` | 283 MB |
| 2 | `clahe_data.zip` | 395 MB |
| 2 | `sam_data.zip` | 241 MB |
| 2 | `sam_crop_data.zip` | 270 MB |
| 2 | `gsam_light_data.zip` | 228 MB |
| 2 | `gsam_light_crop_data.zip` | 284 MB |
| 3 | `histeq_data.zip` | 371 MB |

## ถ้าเทรนบน Colab: ไม่ต้องโหลดลงเครื่อง

แตก zip จาก Drive ลง Colab ได้เลย ทำตาม "ส่วนที่ 6" ใน `doc/legacy/COLAB_GUIDE.md`

## ถ้าใช้ในเครื่อง

โหลดเฉพาะ variant ที่ใช้ (โหลดทีละไฟล์) แล้วแตกที่ `data/preprocessing_data/` จะได้โฟลเดอร์แบบนี้

```text
data/preprocessing_data/
├── original_data/
├── gsam_data/
└── gsam_crop_data/
```

ถ้าจะใช้ `analyst/compare_image.ipynb` ให้โหลด `gsam_stats.csv`, `gsam_light_stats.csv`, `sam_stats.csv` มาไว้ใน `data/preprocessing_data/` ด้วย

## ไฟล์อื่นในโฟลเดอร์

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `manifest.csv` | รายชื่อภาพและ split ใช้ตรวจว่าทุกคนใช้ข้อมูลชุดเดียวกัน |
| `*_stats.csv` | mask area และ fallback ของแต่ละภาพ ใช้ตอนวิเคราะห์ผล |
| `boxes.json` | box จาก Grounding DINO ไม่ต้องใช้ตอนเทรน |
| `pip_freeze.txt` | เวอร์ชัน package ตอนสร้างข้อมูล ไม่ต้องใช้ตอนเทรน |

## กติกา

- ห้ามแก้หรือลบไฟล์ในโฟลเดอร์นี้ ให้คัดลอกไปใช้
- ใช้ train/val/test ตามโฟลเดอร์ที่ให้ ห้ามแบ่งใหม่
- ห้ามเปิดดูหรือใช้ `test` เลือกโมเดล ประเมิน test ครั้งเดียวตอนท้าย
- ในโมเดลเดียวกัน ทุก variant ต้องใช้ setting เหมือนกัน เปลี่ยนแค่ `VARIANT`
- `sam` มี fallback 45.69% เกือบครึ่งชุดจึงยังเป็นภาพ `original` (`gsam` 1.54%, `gsam_light` 0.86%)

ตรวจว่า manifest ตรงกันด้วย `sha256sum data/preprocessing_data/manifest.csv` ต้องได้

```text
6769540ffb34031005b09bc5a6ac86b59c61ca4eb5ad397dd6418728b166a882
```
