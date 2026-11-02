# Preprocessing Pipeline Summary

Source material for building presentation slides about the data preparation stage of the project. Every value below comes from `pipeline_01.py`.

**Project:** 2110651 Digital Image Processing, Chulalongkorn University
**Research question:** Does segmentation before classification improve food menu prediction accuracy?
**Scope of this file:** data preparation, light correction, and segmentation. All three model families use this data.

## Rules for writing slides from this file

- Use only the numbers stated here. Do not invent accuracy, timing, or fallback figures.
- Values marked **PENDING** do not exist yet. Show them as "to be reported", or leave them out.
- There are no results from this dataset yet. The trial in section 8 used a different setup, so do not present it as a final result.
- Use the variant names exactly as written, such as `gsam_crop`.

## 1. Pipeline at a glance

One script builds 9 preprocessing variants of the same 10,000 images.

```
Food-101 (101 classes, 101,000 images)
   │
   │  prepare: keep 10 classes, split, resize
   ▼
original (train 6,750 · val 750 · test 2,500)
   │
   ├─ light ─────── clahe
   │                histeq
   │
   └─ segment
        ├─ Grounding DINO boxes ─┬─ SAM ViT-B ─── gsam, gsam_crop
        │                        └─ SlimSAM ───── gsam_light, gsam_light_crop
        └─ centre-point prompt ───── SAM ViT-B ── sam, sam_crop
```

`original` is both the baseline and the input to every other step. No variant is built on top of another processed variant.

| Step | Output | Hardware |
|---|---|---|
| `prepare` | `original`, `manifest.csv` | CPU |
| `light` | `clahe`, `histeq` | CPU |
| `segment` | 6 segmentation variants, `boxes.json`, `<mode>_stats.csv` | GPU (Google Colab) |

## 2. Step 1: Prepare the data

### Source dataset

Food-101 (Bossard et al., ECCV 2014), from Kaggle `kmader/food41` version 5. It has 101 classes, with an official split of 750 train and 250 test images per class.

### Class selection

We keep 10 of the 101 classes, chosen as 5 visually confusable pairs. Some pairs differ by their container, and others by the food itself.

| Pair | Class A | Class B |
|---|---|---|
| 1 | `ramen` | `pho` |
| 2 | `steak` | `filet_mignon` |
| 3 | `fried_rice` | `pad_thai` |
| 4 | `chocolate_cake` | `chocolate_mousse` |
| 5 | `caesar_salad` | `greek_salad` |

The analysis plan compares container-defined pairs (ramen/pho, cake/mousse) with food-defined pairs (steak/filet).

### Filtering and splitting

1. Read the official lists `meta/train.txt` and `meta/test.txt`.
2. Keep only the lines whose class is one of the 10 selected classes.
3. Sort each class's image IDs so the result does not depend on file order.
4. Shuffle each class's official training IDs with a random generator seeded at 42.
5. Assign the first 10% (75 images) to validation and the remaining 90% (675 images) to train.
6. Keep the official test images unchanged as test.

The split is stratified, so every class has the same counts.

| Split | Per class | Total | Source |
|---|---|---:|---|
| train | 675 | 6,750 | official train |
| val | 75 | 750 | official train |
| test | 250 | 2,500 | official test |
| **all** | **1,000** | **10,000** | |

The script stops with an error if the final counts differ from these totals.

### Resizing and storage

| Setting | Value |
|---|---|
| Resize rule | short side to 256 px, aspect ratio kept |
| Interpolation | `cv2.INTER_AREA` |
| Small images | images with a short side under 256 px are not upscaled |
| Format | JPEG, quality 95 |
| Path | `<variant>_data/{train,val,test}/<class>/<id>.jpg` |

### The manifest

`manifest.csv` lists `relpath`, `label`, and `split` for every image. All variants use the same relative paths, so each image in one variant has an exact counterpart in the others. This makes the comparison paired, because only the preprocessing differs.

## 3. Step 2: Light correction

Both methods run on `original` and change only the lightness channel. Equalising R, G, and B separately would shift food colours, and colour is a useful clue for telling dishes apart.

| Variant | Method | Colour space | Channel | Parameters |
|---|---|---|---|---|
| `clahe` | Contrast Limited Adaptive Histogram Equalization | LAB | L | clip limit 2.0, tile grid 8×8 |
| `histeq` | Global histogram equalization | YCrCb | Y | none |

**CLAHE steps:**
1. Convert BGR to LAB.
2. Apply CLAHE to L with clip limit 2.0 and an 8×8 tile grid.
3. Merge L with the unchanged A and B channels.
4. Convert back to BGR and save as JPEG quality 95.

**Histogram equalization steps:**
1. Convert BGR to YCrCb.
2. Apply `cv2.equalizeHist` to Y.
3. Convert back to BGR and save as JPEG quality 95, keeping Cr and Cb unchanged.

**Difference between them:** global equalization uses one mapping for the whole image. CLAHE computes a mapping per tile and limits contrast amplification with the clip limit, which reduces noise amplification in flat regions.

## 4. Step 3: Segmentation

### Methods

| Method | Prompt | Box or point source | Mask model | Hugging Face ID |
|---|---|---|---|---|
| `gsam` | text "food." | Grounding DINO tiny | SAM ViT-B | `facebook/sam-vit-base` |
| `gsam_light` | text "food." | same Grounding DINO boxes | SlimSAM | `Zigeng/SlimSAM-uniform-77` |
| `sam` | centre point of the image | no detector | SAM ViT-B | `facebook/sam-vit-base` |

The detector is `IDEA-Research/grounding-dino-tiny`, with box threshold 0.30 and text threshold 0.25.

`gsam` and `gsam_light` share the same saved boxes in `boxes.json`. Any difference between them therefore comes only from the mask model.

### Grounded-SAM steps (`gsam`, `gsam_light`)

1. Grounding DINO finds regions matching "food." in the 256 px image.
2. All detected boxes are passed to the mask model as prompts, one mask per box (`multimask_output=False`).
3. The masks are merged by union, so multiple food items in one image are all kept.

### SAM centre-point steps (`sam`)

1. Use the image centre as the single point prompt, with no text guidance.
2. SAM returns 3 candidate masks (`multimask_output=True`).
3. Keep the mask with the highest predicted IoU score.

This variant is the unguided comparison: it tests what SAM does when it receives no information about food.

### From mask to images

Each segmentation method produces two variants from one mask.

| Output | Background | Crop |
|---|---|---|
| `<seg>` such as `gsam` | replaced with grey (124, 116, 104) | yes |
| `<seg>_crop` such as `gsam_crop` | real pixels kept | yes |

1. **Find the crop:** take the bounding box of the mask, then extend it by 5% of the image height and width on each side, clipped to the image.
2. **Masked variant:** replace every pixel outside the mask with grey, then crop.
3. **Crop variant:** apply the same crop to the unmodified image.

The two variants use identical crops. Comparing them isolates the effect of removing background pixels from the effect of zooming in on the food.

The grey is the ImageNet mean colour, so it becomes approximately 0 after ImageNet normalization. The same grey fills the rotation corners during training augmentation.

### Failure handling

An image counts as failed in either of these cases:

- Grounding DINO finds no boxes (`gsam`, `gsam_light`).
- The final mask covers less than 5% of the image.

A failed image keeps its `original` photo in both output variants. No image is dropped, so every variant still has 10,000 images. Each `<mode>_stats.csv` records `relpath`, `split`, `mask_area`, and `fallback`.

| Mode | Fallback rate | Masks above 90% of the image | Mean mask area |
|---|---|---|---|
| `gsam` | 1.54% | 3.05% | 0.421 |
| `gsam_light` | 0.86% | 5.42% | 0.472 |
| `sam` | 45.69% | 0.35% | 0.184 |

The protocol requires reporting the fallback rate, because a high rate makes a segmentation variant behave more like `original`. Nearly half of the `sam` images are therefore unchanged photos, so `sam` results describe a mix of segmented and original images. Masks above 90% are not counted as failures, but they barely separate food from background.

## 5. All variants

| Variant | Description | Priority |
|---|---|---|
| `original` | resized photo, baseline | 1 |
| `gsam` | Grounded-SAM mask, grey background, cropped | 1 |
| `gsam_crop` | Grounded-SAM crop, real background | 1 |
| `clahe` | CLAHE on L of LAB | 2 |
| `sam`, `sam_crop` | SAM centre point, masked or crop only | 2 |
| `gsam_light`, `gsam_light_crop` | SlimSAM with the same boxes, masked or crop only | 2 |
| `histeq` | global histogram equalization on Y of YCrCb | 3 |

## 6. Design decisions

| Decision | Reason |
|---|---|
| Same manifest for all variants | Differences come from preprocessing, not from different images |
| Same preprocessing for train, val, and test | Segmentation is part of the model's input, so test images get it too |
| No offline augmentation | Dataset size stays identical in every variant |
| No colour jitter | Colour is a clue for food classes |
| Augmentation only at training time | Stored images stay deterministic |
| Fallback to the original image | Every variant keeps the same images and splits |
| Grey fill instead of black | Grey is neutral after ImageNet normalization |

## 7. Handoff to the model team

Each model family has one notebook that reads `data/preprocessing_data/<VARIANT>_data/`.

| Notebook | Model | Initialisation |
|---|---|---|
| `01_neural_network` | MLP (512-128) on HSV + HOG + LBP features | scratch |
| `02_cnn` | ResNet-50 (`resnet50.a1_in1k`) | ImageNet |
| `03_transformer` | DeiT-Small (`deit_small_patch16_224.fb_in1k`) | ImageNet |

Stored images are not square. The ResNet-50 and DeiT-Small notebooks produce 224 px inputs at load time:

| Stage | Transform |
|---|---|
| Train | rotate ±30° with grey fill, `RandomResizedCrop(224, scale=(0.6, 1.0))`, horizontal and vertical flip |
| Val, test | `Resize(256)`, `CenterCrop(224)` |

**Team rules:**
1. Do not re-split. Use the given `train/`, `val/`, and `test/` folders.
2. Within one model, run every variant with identical settings and change only `VARIANT`.
3. Select on validation macro-F1, evaluate test once, and report accuracy with macro precision, recall, and F1.

**Priority for the first round:** `original`, `gsam`, and `gsam_crop` across all three models.

## 8. Earlier trial (context only)

A trial on 25 October used a different setup: 20 classes, 400/200/400 images, and one seed. Noise was about ±4.5 points, so most gaps were not yet evidence.

- Grounded-SAM helped a scratch CNN and a handcrafted-feature MLP by about +6 to +7 points, but hurt pretrained MobileNetV3.
- Possible reasons, not yet tested: a grey background is out-of-distribution for ImageNet features, plates and bowls are real clues, and bad masks remove food.
- SAM with a centre-point prompt often failed, while Grounded-SAM usually kept the whole dish.

## 9. Current status

| Item | Status as of 2 November 2026 |
|---|---|
| Pipeline code | complete (`pipeline_01.py`) |
| `original`, `clahe`, `histeq`, manifest | generated and stored on Drive |
| `gsam`, `gsam_light`, `sam` runs | generated and stored on Drive (9 zips, 10,000 images each, same file names) |
| Manifest SHA-256 for the team to verify | `6769540ffb34031005b09bc5a6ac86b59c61ca4eb5ad397dd6418728b166a882` |
| Model training | **PENDING** |

## 10. References

- Food-101: Bossard et al., ECCV 2014
- CLAHE: Zuiderveld, 1994
- SAM: Kirillov et al., ICCV 2023
- Grounding DINO: Liu et al., ECCV 2024
- Grounded SAM: Ren et al., arXiv 2401.14159
- SlimSAM: Chen et al., NeurIPS 2024
- ResNet: He et al., CVPR 2016
- DeiT: Touvron et al., ICML 2021
