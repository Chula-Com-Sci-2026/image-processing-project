# Pipeline 02: dark-class datasets

This replaces the old data process in `pipeline_01.py` (5 confusable pairs, 675/75/250 split). The team agreed to the change on 2026-11-03. The data engineer builds the datasets, and everyone else only reads them.

**Status:** plan only. Nothing below has been built yet. Values marked **PENDING** become known after a step runs.

## 1. What the team gets

Three datasets. All three contain the same images under the same relative paths, so a notebook switches datasets by changing one folder name.

| # | folder | built from | what changes |
|---|---|---|---|
| 1 | `data/dataset/01_original/` | Food-101 raw images | 10 dark classes, noise removed, 60/20/20 split, resized |
| 2 | `data/dataset/02_clahe/` | dataset 1 | CLAHE contrast enhancement |
| 3 | `data/dataset/03_clahe_gsam/` | dataset 2 | Grounded-SAM: food kept, background grey |

What each comparison measures:
- **1 vs 2**: the effect of CLAHE.
- **2 vs 3**: the effect of segmentation, with CLAHE held fixed. This comparison answers the research question.
- **1 vs 3**: both steps together.

```
data/dataset/
├── meta/
│   ├── brightness.csv       mean brightness of every Food-101 image (class selection)
│   ├── clip_flags.csv       CLIP flags + reviewer decision for every official-train image
│   ├── duplicates.csv       near-duplicate pairs
│   ├── removed.csv          images removed, with reason
│   ├── class_counts.csv     images per class after cleaning; N = min
│   └── manifest.csv         relpath,label,split,source (identical for all 3 datasets)
├── 01_original/{train,val,test}/<class>/<id>.jpg
├── 02_clahe/{train,val,test}/<class>/<id>.jpg
└── 03_clahe_gsam/{train,val,test}/<class>/<id>.jpg   (+ meta/gsam_stats.csv)
```

## 2. Process 1: base dataset (`01_original`)

### 1a. Class selection by brightness (done once, then fixed)

- Metric: mean grayscale value (0–255) of each image, averaged per class over all 1,000 images (official train + test).
- Picked the 10 classes with the lowest mean brightness. The Food-101 average is 118.1.

| rank | class | mean brightness | % dark images (< 80) |
|---|---|---|---|
| 1 | bibimbap | 98.8 | 22.8 |
| 2 | miso_soup | 101.7 | 20.2 |
| 3 | mussels | 101.9 | 20.0 |
| 4 | prime_rib | 106.2 | 15.2 |
| 5 | edamame | 106.4 | 15.0 |
| 6 | baby_back_ribs | 107.2 | 12.4 |
| 7 | paella | 107.3 | 13.5 |
| 8 | chicken_wings | 107.8 | 13.4 |
| 9 | guacamole | 108.1 | 14.5 |
| 10 | nachos | 108.4 | 12.3 |

- The list is written as a constant in the script. The ranking is saved in `meta/brightness.csv`, but the script does not rank again, so the class list cannot change between runs.
- Caveat for the report: brightness also reflects food colour (mussels and miso soup are dark foods), not only low-light photos.

### 1b. Flag possible noise in official train (750 per class)

Food-101 authors left the train images uncleaned on purpose, so some have wrong labels or no food. The official test images were cleaned by hand.

- Model: CLIP `openai/clip-vit-base-patch16`, zero-shot.
- Prompts: `a photo of <class>, a type of food.` for the 10 classes, plus non-food prompts (a person, text or a menu, a restaurant interior, a drink, a kitchen).
- An image gets a flag when:
  - `wrong_label`: CLIP's top prompt is another class **and** the probability of the true label is < 0.05.
  - `nonfood`: CLIP's top prompt is a non-food prompt.
- CLIP also runs on the official test set as a **reference**. Because test is clean, its flag rate is CLIP's own error rate. Report train flag rate vs test flag rate.
- CLIP only flags. It does not remove anything.

### 1c. Flag near-duplicates (whole pool)

The pool is split again in 1e, so duplicate images could land in train and test (leakage).
- Cosine similarity of CLIP image embeddings > 0.95 = near-duplicate. Check this threshold by eye on the review sheet.
- Train–train pair: flag the second image (by sorted id).
- Train–test pair: flag the train copy.
- Test–test pair: flag the second image. **This is the only way an official test image can be removed.**

### 1d. Human review

- Contact sheets of flagged images, per class, with the image id.
- The reviewer fills a `decision` column (`keep` / `remove`) in `clip_flags.csv`.
- Write the criteria before looking at the images:
  - Remove: the label is wrong, there is no food in the image, or the image duplicates another.
  - **Keep**: the image is dark, blurry, partly cropped, or shows several dishes with the labelled one visible. **Never remove an image for being dark**, since darkness is why these classes were chosen.
- `removed.csv` = every row marked `remove`, with `reason` = `wrong_label` / `nonfood` / `duplicate`.

### 1e. Pool, balance, split

1. Pool per class: official train after removals (≤ 750) + official test (250).
2. Count images per class into `class_counts.csv`. **N = the smallest count.** PENDING
3. Every class: sort ids, shuffle with seed 42, keep the first N.
4. Split each class: train = round(0.6·N), val = round(0.2·N), test = the rest.

Example with N = 800: 480 / 160 / 160 per class, 4,800 / 1,600 / 1,600 in total.

### 1f. Write images and manifest

- Resize to short side 256 px, JPEG quality 95 (same as before).
- Save to `01_original/<split>/<class>/<id>.jpg`.
- `manifest.csv`: `relpath,label,split,source`, with `source` = `official_train` / `official_test`.

### 1g. Checks (the script stops if any fail)

- Each split has the expected count per class.
- No image id appears in two splits.
- No near-duplicate pair crosses two splits.
- Everyone gets the same `sha256sum data/dataset/meta/manifest.csv`.

## 3. Process 2: CLAHE (`02_clahe`)

- Input: `01_original`. Output: same relpaths.
- CLAHE on the L channel of LAB, clip limit 2.0, tile 8×8. Only lightness is equalised, because equalising R, G, B separately shifts food colours.
- Same transform for train, val and test. It works on each image alone (nothing is fitted on the dataset), so there is no leakage.

## 4. Process 3: Grounded-SAM (`03_clahe_gsam`)

- Input: `02_clahe`. Output: same relpaths.
- Grounding DINO tiny, prompt `food.` (box threshold 0.30, text 0.25) → SAM ViT-B mask → background filled with ImageNet mean grey (124, 116, 104) → crop to the food box (+5% margin).
- A mask smaller than 5% of the image counts as failed, and the dataset-2 image is kept unchanged.
- `meta/gsam_stats.csv`: mask area and fallback per image. **Always report the fallback rate.**
- Needs a GPU (Colab).

## 5. Rules for everyone (unchanged)

- Don't re-split. Use the given train/val/test folders.
- Within one model, run all datasets with identical notebook settings.
- Pick epochs and hyper-parameters on **val** macro-F1. Evaluate test once.
- Report accuracy, macro P/R/F1, per-class F1, confusion matrix, per-image predictions.

## 6. Open questions

1. **"contrast → CLAHE"**: is this one step (CLAHE is the contrast enhancement), or two steps (a global contrast stretch first, then CLAHE)? This plan assumes one step.
2. **Optional dataset 4 `04_gsam`** (Grounded-SAM on dataset 1, no CLAHE): without it we can't tell whether segmentation helps only after CLAHE. It costs one more GPU run.
3. **Test–test duplicates (1c)**: OK to remove one copy, even though it is official test?
4. **Old `data/preprocessing_data/`**: keep as archive, or delete?
5. **Reviewer** for 1d.
6. **Git**: track `data/dataset/meta/*.csv` like the old `manifest.csv` was tracked, and ignore the images?
