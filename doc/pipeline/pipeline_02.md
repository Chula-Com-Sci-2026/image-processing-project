# Pipeline 02: dark-class datasets

This replaces the old data process in `pipeline_01.py` (5 confusable pairs, 675/75/250 split). The team agreed to the change on 2026-11-03. The data engineer builds the datasets, and everyone else only reads them.

**Status (2026-11-05):** process 1 done, `01_original` built. Processes 2 and 3 not built yet. Values marked **PENDING** become known after a step runs.

## 1. What the team gets

Three datasets. All three contain the same images under the same relative paths, so a notebook switches datasets by changing one folder name.

| # | folder | built from | what changes |
|---|---|---|---|
| 1 | `data/dataset/01_original/` | Food-101 raw images | 10 dark classes, noisy labels removed, 60/20/20 split, resized |
| 2 | `data/dataset/02_denoise_clahe/` | dataset 1 | wavelet denoise → CLAHE |
| 3 | `data/dataset/03_denoise_clahe_gsam/` | dataset 2 | Grounded-SAM: food kept, background grey |

The course requires 3 preprocessing methods. They match the three kinds of pre-processing in the lecture notes (§10.1, PDF p.241: "noise removal, image enhancement, segmentation, before feature extraction"):

| kind (lecture notes §10.1) | our method | in dataset |
|---|---|---|
| Noise Removal | wavelet denoise | 2 |
| Image Enhancement | CLAHE | 2 |
| Segmentation | Grounded-SAM | 3 |

What each comparison measures:
- **1 vs 2**: denoise + CLAHE **together**. The two steps are not measured separately (decided 2026-11-05 to keep 3 datasets).
- **2 vs 3**: the effect of segmentation, with denoise + CLAHE held fixed. This comparison answers the research question.
- **1 vs 3**: all three steps together.

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
├── 02_denoise_clahe/{train,val,test}/<class>/<id>.jpg
└── 03_denoise_clahe_gsam/{train,val,test}/<class>/<id>.jpg   (+ meta/gsam_stats.csv)
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

**Result** (run 2026-11-05; CLIP probabilities and embeddings cached in `meta/clip_cache.npz`):

| | images | `wrong_label` | `nonfood` | flagged |
|---|---|---|---|---|
| official train | 7,500 | 34 (0.45%) | 22 (0.29%) | 56 (0.75%) |
| official test (reference) | 2,500 | 6 (0.24%) | 0 | 6 (0.24%, not reviewed) |

Residual noise that was **not** flagged: CLIP's top-1 is another class but p(label) ≥ 0.05 for 164 train images (2.2%), against 42 test images (1.7%). Test is clean, so about 0.5% of train (~35–40 images) may still have a wrong label. Most of the 164 are CLIP confusing dishes that share ingredients: baby_back_ribs → prime_rib (39), guacamole → nachos (35, guacamole is served with chips), paella → mussels (10). Wrong labels that CLIP agrees with can't be found this way. Report this in the report as the known residual noise. It is the same in every dataset, so it doesn't bias the comparison between datasets.

### 1c. Flag near-duplicates (whole pool)

The pool is split again in 1e, so duplicate images could land in train and test (leakage).
- Cosine similarity of CLIP image embeddings > 0.97 = **possible** near-duplicate. Every pair goes to review.
  - Threshold checked by eye (2026-11-05). The first try at 0.95 flagged 3,691 pairs, mostly different edamame photos. Below ~0.975 pairs are different photos. Real duplicates (re-cropped or colour-shifted) start at ~0.976. Above 0.98 almost all pairs are the same photo. No single value separates them, so the reviewer decides.
- Train–train pair: flag the second image (by sorted id).
- Train–test pair: flag the train copy.
- Test–test pair: flag the second image. **This is the only way an official test image can be removed.**
  - Decided 2026-11-05: remove one copy instead of forcing the pair into the same split. A copy adds no new information, and a dataset with no duplicates is easier to explain. Each removal is listed in `removed.csv` with reason `duplicate`.
  - Why it matters now: in the old process both copies stayed in test, so they were only counted twice. Here the pool is split again, so the two copies could land in train and test. The model would then recognise the photo instead of the food, and test accuracy would be too high.

**Result:** 80 pairs with cosine 0.97–1.00: train–train 41, train–test 32, test–test 7. They flag 63 images. One image sits in several pairs, which is why there are fewer images than pairs.

### 1d. Human review

- Contact sheets of flagged images, per class, with the image id.
- The reviewer fills a `decision` column (`keep` / `remove`) in `clip_flags.csv`.
- Write the criteria before looking at the images:
  - Remove: the label is wrong, there is no food in the image, or the image duplicates another.
  - **Keep**: the image is dark, blurry, partly cropped, or shows several dishes with the labelled one visible. **Never remove an image for being dark**, since darkness is why these classes were chosen.
- `removed.csv` = every row marked `remove`, with `reason` = `wrong_label` / `nonfood` / `duplicate`.

**Result** (reviewed by the data engineer, 2026-11-05):
- 118 images to review = 56 CLIP flags + 63 duplicate flags − 1 image with both.
- All 118 were judged noise and marked `remove`. No image outside the review list was removed.
- An image with both a CLIP flag and a duplicate flag is counted once, as `duplicate`. So the table shows 21 `nonfood`, not 22.
- The 6 official test images removed are all duplicates, as the rules allow.

### 1e. Pool, balance, split

1. Pool per class: official train after removals (≤ 750) + official test (250).
2. Count images per class into `class_counts.csv`. **N = the smallest count.**
3. Every class: sort, shuffle with seed 42, keep the first N.
4. Split each class: train = round(0.6·N), val = round(0.2·N), test = the rest.

**Random sampling, exactly** (`step_build` in `pipeline_02.py`, Python 3.12.3):
- Seed: `SEED = 42`.
- Generator: one `random.Random(42)` (Mersenne Twister), created once and **shared by all classes**.
- Classes are processed in the order of the `CLASSES` list: bibimbap, miso_soup, mussels, prime_rib, edamame, baby_back_ribs, paella, chicken_wings, guacamole, nachos. Since the generator is shared, changing this order changes the split.
- Inside a class: sort the kept images by key `"<class>/<id>"` (string sort, so `"1005649"` comes before `"101251"`), then `rng.shuffle`, then keep the first N.
- Split by position in the shuffled list: the first 578 are train, the next 193 val, the last 192 test.
- Official train and test images are pooled before shuffling, so every split mixes both sources. The `source` column in `manifest.csv` says which is which.
- The same code, seed, `clip_flags.csv` and Python version always give the same split. `sha256sum meta/manifest.csv` checks this.

**Result** (2026-11-05):

| class | removed: duplicate (test) | duplicate (train) | nonfood | wrong label | removed total | kept |
|---|---|---|---|---|---|---|
| edamame | 4 | 31 | 0 | 2 | 37 | **963** ← smallest |
| prime_rib | 0 | 9 | 3 | 5 | 17 | 983 |
| paella | 0 | 4 | 2 | 9 | 15 | 985 |
| baby_back_ribs | 0 | 5 | 2 | 7 | 14 | 986 |
| bibimbap | 0 | 2 | 4 | 5 | 11 | 989 |
| chicken_wings | 0 | 1 | 5 | 0 | 6 | 994 |
| mussels | 2 | 1 | 1 | 2 | 6 | 994 |
| guacamole | 0 | 0 | 2 | 3 | 5 | 995 |
| nachos | 0 | 2 | 2 | 1 | 5 | 995 |
| miso_soup | 0 | 2 | 0 | 0 | 2 | 998 |
| **total** | **6** | **57** | **21** | **34** | **118** | 9,882 |

- **N = 963**, set by edamame.
- Per class: train **578** · val **193** · test **192**.
- Total: train **5,780** · val **1,930** · test **1,920** = **9,630** images.
- 252 clean images are left out by balancing (kept − 963, summed over classes).
- Edamame sets N because most of its removals are duplicates (35 of 37). Edamame photos look alike, so some flagged pairs may be different photos. Removing a non-duplicate only loses images. It can't cause leakage or bias.

### 1f. Write images and manifest

- Resize to short side 256 px, JPEG quality 95 (same as before).
- Save to `01_original/<split>/<class>/<id>.jpg`.
- `manifest.csv`: `relpath,label,split,source`, with `source` = `official_train` / `official_test`.

### 1g. Checks (the script stops if any fail)

- Each split has the expected count per class.
- No image id appears in two splits.
- Everyone gets the same `sha256sum data/dataset/meta/manifest.csv`.
- Hash of the built manifest: `79eb1d1cf8a475b8deaa2a301acc64a1c3a334fd4a289c8d45af91c359c3c2a4`

**Result** (`build`, 2026-11-05, 44 s on CPU), all checks passed, plus an independent check afterwards:
- 9,630 images on disk = 9,630 rows in `manifest.csv`. 578 / 193 / 192 per class in every class.
- None of the 118 removed images is in the dataset. No image appears twice. No flagged duplicate pair has both images in the dataset.
- Every image has short side 256 px (300-image sample). `01_original` is 335 MB.
- Share of official-test images: train 24.8%, val 24.8%, test 26.8%. Close to the pool's 25%, so no split is dominated by one source.

## 3. Process 2: denoise + CLAHE (`02_denoise_clahe`)

- Input: `01_original`. Output: same relpaths.
- **Step 1, wavelet denoise:** `skimage.restoration.denoise_wavelet(img, channel_axis=-1, convert2ycbcr=True, method="BayesShrink", mode="soft", rescale_sigma=True)`.
- **Step 2, CLAHE** on the L channel of LAB, clip limit 2.0, tile 8×8. Only lightness is equalised, because equalising R, G, B separately shifts food colours.
- Same transform for train, val and test. It works on each image alone (nothing is fitted on the dataset), so there is no leakage.

### Why this order and these methods (decided 2026-11-05)

Measured on 10 dark train images (`doc/analysis/preprocess_metrics.md`):
- CLAHE brightens dark images (mean L 64.6 → 83.5) but **doubles the noise** (σ 1.50 → 3.15), because it stretches contrast in dark areas where camera and JPEG noise sit.
- Denoising **first** leaves less noise for CLAHE to amplify (σ 2.36 with the same brightness). Denoising after CLAHE would have to remove noise that has already been amplified.

Why wavelet as the denoiser:
- Noise sits mostly in high frequencies, but so do edges and texture. A low-pass filter (Gaussian, box) removes both. Wavelet thresholding keeps large coefficients (edges, which are sparse) and shrinks small ones (noise, which is spread out), so it keeps more texture.
- Median filter (taught in ch. 3 and 5) targets salt-and-pepper noise, not low-light camera noise.
- Cost: some fine texture is lost (Laplacian variance 1334 → 1199). Texture matters for classification, so this is a known risk. If dataset 2 does worse than dataset 1, this is one candidate reason to discuss.
- The lecture notes only name the idea (p.146). Cite Donoho, IEEE TIT 1995 (soft thresholding) and Chang, Yu & Vetterli, IEEE TIP 2000 (BayesShrink).

Ruled out (`analyst/preprocess_compare.ipynb`):
- unsharp mask, high-boost, and either one combined with CLAHE: noise σ goes up 4–5× (1.5 → 5.6–7.6), and they don't brighten the image.
- global hist-eq and gamma: they stretch near-black areas until the 8×8 JPEG blocks show. CLAHE's clip limit prevents this.
- contrast stretch → CLAHE: a second enhancement step on top of CLAHE. It adds a step without adding a new kind of pre-processing.

### To state in the report
- 1 vs 2 measures denoise + CLAHE together. We can't say which of the two caused a gain or loss.
- The parameters were set before training (defaults above), not tuned on val or test.

## 4. Process 3: Grounded-SAM (`03_denoise_clahe_gsam`)

- Input: `02_denoise_clahe`. Output: same relpaths.
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

1. ~~**"contrast → CLAHE"**~~: decided, dataset 2 = wavelet denoise → CLAHE (see section 3).
2. ~~**Optional dataset 4**~~: decided 2026-11-05, 3 datasets only.
3. ~~**Test–test duplicates (1c)**~~: decided, remove one copy (see 1c).
4. **Old `data/preprocessing_data/`**: keep as archive, or delete?
5. ~~**Reviewer** for 1d~~: done by the data engineer, 2026-11-05.
6. **Git**: track `data/dataset/meta/*.csv` like the old `manifest.csv` was tracked, and ignore the images?
