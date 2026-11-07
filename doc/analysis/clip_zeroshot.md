# CLIP zero-shot on the three datasets

Which preprocessing gives CLIP the best accuracy, with no training at all? Source: `analyst/clip_zeroshot.py` (2026-11-07). Raw output: `analyst/results/clip_zeroshot/summary.md`, plus one prediction csv per dataset.

**Status:** datasets 1 and 2 are done. Dataset 3 (`03_denoise_clahe_gsam`) was built on Colab and is not in this workspace yet. Copy it to `data/dataset/03_denoise_clahe_gsam/` and run the script again. The cached datasets are not re-scored.

## Short answer (so far)

Denoise + CLAHE **lowers** CLIP accuracy, from 97.45% to 96.20% (−1.25 pt). The paired test says this is a real effect (McNemar p = 0.0015). Almost all of the loss is in two class pairs that are already confused: ribs and paella/mussels.

## Setup

| item | value |
|---|---|
| Model | `openai/clip-vit-base-patch16` (same as `CLIP_ID` in `pipeline_02.py`), Hugging Face `transformers` |
| Method | zero-shot: cosine similarity between the image embedding and each class's text embedding, softmax, argmax. No training, no fine-tuning |
| Prompt | `"a photo of {class}, a type of food."`, `_` replaced by a space. One prompt per class, 10-way choice |
| Prompt choice | taken unchanged from the flag step. Not tuned on val, so val was not used at all |
| Image input | the stored dataset JPEG (short side 256). The CLIP processor resizes to 224 and centre-crops, the same for every dataset |
| Hardware | CPU (12 threads, torch 2.15 CPU build), batch 64, about 7.5 min per dataset |
| Metrics | accuracy (± binomial SE), macro precision / recall / F1 over the 10 classes |
| Comparison | exact two-sided McNemar test on paired correctness, because every dataset has the same images |

Run it:
```bash
.venv/bin/python analyst/clip_zeroshot.py                 # every dataset folder found
.venv/bin/python analyst/clip_zeroshot.py --datasets 03_denoise_clahe_gsam
```

## Data used

The test split from `data/dataset/meta/manifest.csv`, the same file for every dataset.

| item | value |
|---|---|
| Classes | bibimbap, miso_soup, mussels, prime_rib, edamame, baby_back_ribs, paella, chicken_wings, guacamole, nachos |
| Test images | 1,920 (192 per class), same relpaths in every dataset |
| From official Food-101 test | 515 |
| From official Food-101 train | 1,405 (these went through the CLIP noise filter) |
| `01_original` | resized only |
| `02_denoise_clahe` | wavelet denoise (BayesShrink, soft, YCbCr) → CLAHE on L (clip 2.0, 8×8) |
| `03_denoise_clahe_gsam` | not run yet |

### Bias from the CLIP noise filter

The flag step used **this same CLIP model** to find mislabelled images in the official train set, and a person then removed them. CLIP's worst mistakes on official-train images were therefore removed before the split, so CLIP's accuracy here is a little too high. The effect is small. Of about 7,500 official train images, only 55 were removed on CLIP's flag (34 `wrong_label`, 21 `nonfood`), plus 63 near-duplicates. That is about 0.7% of the pool, so CLIP accuracy is at most about 0.7 pt too high.

The filter ran on the original pixels, so if anything this bias favours dataset 1. To check this, the results are also reported on the 515 official-test images, which the filter never touched.

## Results

### All test images (n = 1,920), main result

| dataset | accuracy | ±SE | macro P | macro R | macro F1 |
|---|---|---|---|---|---|
| 01_original | **97.45** | 0.36 | 97.51 | 97.45 | **97.44** |
| 02_denoise_clahe | 96.20 | 0.44 | 96.38 | 96.20 | 96.19 |

McNemar 1 vs 2: dataset 1 gets 39 images right that dataset 2 gets wrong. The reverse happens 15 times. **p = 0.0015**.

### Official-test images only (n = 515), bias check

| dataset | accuracy | ±SE | macro P | macro R | macro F1 |
|---|---|---|---|---|---|
| 01_original | **96.50** | 0.81 | 96.81 | 96.40 | **96.49** |
| 02_denoise_clahe | 94.95 | 0.96 | 95.58 | 94.96 | 94.99 |

McNemar: 12 vs 4, p = 0.077. The gap is the same size as before (−1.55 pt), and the p-value is higher only because n is smaller. As expected, both datasets score about 1 pt lower on these images than on the full set, but the ranking does not change. The filter bias doesn't explain the result.

### Per-class F1 (all test)

| class | 01_original | 02_denoise_clahe | change |
|---|---|---|---|
| bibimbap | 98.2 | 97.6 | −0.6 |
| miso_soup | 99.5 | 99.0 | −0.5 |
| mussels | 97.7 | 95.0 | −2.7 |
| prime_rib | 95.7 | 92.6 | −3.1 |
| edamame | 99.5 | 99.5 | 0.0 |
| baby_back_ribs | 94.7 | 91.3 | −3.4 |
| paella | 96.0 | 93.4 | −2.6 |
| chicken_wings | 98.7 | 98.7 | 0.0 |
| guacamole | 97.1 | 97.6 | +0.5 |
| nachos | 97.4 | 97.2 | −0.2 |

### Most common confusions (true → predicted)

| confusion | 01_original | 02_denoise_clahe |
|---|---|---|
| baby_back_ribs → prime_rib | 13 | 24 |
| paella → mussels | 7 | 17 |
| guacamole → nachos | 7 | 6 |
| chicken_wings → baby_back_ribs | 3 | 3 |

## Reading the results

- **The loss comes from pairs that were already hard.** Ribs ↔ prime rib and paella → mussels account for 21 extra errors out of the net 24. Both pairs are brown or dark meat and seafood, told apart by fine texture and colour. CLAHE raises local contrast, and that seems to blur those fine cues.
- **Likely reason: CLIP was trained on ordinary web photos.** CLAHE images look less like those photos. The 25-Oct trial saw the same pattern: preprocessing that changes how an image looks helped models trained from scratch but hurt pretrained MobileNetV3. So a pretrained model is not a neutral judge of pixel-level enhancement.
- **Ceiling effect.** At 97% there is little room to improve, so CLIP zero-shot is not a sensitive test for the research question. Dataset 3 vs 2 may move only a point or two either way. Use McNemar, not the raw gap, to judge it.
- **This doesn't mean "drop denoise + CLAHE".** It is one model with no training. The trained classifiers (classical, scratch CNN, Transformer) may react differently. Also, dataset 3 is built on dataset 2 by design, so 2 vs 3 still isolates segmentation.

## Still to do

- [ ] Run `03_denoise_clahe_gsam` and add it to the tables (2 vs 3 answers the research question).
- [ ] For dataset 3, also report accuracy separately on the fallback images (mask < 5%, identical to dataset 2) and on the segmented images, plus the fallback rate.
