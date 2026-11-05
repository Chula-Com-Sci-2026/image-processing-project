# CLAUDE.md: Food segmentation-before-classification project

Team project for 2110651 Digital Image Processing (Chulalongkorn, MSc CS, 4-person team).

**Research question:** Does segmentation before classification improve food menu prediction accuracy?

Team: 2 data scientists, 1 data engineer (data preparation & transforms), 1 software developer.
**This workspace is the data engineer's.** It builds the datasets the team trains on. Models are trained outside this workspace.

## How to work with me
- I want to understand and write parts myself. Explain *why* before *how*, and prefer small diffs I can review.
- Push back when a choice would bias the comparison. Experiment correctness matters more than speed.
- Ask before changing anything under "Fixed protocol" or the folder layout of `data/dataset/`, since the whole team depends on it.

## Project layout
See the folder tree in `README.md`. In short:
- `pipeline_preprocess/pipeline_02.py`: builds the 3 datasets in `data/dataset/`. All constants (classes, seed, ratios, sizes, model IDs) sit at the top of the file. The plan, step by step, is in `doc/pipeline/pipeline_02.md`. Keep the two in sync.
- `pipeline_preprocess/pipeline_01.py` and `data/preprocessing_data/`: the old process (5 confusable pairs, 675/75/250 split). Replaced on 2026-11-03. Don't build on it.
- `analyst/`: notebooks for looking at images, masks and stats (figures for slides).
- `doc/`: every md except CLAUDE.md and README.md. `pipeline/` current process, `analysis/` findings, `references/` citations, `legacy/` old process.
- `data/raw_data/`: Food-101 as downloaded. `data/dataset/`: the datasets for the team.

## Datasets for the team
Each dataset is built from the one before it. All three hold the same images under the same relpaths and share one `manifest.csv`.

| # | folder | built from | what changes |
|---|---|---|---|
| 1 | `01_original` | Food-101 raw | 10 dark classes, noise removed, 60/20/20 split, resized |
| 2 | `02_denoise_clahe` | dataset 1 | wavelet denoise → CLAHE |
| 3 | `03_denoise_clahe_gsam` | dataset 2 | Grounded-SAM: food kept, grey background, cropped to food |

The course requires 3 preprocessing methods: noise removal (wavelet), enhancement (CLAHE), segmentation (Grounded-SAM), matching the lecture notes §10.1. 1 vs 2 measures denoise + CLAHE together. 2 vs 3 measures segmentation with them held fixed, which answers the research question. Classifiers (zero-shot, classical, CNN, Transformer) are trained by the rest of the team.

## Fixed protocol
| item | value |
|---|---|
| Dataset | Food-101 (Bossard et al., ECCV 2014), Kaggle `kmader/food41` v5 |
| Classes | 10 classes with the lowest mean image brightness: bibimbap, miso_soup, mussels, prime_rib, edamame, baby_back_ribs, paella, chicken_wings, guacamole, nachos |
| Noise removal | official train only (750/class): CLIP flags, then a human decides. The official test set (250/class) is already hand-cleaned. Near-duplicates are removed from the whole pool |
| Balance | pool clean train + test per class; N = smallest class count = **963** (edamame); sample N per class, seed 42 |
| Split | 60 / 20 / 20 train / val / test per class = 578 / 193 / 192 (9,630 images, manifest sha256 `79eb1d1c…`) |
| Stored size | short side 256 px, JPEG q95 |
| Denoise | `skimage` `denoise_wavelet`, BayesShrink, soft, YCbCr, `rescale_sigma=True` (before CLAHE) |
| CLAHE | L channel of LAB, clip 2.0, tile 8×8 |
| Grounded-SAM | Grounding DINO tiny ("food.", box 0.30, text 0.25) + SAM ViT-B |
| Fill colour | ImageNet mean grey (124,116,104) for the mask background |
| Failed mask | below 5% of the image: keep the dataset-2 image, and count it as a fallback |

Rules:
- Never remove an image for being dark. Darkness is why these classes were chosen.
- Every removal goes in `data/dataset/meta/removed.csv` with a reason.
- Same manifest for every dataset. No offline augmentation.
- Train, val and test get the same preprocessing.
- Always report the segmentation fallback rate.

## Rules for the team (training outside this workspace)
1. Don't re-split. Use the given train/val/test folders.
2. Within one model, run every dataset with identical settings.
3. Select on val macro-F1, evaluate test once, and report accuracy + macro P/R/F1.

## What we already know (25-Oct trial on the old classes, `trial_results.html` in the Project)
- Setup: 20 classes, 400/200/400 images, one seed. Noise was about ±4.5 pt, so most gaps are not evidence yet.
- Grounded-SAM helped the handcrafted-feature MLP and the scratch CNN. It hurt pretrained MobileNetV3. Likely reasons: the grey background is out-of-distribution for ImageNet features, and bad masks remove food.
- SAM with a centre-point prompt often fails. Grounded-SAM usually keeps the whole dish.

## Commands
```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python pipeline_preprocess/pipeline_02.py --steps brightness flag   # then human review
.venv/bin/python pipeline_preprocess/pipeline_02.py --steps build
.venv/bin/python pipeline_preprocess/pipeline_02.py --steps denoise_clahe   # dataset 2, CPU
# dataset 3 (--steps gsam) needs a GPU: doc/pipeline/COLAB_GUIDE.md
.venv/bin/python pipeline_preprocess/pipeline_02.py --steps zip   # self-contained zip per dataset for the team (no git needed)
sha256sum data/dataset/meta/manifest.csv                   # everyone must get the same hash
```

## Key references
Food-101 (Bossard et al., ECCV 2014) · CLIP (Radford et al., ICML 2021) · SAM (Kirillov et al., ICCV 2023) · Grounding DINO (Liu et al., ECCV 2024) · Grounded SAM (Ren et al., arXiv 2401.14159) · FoodSAM (Lan et al., IEEE TMM 2023) · Segment Anything in Food Images (Alahmari et al., CVPRW 2024) · CLAHE (Zuiderveld, 1994) · Noise or Signal (Xiao et al., ICLR 2021)
