# CLAUDE.md: Food segmentation-before-classification project

Team project for 2110651 Digital Image Processing (Chulalongkorn, MSc CS, 4-person team).

**Research question:** Does segmentation before classification improve food menu prediction accuracy?

Team: 2 data scientists (work in notebooks), 1 data engineer (data preparation & transforms), 1 software developer.

## How to work with me
- I want to understand and write parts myself. Explain *why* before *how*, and prefer small diffs I can review.
- Push back when a choice would bias the comparison. Experiment correctness matters more than speed.
- Ask before changing anything under "Fixed protocol" or the folder layout of `data/preprocessing_data/`, since every notebook depends on it.

## Project layout
See the folder tree in `README.md`. In short:
- `pipeline_preprocess/pipeline_01.py` (data engineer): one script with three steps (`prepare`, `light`, `segment`) that builds `data/preprocessing_data/<variant>_data/{train,val,test}/<class>/*.jpg`. All constants (classes, split seed, sizes, model IDs) sit at the top of the file. Every step reads `original_data/` and keeps the same relative paths as `manifest.csv`.
- `model_pipeline/` (everyone): one self-contained notebook per model family: `01_neural_network`, `02_cnn`, `03_transformer`. Each reads `data/preprocessing_data/<VARIANT>_data/` directly and writes to `results/<family>/`. Paths are preset in the second cell. No shared library on purpose: each person owns their notebook.
- `data/raw_data/`: Food-101 as downloaded. `data/preprocessing_data/`: all generated variants.

New preprocessing ideas go in a new `pipeline_preprocess/pipeline_02.py` that reads `original_data/` and writes new `<name>_data/` folders. Don't edit `pipeline_01.py` in a way that changes data that has already been used for results.

Team rules (deliberately minimal):
1. Don't re-split. Use the given train/val/test folders.
2. Within one model, run every variant with identical notebook settings. The research question compares variants *within* a model, so consistency only has to hold inside each person's notebook.
3. Select on val, evaluate test once, and report accuracy + macro P/R/F1.

## What we already know (25-Oct trial, `trial_results.html` in the Project)
- Setup: 20 classes, 400/200/400 images, one seed. Noise was about ±4.5 pt, so most gaps are not evidence yet.
- Segmentation (Grounded-SAM) helped the scratch CNN and the handcrafted-feature MLP by +6–7 pt, but hurt pretrained MobileNetV3.
- Likely reasons, not yet tested: a grey background is out-of-distribution for ImageNet features; plates and bowls are real clues; bad masks remove food.
- SAM with a centre-point prompt often fails. Grounded-SAM usually keeps the whole dish.

## Fixed protocol (identical for every run)
| item | value |
|---|---|
| Dataset | Food-101 (Bossard et al., ECCV 2014), Kaggle `kmader/food41` v5 |
| Classes | 10 = 5 confusable pairs: ramen/pho, steak/filet_mignon, fried_rice/pad_thai, chocolate_cake/chocolate_mousse, caesar_salad/greek_salad |
| Split | official train 750/class → 675 train + 75 val (seed 42); official test 250/class |
| Expected counts | train 6,750 · val 750 · test 2,500 (`pipeline_01.py` stops otherwise) |
| Stored size | short side 256 px, JPEG q95 |
| Augmentation | online, train only: rotate ±30°, RandomResizedCrop(224, 0.6–1.0), h/v flip |
| Fill colour | ImageNet mean grey (124,116,104), used for mask background and rotation corners |
| Selection | best epoch / hyper-params by **val** macro-F1; test evaluated once |
| Metrics | accuracy, macro P/R/F1, per-class F1, confusion matrix, per-image predictions |

Rules: same manifest for every variant; no offline augmentation; no colour jitter; never tune on test; train and test get the same preprocessing; always report the segmentation fallback rate.

## Experiment grid
**Variants** (`data/preprocessing_data/<variant>_data`):
| variant | what | made by |
|---|---|---|
| `original` | resized photo (baseline) | `pipeline_01.py` prepare |
| `clahe` | CLAHE on L of LAB (clip 2.0, 8×8) | `pipeline_01.py` light |
| `histeq` | global hist-eq on Y of YCrCb | `pipeline_01.py` light |
| `gsam`, `gsam_crop` | Grounding DINO ("food.") boxes + SAM ViT-B | `pipeline_01.py` segment |
| `gsam_light`, `gsam_light_crop` | same boxes + SlimSAM | `pipeline_01.py` segment |
| `sam`, `sam_crop` | SAM ViT-B, centre-point prompt | `pipeline_01.py` segment |

- `<seg>`: mask, grey background, crop to food. `<seg>_crop`: crop to the food box, real pixels kept.
- Masks below 5% of the image count as failed, and the original image is kept.

**Models:**
| notebook | model | init | role |
|---|---|---|---|
| `01_neural_network` | MLP (512-128) on HSV + HOG + LBP features | scratch | classical side of the rubric's "classical vs DL" |
| `02_cnn` | ResNet-50 (timm `resnet50.a1_in1k`) | ImageNet | CNN, professor's recommended backbone |
| `03_transformer` | DeiT-Small (timm `deit_small_patch16_224.fb_in1k`) | ImageNet | Transformer, 22M params ≈ ResNet-50 |

**Priority:** (1) `original`, `gsam`, `gsam_crop` × all models; (2) `clahe`, `sam*`, `gsam_light*`; (3) `histeq`, extra seeds.

## Commands
```bash
pip install -r requirements.txt
python pipeline_preprocess/pipeline_01.py                   # prepare + light (+ segment if CUDA)
python pipeline_preprocess/pipeline_01.py --steps segment   # Colab GPU
sha256sum data/preprocessing_data/manifest.csv              # everyone must get the same hash
```
Then open the notebook for the model in `model_pipeline/`, change only `VARIANT`, and Run All.

## Analysis plan
- Main table: test accuracy per variant × model, Δ vs `original`. Optional: McNemar on per-image predictions (`results/<family>/*_preds.csv`).
- Per-class F1 on the confusable pairs: does segmentation help container-defined pairs (ramen/pho, cake/mousse) differently from food-defined pairs (steak/filet)?
- Relate gains to mask quality and fallback rate (`<seg>_stats.csv`).

## Optional extension: FoodSeg103
- Mask quality: run the segmenters on the FoodSeg103 test set, collapse labels to food/background, and report IoU per method.
- If time allows: train a binary U-Net on FoodSeg103 as a supervised segmentation variant.
- FoodSeg103 labels are ingredients, not dishes. Do not use it for classification.

## Notebook hygiene
- Clear outputs before committing. Notebook outputs cause most merge conflicts.
- Each person works on their own notebook copy, never someone else's in place.

## Key references
Food-101 (Bossard et al., ECCV 2014) · ResNet (He et al., CVPR 2016) · DeiT (Touvron et al., ICML 2021) · SAM (Kirillov et al., ICCV 2023) · Grounding DINO (Liu et al., ECCV 2024) · Grounded SAM (Ren et al., arXiv 2401.14159) · SlimSAM (Chen et al., NeurIPS 2024) · FoodSAM (Lan et al., IEEE TMM 2023) · Segment Anything in Food Images (Alahmari et al., CVPRW 2024) · Noise or Signal (Xiao et al., ICLR 2021) · Zhu et al., IJCAI 2017 · FoodSeg103 (Wu et al., ACM MM 2021) · CLAHE (Zuiderveld, 1994) · McNemar for classifiers (Dietterich, 1998)
