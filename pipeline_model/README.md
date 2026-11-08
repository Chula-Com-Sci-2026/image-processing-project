# pipeline_model: train and evaluate the classifiers

The team's model code. One notebook trains and evaluates every model on **one dataset**. You run it once per dataset by changing a single variable, so all datasets get identical settings.

| file | what it does |
|---|---|
| `04_denoise_clahe_dino.ipynb` | the notebook. Despite its name it serves all 4 datasets: set `DATASET` in the first code cell |

The datasets come from `pipeline_preprocess/` (see the repo README). This folder only reads them.

## Models

| # | model | tuned on val | trained? |
|---|---|---|---|
| 1 | CLIP ViT-B/32 zero-shot (baseline) | prompt template | no |
| 2 | Classical: HSV histogram + HOG + LBP → Logistic Regression / SVM (RBF) | `C`, `gamma` | yes |
| 3 | CNN from scratch | depth (3/4/5 conv blocks), epochs (early stopping) | yes |
| 4 | ResNet-50, ImageNet weights | how much to unfreeze (`fc` / `layer4` / `all`), epochs | yes |

Selection uses **val macro-F1**. Test is scored for every config but never used to choose one.

Reported per model: top-1/3/5 accuracy, macro precision/recall/F1, per-class AP and mAP, confusion matrix, correct and wrong examples, Grad-CAM.

## Setup

```bash
uv pip install --python .venv/bin/python -r requirements.txt   # torch, torchvision, transformers, scikit-learn, scikit-image, opencv
```

Device is picked automatically: CUDA → Apple MPS → CPU.

## Get the data

Every dataset zip already contains `meta/manifest.csv`, so you don't need git for the data.

**Local (VS Code):** download the zip(s) from Drive `dip-project/dataset_zips/` and unzip into `data/dataset/` at the repo root:

```text
data/dataset/
├── meta/manifest.csv
└── 04_denoise_clahe_dino/{train,val,test}/<class>/<id>.jpg
```

**Colab:** the notebook looks for `MyDrive/food_classify/<DATASET>.zip` and unzips it itself. Copy the zip from `dip-project/dataset_zips/` into `MyDrive/food_classify/`, or change `DRIVE_DIR` in the paths cell. Results are written to `MyDrive/food_classify/results/`, so they survive a disconnect.

The notebook stops if the manifest hash isn't `79eb1d1cf8a475b8deaa2a301acc64a1c3a334fd4a289c8d45af91c359c3c2a4`, which means a wrong or partial zip.

## Run

1. In the first code cell, set `DATASET` to `01_original`, `02_denoise_clahe`, `03_denoise_clahe_gsam` or `04_denoise_clahe_dino`.
2. Run with `RUN_MODE = "smoke"` first (20 images per class, 2 epochs, small grids). It checks that every cell works in a few minutes. Smoke results go to `results/smoke/`.
3. Set `RUN_MODE = "full"` and *Run All*.
4. Repeat for each dataset.

Anything already trained is saved and reloaded on the next run. Set `RETRAIN = True` to train again.

**Time for one full run (one dataset):** about 2 h on a Colab T4 (SVM grid ≈ 45 min, CNN ≈ 45 min, ResNet-50 ≈ 25 min, the rest ≈ 10 min), and about 3–4 h on an Apple M1.

## Rules that keep the comparison fair

These come from section 0.1 of the notebook.

- **Change only `DATASET`.** Leave every other setting the same for all datasets. If a value must change, change it for every dataset and re-run all of them.
- **Don't make one copy of the notebook per dataset.** Copies drift apart. Keep one notebook and switch `DATASET`.
- Same input pipeline everywhere: pad to square with grey (124,116,104), resize to 224×224, ImageNet normalisation.
- Augmentation on train only: mild random crop and horizontal flip. **No colour jitter**, because colour and contrast are what dataset 02 changes.
- Don't re-split. The folders are the split.

## Outputs

```text
results/<dataset>/                    (results/smoke/<dataset>/ in smoke mode)
├── <model>/predictions_test.csv      per-image predictions, keyed by relpath
├── summary_test.csv                  test metrics of the chosen config per model
├── all_configs_test.csv              every config tried, val and test
├── all_models/                       every trained config, saved as soon as it finishes
├── checkpoints/                      model weights, cached features, CLIP embeddings
└── figures/                          learning curves, confusion matrices, Grad-CAM, ...
```

`<model>` is one of `clip_zeroshot`, `logreg`, `svm`, `cnn`, `resnet50`.

**Section 11 (cross-dataset comparison)** reads `results/<dataset>/<model>/predictions_test.csv` for every dataset it finds, and runs paired tests (McNemar, bootstrap CI of the macro-F1 difference) on the same test images. To use it, put all datasets' result folders under one `results/`. If teammates ran different datasets, copy their `results/<dataset>/` folders in.

Pairs it compares:

| pair | what it measures |
|---|---|
| 01 → 02 | denoise + CLAHE |
| 02 → 03 | Grounded-SAM mask (the research question) |
| 02 → 04 | crop to the DINO box |
| 03 → 04 | box instead of mask |
| 01 → 03, 01 → 04 | everything together |

## Don't commit results or weights

Locally, `results/` and `checkpoints/` are created at the repo root, and ResNet-50 weights are about 100 MB per file (GitHub's limit is 100 MB). Don't `git add` them. Share results through Drive.

Clear notebook outputs before committing (team rule).
