# model_exploration: 4 other models on the team's datasets

A second set of classifiers, next to `pipeline_model/`. It follows the same **process**
(fixed split, one notebook for every dataset, choose on val, test once, paired tests) but tries **different models**,
one per box in the project diagram. It has its own venv and downloads its own copy of the datasets from the team Drive
into `model_exploration/data/` (ignored by git). Results so far: `doc/analysis/model_exploration.md`.

**Research question:** does segmentation before classification improve food menu prediction accuracy?

## Models

| diagram box | model | tuned on val | team's model (for contrast) |
|---|---|---|---|
| Baseline (zero-shot) | SigLIP base (`google/siglip-base-patch16-224`) | prompt template set | CLIP ViT-B/32 |
| Classic | dense SIFT → bag of visual words + 2×2 spatial pyramid → χ² SVM | codebook size, `C` | HSV + HOG + LBP → LogReg / SVM |
| Transfer learning | EfficientNet-B0, ImageNet weights | how much to unfreeze, lr | ResNet-50 |
| DNN / CNN | ResNet-18 from scratch | width, dropout | plain 3–5 block CNN |

Why these models: see `doc/references/model_exploration.md`.

## Datasets

| dataset | preprocessing |
|---|---|
| `01_original` | resized originals (baseline) |
| `02_denoise_clahe` | 01 + wavelet denoise (BayesShrink) + CLAHE on L of LAB |
| `03_denoise_clahe_gsam` | 02 + Grounded-SAM: food kept, grey background (124,116,104), cropped to the food |

Comparisons: 01 → 02 = denoise + CLAHE; **02 → 03 = segmentation (the research question)**; 01 → 03 = everything.
10 classes × (578 / 193 / 192) = 9,630 images, the same images in every dataset. Manifest sha256 `79eb1d1c…`.

## Setup (Mac)

```bash
cd model_exploration
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The notebook downloads a missing dataset zip from the team Drive (`gdown`) into `data/dataset/_zips/` and unzips it into
`data/dataset/`. If Drive says "too many accesses", download the zip by hand into `data/dataset/_zips/` and re-run.

## Run

1. Open `explore_models.ipynb` with the `.venv` kernel.
2. Set `DATASET` and `RUN_MODE = "smoke"` in the first code cell, then *Run All* (20 images per class, 2 epochs: checks every cell works).
3. Set `RUN_MODE = "full"` and *Run All*. Repeat for `01_original`, `02_denoise_clahe`, `03_denoise_clahe_gsam`.

Or from the terminal, writing outputs into the notebook:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 explore_models.ipynb
```

**Colab:** upload the notebook, *Runtime → Change runtime type → T4 GPU*, *Run All*. Data goes to `/content/data/dataset`, results to
`MyDrive/food_explore/results/`. To merge with Mac results, copy `results/<dataset>/` folders into one `results/`.

Everything trained is saved and reloaded on a re-run. Set `RETRAIN = True` to train again.

## Rules (same as the team's)

1. Don't re-split. The folders are the split.
2. Change only `DATASET` between runs. If a setting must change, change it for every dataset and re-run all of them.
3. Choose on val macro-F1, evaluate test once. Report accuracy + macro P/R/F1.
4. One seed only: read the bootstrap CI in the cross-dataset section before calling a gap real.

## Outputs

```text
results/<dataset>/                    (results/smoke/<dataset>/ in smoke mode)
├── <model>/predictions_test.csv      per-image predictions, keyed by relpath
├── <model>/failures.csv              every wrong test image with its "why" tags
├── summary_test.csv                  test metrics of the chosen config per model
├── all_configs_test.csv              every config tried, val and test
├── all_models/                       every trained config, saved as soon as it finishes
├── checkpoints/                      weights, cached features and embeddings
└── figures/                          learning curves, confusion matrices, Grad-CAM, failure galleries
results/paired_tests.csv              01→02, 02→03, 01→03 per model (McNemar + bootstrap CI)
results/flips_02_03.csv               test images that change from right to wrong (or back) when segmentation is added
```

`<model>` is one of `siglip_zeroshot`, `bovw_svm`, `effnet_b0`, `resnet18_scratch`.

`results/` is not committed (about 70 MB of figures and executed notebooks per dataset, plus model weights); share it via Drive.
The small tables behind `doc/analysis/model_exploration.md` are committed in `reported_results/`:
`summary_test_<dataset>.csv` (test metrics of each chosen config) and `paired_tests.csv`.
