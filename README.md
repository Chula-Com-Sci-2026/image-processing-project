# Does Segmentation Before Classification Improve Food Menu Prediction Accuracy?

2110651 Digital Image Processing, team project. Full protocol is in `CLAUDE.md`.

**Idea:** the data engineer builds 3 ready-to-use datasets. Everyone else points their own training code at `data/dataset/<dataset>/` and runs. Models are trained outside this workspace.

## Folder structure
```
image-processing-project/
├── CLAUDE.md                        # full protocol (for humans and Claude Code)
├── README.md
├── requirements.txt
├── .gitignore
│
├── pipeline_preprocess/
│   ├── pipeline_02.py               # builds data/dataset/
│   └── pipeline_01.py               # old process (replaced 2026-11-03)
│
├── doc/
│   ├── pipeline/pipeline_02.md      # the current data process, step by step
│   ├── analysis/                    # preprocess_metrics.md: what each preprocessing step does
│   ├── references/                  # references_map.md: textbook ↔ papers
│   └── legacy/                      # old process (pipeline_01) docs
│
├── analyst/                         # notebooks for viewing images, masks, stats
│
└── data/                            # not committed
    ├── raw_data/                    #   Food-101 as downloaded (Kaggle kmader/food41 v5)
    ├── preprocessing_data/          #   old process output
    └── dataset/                     #   what the team uses
        ├── meta/                    #   manifest.csv, removed.csv, class_counts.csv, ...
        ├── 01_original/
        ├── 02_denoise_clahe/
        └── 03_denoise_clahe_gsam/
            └── {train,val,test}/<class>/<id>.jpg   # same images in every dataset
```

## Data you get
- 10 classes with the darkest images in Food-101: bibimbap, miso_soup, mussels, prime_rib, edamame, baby_back_ribs, paella, chicken_wings, guacamole, nachos
- Noisy images (wrong label, no food, duplicates) removed by hand, listed in `meta/removed.csv`
- N images per class (N = the smallest class after cleaning, **PENDING**), split 60 / 20 / 20 train / val / test
- Every dataset folder holds **the same images in the same split**. Only the preprocessing differs.

| dataset | preprocessing |
|---|---|
| `01_original` | resized photo (baseline) |
| `02_denoise_clahe` | `01_original` + wavelet denoise, then CLAHE on the lightness channel |
| `03_denoise_clahe_gsam` | `02_denoise_clahe` + Grounded-SAM: food masked, grey background, cropped to food |

## Only 3 rules
1. **Don't re-split.** Use the `train/ val/ test/` folders as given.
2. **For your model, run every dataset with the same settings.** Change only the dataset folder.
3. **Choose epochs and hyper-parameters on `val`; evaluate `test` once.** Report accuracy and macro P/R/F1.

## Quick start
```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python pipeline_preprocess/pipeline_02.py   # data engineer only
sha256sum data/dataset/meta/manifest.csv              # same hash for everyone = same split
```
