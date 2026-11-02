# Does Segmentation Before Classification Improve Food Menu Prediction Accuracy?

2110651 Digital Image Processing, team project. Full protocol is in `CLAUDE.md`.

**Idea:** the data engineer prepares every preprocessing variant as a ready-to-use folder.
Everyone else just points their own notebook at `data/preprocessing_data/<variant>_data/` and runs.

## Folder structure
```
image-processing-project/
├── CLAUDE.md                        # full protocol (for humans and Claude Code)
├── README.md
├── requirements.txt
├── .gitignore
│
├── pipeline_preprocess/
│   └── pipeline_01.py               # the whole preprocessing in one script:
│                                    #   prepare → light (CLAHE, hist-eq) → segment (GPU)
│
├── model_pipeline/                  # one notebook per model family, paths preset
│   ├── 01_neural_network.ipynb      #   MLP on HSV + HOG + LBP features (classical side)
│   ├── 02_cnn.ipynb                 #   ResNet-50, ImageNet-pretrained, fine-tuned
│   └── 03_transformer.ipynb         #   DeiT-Small, ImageNet-pretrained, fine-tuned
│
├── data/                            # not committed (except manifest.csv)
│   ├── raw_data/                    #   Food-101 as downloaded (Kaggle kmader/food41 v5)
│   └── preprocessing_data/          #   everything pipeline_01.py produces
│       ├── manifest.csv             #   which image is in which split
│       ├── boxes.json, <seg>_stats.csv
│       ├── original_data/           #   baseline, input of every other step
│       ├── clahe_data/
│       ├── histeq_data/
│       ├── gsam_data/   gsam_crop_data/
│       ├── gsam_light_data/   gsam_light_crop_data/
│       └── sam_data/   sam_crop_data/
│           └── {train,val,test}/<class>/<id>.jpg   # same images in every variant
│
└── results/                         # written by the notebooks
    ├── neural_network/
    ├── cnn/
    └── transformer/
        ├── <variant>__<model>__s<seed>.json        # test metrics + settings
        └── <variant>__<model>__s<seed>_preds.csv   # per-image predictions
```

## Data you get
- 10 classes, 5 look-alike pairs: ramen/pho, steak/filet_mignon, fried_rice/pad_thai, chocolate_cake/chocolate_mousse, caesar_salad/greek_salad
- train 6,750 · val 750 · test 2,500 (675 / 75 / 250 per class), already split into folders
- Every variant folder holds **the same images in the same split**. Only the preprocessing differs.

| variant | preprocessing |
|---|---|
| `original` | resized photo (baseline) |
| `clahe`, `histeq` | light correction on the lightness channel |
| `gsam`, `gsam_light`, `sam` | food masked, grey background, cropped to food |
| `gsam_crop`, `gsam_light_crop`, `sam_crop` | cropped to food, background kept |

## Only 3 rules
1. **Don't re-split.** Use the `train/ val/ test/` folders as given.
2. **For your model, run every variant with the same settings.** Change only `VARIANT`.
3. **Choose epochs and hyper-parameters on `val`; evaluate `test` once.** Report accuracy and macro P/R/F1.

## Quick start
```bash
pip install -r requirements.txt
python pipeline_preprocess/pipeline_01.py                   # data engineer: builds data/preprocessing_data/
python pipeline_preprocess/pipeline_01.py --steps segment   # on Colab (GPU) for the segmentation variants
sha256sum data/preprocessing_data/manifest.csv              # same hash for everyone = same split
```
Every step skips work that already exists, so re-running is safe.

Everyone else: open the notebook for your model in `model_pipeline/`, change only `VARIANT`, and Run All.
Paths are preset (the notebook finds the project root by itself). On Colab, uncomment the Drive paths in the paths cell.

Next preprocessing idea? Add `pipeline_preprocess/pipeline_02.py` that writes new `<name>_data/` folders into `data/preprocessing_data/`.
