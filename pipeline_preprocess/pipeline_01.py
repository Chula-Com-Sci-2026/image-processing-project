"""Preprocessing pipeline 01: Food-101 subset -> every preprocessing variant.

    python pipeline_preprocess/pipeline_01.py                      # all steps, skips what exists
    python pipeline_preprocess/pipeline_01.py --steps prepare light
    python pipeline_preprocess/pipeline_01.py --steps segment      # GPU (Colab)

Input   data/raw_download/                 Food-101 as downloaded (any nesting)
Output  data/preprocessing_data/
          manifest.csv                     relpath,label,split (same for every variant)
          original_data/                   resized photo, baseline + input of every other step
          clahe_data/  histeq_data/        light correction
          <seg>_data/                      food masked, grey background, cropped to food
          <seg>_crop_data/                 cropped to food, background kept
          boxes.json  <seg>_stats.csv      Grounding DINO boxes, mask area + fallback per image
        each variant: {train,val,test}/<class>/<id>.jpg

Steps
  prepare   pick 10 classes, split (official train -> train/val, official test), resize
  light     CLAHE and global hist-eq on the lightness channel
  segment   Grounding DINO + SAM / SlimSAM, and SAM with a centre point (needs CUDA)
"""
import argparse
import csv
import glob
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# =============================================================== config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FOOD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DOWNLOAD_DIR = DATA_DIR / "raw_download"
OUT_DIR = DATA_DIR / "preprocessing_data"
MANIFEST = OUT_DIR / "manifest.csv"

# 10 classes = 5 look-alike pairs (some differ by container, some by the food itself)
CLASSES = [
    "ramen", "pho",
    "steak", "filet_mignon",
    "fried_rice", "pad_thai",
    "chocolate_cake", "chocolate_mousse",
    "caesar_salad", "greek_salad",
]
SPLIT_SEED = 42
VAL_FRACTION = 0.10
EXPECTED_COUNTS = {"train": 6750, "val": 750, "test": 2500}
STORE_SHORT_SIDE = 256
JPEG_QUALITY = 95

FILL_RGB = (124, 116, 104)    # ImageNet mean grey -> ~0 after normalisation
MIN_AREA = 0.05               # mask smaller than this = failed -> keep original
CROP_MARGIN = 0.05
DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_IDS = {"sam": "facebook/sam-vit-base",
           "gsam": "facebook/sam-vit-base",
           "gsam_light": "Zigeng/SlimSAM-uniform-77"}
TEXT_PROMPT = "food."
BOX_THR, TEXT_THR = 0.30, 0.25


def variant_dir(name):
    return OUT_DIR / f"{name}_data"


def read_manifest():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def write_jpg(path, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


# =============================================================== step 1: prepare
def _locate(src):
    find = lambda name: glob.glob(os.path.join(src, "**", name), recursive=True)
    train_txt, test_txt = find("train.txt"), find("test.txt")
    img_dirs = [p for p in find("images") if os.path.isdir(os.path.join(p, CLASSES[0]))]
    if not (train_txt and test_txt and img_dirs):
        raise SystemExit(f"Could not find meta/train.txt, test.txt and images/ under {src}")
    return train_txt[0], test_txt[0], img_dirs[0]


def _read_split(txt):
    by_class = defaultdict(list)
    for line in open(txt):
        line = line.strip()
        if line:
            cls, img_id = line.split("/")
            if cls in CLASSES:
                by_class[cls].append(img_id)
    return by_class


def _resize_short(img, short):
    h, w = img.shape[:2]
    s = short / min(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def step_prepare(src):
    if MANIFEST.exists():
        print("[prepare] manifest exists, skipping")
        return
    train_txt, test_txt, img_dir = _locate(src)
    rng = random.Random(SPLIT_SEED)
    rows = []
    for cls, ids in sorted(_read_split(train_txt).items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n_val = round(len(ids) * VAL_FRACTION)
        rows += [(cls, i, "val") for i in ids[:n_val]] + [(cls, i, "train") for i in ids[n_val:]]
    for cls, ids in sorted(_read_split(test_txt).items()):
        rows += [(cls, i, "test") for i in sorted(ids)]

    out, manifest = variant_dir("original"), []
    for n, (cls, img_id, split) in enumerate(rows):
        rel = f"{split}/{cls}/{img_id}.jpg"
        img = cv2.imread(os.path.join(img_dir, cls, f"{img_id}.jpg"))
        if img is None:
            print("[prepare] skip unreadable", rel)
            continue
        write_jpg(out / rel, _resize_short(img, STORE_SHORT_SIDE))
        manifest.append({"relpath": rel, "label": cls, "split": split})
        if n % 1000 == 0:
            print(f"[prepare] {n}/{len(rows)}")

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["relpath", "label", "split"])
        w.writeheader()
        w.writerows(manifest)
    got = dict(Counter(r["split"] for r in manifest))
    print("[prepare] done:", got)
    if got != EXPECTED_COUNTS:
        raise SystemExit(f"[prepare] STOP: expected {EXPECTED_COUNTS}. Check 'skip unreadable' lines.")


# =============================================================== step 2: light correction
def clahe(img):
    # lightness only: equalising R,G,B separately would shift food colours
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def histeq(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def step_light():
    src, rows = variant_dir("original"), read_manifest()
    for name, fn in {"clahe": clahe, "histeq": histeq}.items():
        if variant_dir(name).exists():
            print(f"[{name}] exists, skipping")
            continue
        for n, r in enumerate(rows):
            write_jpg(variant_dir(name) / r["relpath"], fn(cv2.imread(str(src / r["relpath"]))))
            if n % 2000 == 0:
                print(f"[{name}] {n}/{len(rows)}")
        print(f"[{name}] done")


# =============================================================== step 3: segmentation
def apply_mask(arr, mask):
    """RGB array + bool mask -> (masked+cropped, crop-only, area, fallback)."""
    area = float(mask.mean()) if mask is not None else 0.0
    if mask is None or area < MIN_AREA:
        return arr, arr, area, True
    ys, xs = np.where(mask)
    h, w = mask.shape
    my, mx = int(h * CROP_MARGIN), int(w * CROP_MARGIN)
    y0, y1 = max(ys.min() - my, 0), min(ys.max() + my + 1, h)
    x0, x1 = max(xs.min() - mx, 0), min(xs.max() + mx + 1, w)
    masked = np.where(mask[..., None], arr, np.array(FILL_RGB, dtype=arr.dtype))
    return masked[y0:y1, x0:x1], arr[y0:y1, x0:x1], area, False


def _detect_boxes(device):
    """Grounding DINO once; boxes are shared by gsam and gsam_light (only the mask model differs)."""
    path = OUT_DIR / "boxes.json"
    if path.exists():
        return json.load(open(path))
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(DINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()
    src, rows, boxes = variant_dir("original"), read_manifest(), {}
    for n, r in enumerate(rows):
        img = Image.open(src / r["relpath"]).convert("RGB")
        inp = proc(images=img, text=TEXT_PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        kw = dict(text_threshold=TEXT_THR, target_sizes=[img.size[::-1]])
        try:  # argument name differs across transformers versions
            res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=BOX_THR, **kw)[0]
        except TypeError:
            res = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=BOX_THR, **kw)[0]
        boxes[r["relpath"]] = [[round(v, 1) for v in b] for b in res["boxes"].tolist()]
        if n % 500 == 0:
            print(f"[boxes] {n}/{len(rows)}")
    json.dump(boxes, open(path, "w"))
    return boxes


def _segment_mode(mode, device, boxes):
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor
    proc = SamProcessor.from_pretrained(SAM_IDS[mode])
    sam = SamModel.from_pretrained(SAM_IDS[mode]).to(device).eval()
    src, rows, stats = variant_dir("original"), read_manifest(), []
    for n, r in enumerate(rows):
        img = Image.open(src / r["relpath"]).convert("RGB")
        w, h = img.size
        mask = None
        if mode == "sam":   # no text guidance: centre point, keep the most confident of 3 masks
            inp = proc(img, input_points=[[[w / 2, h / 2]]], return_tensors="pt").to(device)
            with torch.no_grad():
                out = sam(**inp, multimask_output=True)
            m = proc.image_processor.post_process_masks(
                out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
            mask = m[0, out.iou_scores.cpu()[0, 0].argmax()].numpy().astype(bool)
        elif boxes[r["relpath"]]:
            inp = proc(img, input_boxes=[boxes[r["relpath"]]], return_tensors="pt").to(device)
            with torch.no_grad():
                out = sam(**inp, multimask_output=False)
            m = proc.image_processor.post_process_masks(
                out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
            mask = m.any(dim=0)[0].numpy().astype(bool)   # union of all food boxes

        masked, cropped, area, fallback = apply_mask(np.array(img), mask)
        for name, arr in ((mode, masked), (mode + "_crop", cropped)):
            p = variant_dir(name) / r["relpath"]
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(p, quality=JPEG_QUALITY)
        stats.append({"relpath": r["relpath"], "split": r["split"],
                      "mask_area": round(area, 4), "fallback": int(fallback)})
        if n % 500 == 0:
            print(f"[{mode}] {n}/{len(rows)}")

    with open(OUT_DIR / f"{mode}_stats.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(stats[0]))
        wr.writeheader()
        wr.writerows(stats)
    print(f"[{mode}] done, fallback rate {sum(s['fallback'] for s in stats) / len(stats):.1%}")


def step_segment(modes):
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not has_cuda:
        print("[segment] no CUDA GPU -> run '--steps segment' on Colab")
        return
    todo = [m for m in modes if not (OUT_DIR / f"{m}_stats.csv").exists()]
    if not todo:
        print("[segment] all modes exist, skipping")
        return
    boxes = _detect_boxes("cuda") if any(m != "sam" for m in todo) else None
    for m in todo:
        _segment_mode(m, "cuda", boxes)


# =============================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=["prepare", "light", "segment"],
                    choices=["prepare", "light", "segment"])
    ap.add_argument("--seg-modes", nargs="+", default=["gsam", "gsam_light", "sam"],
                    choices=list(SAM_IDS))
    ap.add_argument("--src", default=str(RAW_DOWNLOAD_DIR))
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "prepare" in a.steps:
        step_prepare(a.src)
    if not MANIFEST.exists():
        raise SystemExit("manifest.csv missing: run the prepare step first")
    if "light" in a.steps:
        step_light()
    if "segment" in a.steps:
        step_segment(a.seg_modes)
    print("variants:", sorted(p.name for p in OUT_DIR.glob("*_data")))


if __name__ == "__main__":
    main()
