"""Preprocessing pipeline 02: dark-class datasets. Plan and reasons: doc/pipeline/pipeline_02.md.

    python pipeline_preprocess/pipeline_02.py --steps brightness flag   # CPU, then review by hand
    python pipeline_preprocess/pipeline_02.py --steps build             # after review
    python pipeline_preprocess/pipeline_02.py --steps denoise_clahe     # dataset 2, CPU
    python pipeline_preprocess/pipeline_02.py --steps gsam              # dataset 3, GPU (Colab: doc/pipeline/COLAB_GUIDE.md)
    python pipeline_preprocess/pipeline_02.py --steps zip               # one self-contained zip per built dataset

Input   data/raw_data/                     Food-101 as downloaded (any nesting)
Output  data/dataset/
          meta/brightness.csv              mean grey level of every Food-101 image (class selection)
          meta/clip_flags.csv              CLIP + duplicate flags; reviewer fills `decision`
          meta/duplicates.csv              near-duplicate pairs
          meta/review/*_p<n>.jpg           contact sheets of flagged images
          meta/clip_cache.npz              CLIP probabilities + embeddings (re-flag without re-running CLIP)
          meta/removed.csv                 images removed, with reason
          meta/class_counts.csv            images per class after cleaning; N = min
          meta/manifest.csv                relpath,label,split,source (same for every dataset)
          01_original/{train,val,test}/<class>/<id>.jpg
          02_denoise_clahe/{train,val,test}/<class>/<id>.jpg   wavelet denoise -> CLAHE, from 01_original
          03_denoise_clahe_gsam/{train,val,test}/<class>/<id>.jpg   Grounded-SAM on 02: food kept, grey background, cropped
          meta/gsam_boxes.json, meta/gsam_stats.csv   Grounding DINO boxes; mask area + fallback per image
          zips/<dataset>.zip               images + meta/ (manifest, counts, removals) + README_<dataset>.md
Checkpoints  data/checkpoints/             gsam progress, or FOOD_CHECKPOINT_DIR (a Drive folder on Colab)

Steps
  brightness  record mean brightness of all 101 classes (the class list below is fixed, not re-ranked)
  flag        CLIP zero-shot flags on official train, near-duplicates on the whole pool, review sheets
  build          apply review decisions, balance to N per class, split 60/20/20, resize
  denoise_clahe  dataset 2: wavelet denoise (BayesShrink) then CLAHE on L of LAB
  gsam           dataset 3: Grounding DINO ("food.") boxes -> SAM ViT-B mask, on dataset 2 (needs CUDA)
  zip            zip every built dataset so teammates need nothing else (no git)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# =============================================================== config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FOOD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw_data"
OUT_DIR = DATA_DIR / "dataset"
META_DIR = OUT_DIR / "meta"
MANIFEST = META_DIR / "manifest.csv"
FLAGS = META_DIR / "clip_flags.csv"

# 10 classes with the lowest mean brightness (step `brightness`, 2026-11-03). Fixed on purpose.
CLASSES = [
    "bibimbap", "miso_soup", "mussels", "prime_rib", "edamame",
    "baby_back_ribs", "paella", "chicken_wings", "guacamole", "nachos",
]
DARK_LEVEL = 80                   # image mean grey below this counts as "dark" (reported only)

CLIP_ID = "openai/clip-vit-base-patch16"
NONFOOD = ["a person", "text or a menu", "a restaurant interior", "a drink", "a kitchen"]
WRONG_P = 0.05                    # flag when CLIP picks another class and p(label) < this
# CLIP embedding cosine above this = possible near-duplicate, decided in review. Checked by eye
# (2026-11-05): below ~0.975 pairs are different photos (edamame all looks alike), real
# duplicates (re-cropped / colour-shifted) start at ~0.976, above 0.98 almost all are the same photo.
DUP_COS = 0.97

SEED = 42
RATIOS = {"train": 0.6, "val": 0.2}   # test = the rest
STORE_SHORT_SIDE = 256
JPEG_QUALITY = 95

# dataset 2 (doc section 3); same values as analyst/preprocess_compare.ipynb
WAVELET = dict(method="BayesShrink", mode="soft", rescale_sigma=True)
CLAHE_CLIP, CLAHE_TILE = 2.0, 8

# dataset 3 (doc section 4); same models and thresholds as pipeline_01's gsam
DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"
TEXT_PROMPT = "food."
BOX_THR, TEXT_THR = 0.30, 0.25
FILL_RGB = (124, 116, 104)        # ImageNet mean grey -> ~0 after normalisation
MIN_AREA = 0.05                   # mask smaller than this = failed -> keep the dataset-2 image
CROP_MARGIN = 0.05
CKPT_DIR = Path(os.environ.get("FOOD_CHECKPOINT_DIR", DATA_DIR / "checkpoints"))
CKPT_EVERY = 500                  # a dead Colab session loses at most this many images


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# =============================================================== raw data
def _locate(src):
    find = lambda name: glob.glob(os.path.join(src, "**", name), recursive=True)
    train_txt, test_txt = find("train.txt"), find("test.txt")
    img_dirs = [p for p in find("images") if os.path.isdir(os.path.join(p, CLASSES[0]))]
    if not (train_txt and test_txt and img_dirs):
        raise SystemExit(f"Could not find meta/train.txt, test.txt and images/ under {src}")
    return train_txt[0], test_txt[0], img_dirs[0]


def raw_pool(src):
    """[(class, id, source)] for the 10 classes, official train first, sorted."""
    train_txt, test_txt, img_dir = _locate(src)
    rows = []
    for txt, source in [(train_txt, "official_train"), (test_txt, "official_test")]:
        for line in sorted(open(txt).read().split()):
            cls, img_id = line.split("/")
            if cls in CLASSES:
                rows.append((cls, img_id, source))
    return rows, img_dir


# =============================================================== step 1: brightness
def _mean_grey(path):
    im = Image.open(path)
    im.draft("L", (128, 128))      # fast JPEG decode at reduced size; the mean barely changes
    return path.parent.name, path.stem, round(float(np.asarray(im.convert("L")).mean()), 2)


def step_brightness(src):
    out = META_DIR / "brightness.csv"
    if out.exists():
        print("[brightness] exists, skipping")
        return
    _, _, img_dir = _locate(src)
    files = sorted(Path(img_dir).glob("*/*.jpg"))
    with Pool() as pool:
        rows = pool.map(_mean_grey, files, chunksize=256)
    write_csv(out, [dict(zip(["label", "id", "mean_grey"], r)) for r in rows], ["label", "id", "mean_grey"])

    by_class = defaultdict(list)
    for cls, _, v in rows:
        by_class[cls].append(v)
    rank = sorted(by_class, key=lambda c: np.mean(by_class[c]))
    print("[brightness] darkest 12 (* = in CLASSES):")
    for c in rank[:12]:
        v = np.array(by_class[c])
        print(f"  {'*' if c in CLASSES else ' '} {c:16s} mean {v.mean():6.1f}  dark {100 * (v < DARK_LEVEL).mean():4.1f}%")
    if set(rank[:10]) != set(CLASSES):
        print("[brightness] WARNING: CLASSES differ from the current 10 darkest")


# =============================================================== step 2: flag
def _clip_scores(pool, img_dir):
    """Cached in meta/clip_cache.npz, so thresholds can change without re-running CLIP (~35 min on CPU)."""
    cache = META_DIR / "clip_cache.npz"
    prompts = [f"a photo of {c.replace('_', ' ')}, a type of food." for c in CLASSES]
    prompts += [f"a photo of {n}." for n in NONFOOD]
    keys = np.array([f"{c}/{k}" for c, k, _ in pool])
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["keys"], keys) and list(z["prompts"]) == prompts:
            print("[flag] using cached CLIP scores")
            return z["probs"], z["emb"], prompts
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model, proc = CLIPModel.from_pretrained(CLIP_ID).eval(), CLIPProcessor.from_pretrained(CLIP_ID)
    # transformers >= 5 returns an output object with the embedding in pooler_output, older versions a tensor
    unit = lambda out: (lambda x: x / x.norm(dim=-1, keepdim=True))(getattr(out, "pooler_output", out))
    probs, embs = [], []
    with torch.no_grad():
        t = unit(model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True)))
        for i in range(0, len(pool), 64):
            ims = [Image.open(os.path.join(img_dir, c, f"{k}.jpg")).convert("RGB") for c, k, _ in pool[i:i + 64]]
            v = unit(model.get_image_features(**proc(images=ims, return_tensors="pt")))
            probs.append((model.logit_scale.exp() * v @ t.T).softmax(-1).numpy())
            embs.append(v.numpy())
            if i % 1280 == 0:
                print(f"[flag] CLIP {i}/{len(pool)}", flush=True)
    probs, embs = np.concatenate(probs), np.concatenate(embs)
    np.savez(cache, keys=keys, prompts=np.array(prompts), probs=probs, emb=embs)
    return probs, embs, prompts


def _duplicates(pool, emb):
    """Pairs above DUP_COS. Flag the train copy of a train-test pair, else the later id."""
    sim = np.triu(emb @ emb.T, 1)
    pairs = []
    for a, b in zip(*np.where(sim > DUP_COS)):
        src_a, src_b = pool[a][2], pool[b][2]
        drop, keep = (a, b) if src_a == "official_train" and src_b == "official_test" else \
                     (b, a) if src_b == "official_train" and src_a == "official_test" else (b, a)
        pairs.append({"drop": f"{pool[drop][0]}/{pool[drop][1]}", "keep": f"{pool[keep][0]}/{pool[keep][1]}",
                      "drop_source": pool[drop][2], "keep_source": pool[keep][2],
                      "cos": round(float(sim[a, b]), 4)})
    return pairs


def _sheet(items, img_dir, path, tile=160, per_page=100):
    """items: [(class, id, caption)] -> numbered grids, 10 per row, <name>_p1.jpg, _p2.jpg, ..."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(items), per_page), 1):
        chunk = items[start:start + per_page]
        cols, rows = 10, (len(chunk) + 9) // 10
        sheet = Image.new("RGB", (cols * tile, rows * (tile + 28)), "white")
        draw = ImageDraw.Draw(sheet)
        for n, (cls, img_id, cap) in enumerate(chunk):
            im = Image.open(os.path.join(img_dir, cls, f"{img_id}.jpg")).convert("RGB")
            im.thumbnail((tile, tile))
            x, y = (n % cols) * tile, (n // cols) * (tile + 28)
            sheet.paste(im, (x, y))
            draw.text((x + 2, y + tile), f"{start + n + 1}. {img_id}", fill="black")
            draw.text((x + 2, y + tile + 13), cap[:26], fill="red")
        sheet.save(path.with_name(f"{path.stem}_p{page}.jpg"), quality=90)


def step_flag(src):
    if FLAGS.exists():
        print("[flag] clip_flags.csv exists, skipping (delete it to re-flag; this loses review decisions)")
        return
    pool, img_dir = raw_pool(src)
    probs, emb, prompts = _clip_scores(pool, img_dir)
    dups = _duplicates(pool, emb)
    write_csv(META_DIR / "duplicates.csv", dups, ["drop", "keep", "drop_source", "keep_source", "cos"])
    dup_of = {d["drop"]: d["keep"] for d in dups}

    rows = []
    for (cls, img_id, source), p in zip(pool, probs):
        top = int(p.argmax())
        pred = CLASSES[top] if top < len(CLASSES) else "nonfood: " + NONFOOD[top - len(CLASSES)]
        p_label = float(p[CLASSES.index(cls)])
        flag = "nonfood" if top >= len(CLASSES) else "wrong_label" if pred != cls and p_label < WRONG_P else ""
        key = f"{cls}/{img_id}"
        # official test is hand-cleaned: only duplicates send a test image to review
        review = bool(dup_of.get(key)) or (source == "official_train" and flag != "")
        rows.append({"key": key, "label": cls, "id": img_id, "source": source,
                     "clip_pred": pred, "p_label": round(p_label, 4), "clip_flag": flag,
                     "dup_of": dup_of.get(key, ""), "review": int(review),
                     "decision": "" if review else "keep"})
    write_csv(FLAGS, rows, list(rows[0]))

    for cls in CLASSES:
        items = [(r["label"], r["id"], r["clip_pred"]) for r in rows
                 if r["label"] == cls and r["review"] and not r["dup_of"]]
        if items:
            _sheet(items, img_dir, META_DIR / "review" / f"{cls}.jpg")
    pairs = []
    for d in dups:
        for key, tag in [(d["drop"], "DROP?"), (d["keep"], "keep")]:
            c, k = key.split("/")
            pairs.append((c, k, f"{tag} {d['cos']}"))
    if pairs:
        _sheet(pairs, img_dir, META_DIR / "review" / "duplicates.jpg")

    for source in ["official_train", "official_test"]:
        part = [r for r in rows if r["source"] == source]
        c = Counter(r["clip_flag"] for r in part)
        print(f"[flag] {source:15s} n={len(part)}  wrong_label {100 * c['wrong_label'] / len(part):.1f}%"
              f"  nonfood {100 * c['nonfood'] / len(part):.1f}%")
    print(f"[flag] duplicate pairs: {len(dups)}  to review: {sum(r['review'] for r in rows)}")
    print(f"[flag] next: open meta/review/*.jpg, fill `decision` (keep/remove) in {FLAGS}, then --steps build")


# =============================================================== step 3: build
def _resize_short(img, short):
    h, w = img.shape[:2]
    s = short / min(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def step_build(src):
    if MANIFEST.exists():
        print("[build] manifest exists, skipping")
        return
    if not FLAGS.exists():
        raise SystemExit("[build] clip_flags.csv missing: run --steps flag first")
    flags = read_csv(FLAGS)
    bad = [r["key"] for r in flags if r["decision"] not in ("keep", "remove")]
    if bad:
        raise SystemExit(f"[build] STOP: {len(bad)} rows without decision keep/remove, e.g. {bad[:3]}")
    if any(r["decision"] == "remove" and r["source"] == "official_test" and not r["dup_of"] for r in flags):
        raise SystemExit("[build] STOP: official test images may only be removed as duplicates")

    removed = [r for r in flags if r["decision"] == "remove"]
    write_csv(META_DIR / "removed.csv",
              [{"key": r["key"], "source": r["source"], "reason": "duplicate" if r["dup_of"] else r["clip_flag"] or "other"}
               for r in removed], ["key", "source", "reason"])
    kept = defaultdict(list)
    for r in flags:
        if r["decision"] == "keep":
            kept[r["label"]].append(r)
    counts = {c: len(kept[c]) for c in CLASSES}
    n = min(counts.values())
    write_csv(META_DIR / "class_counts.csv",
              [{"label": c, "kept": counts[c], "removed": 1000 - counts[c]} for c in CLASSES],
              ["label", "kept", "removed"])
    sizes = {s: round(n * r) for s, r in RATIOS.items()}
    sizes["test"] = n - sum(sizes.values())
    print(f"[build] kept per class {counts}\n[build] N = {n} -> per class {sizes}")

    rng = random.Random(SEED)
    plan = []
    for cls in CLASSES:
        rows = sorted(kept[cls], key=lambda r: r["key"])
        rng.shuffle(rows)
        rows, start = rows[:n], 0
        for split, size in sizes.items():
            plan += [(split, r) for r in rows[start:start + size]]
            start += size

    _, _, img_dir = _locate(src)
    out, manifest = OUT_DIR / "01_original", []
    for i, (split, r) in enumerate(plan):
        rel = f"{split}/{r['label']}/{r['id']}.jpg"
        img = cv2.imread(os.path.join(img_dir, r["label"], f"{r['id']}.jpg"))
        if img is None:
            raise SystemExit(f"[build] STOP: unreadable {r['key']}")
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / rel), _resize_short(img, STORE_SHORT_SIDE), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        manifest.append({"relpath": rel, "label": r["label"], "split": split, "source": r["source"]})
        if i % 2000 == 0:
            print(f"[build] {i}/{len(plan)}")

    # checks: counts per split/class, no key in two splits. A flagged pair with both images kept
    # was judged "not a duplicate" in review, so it may cross splits.
    got = Counter((m["split"], m["label"]) for m in manifest)
    if any(got[(s, c)] != sizes[s] for s in sizes for c in CLASSES):
        raise SystemExit(f"[build] STOP: unexpected counts {dict(got)}")
    split_of = {f"{m['label']}/{Path(m['relpath']).stem}": m["split"] for m in manifest}
    if len(split_of) != len(manifest):
        raise SystemExit("[build] STOP: an image appears twice")
    write_csv(MANIFEST, manifest, ["relpath", "label", "split", "source"])
    print("[build] done:", dict(Counter(m["split"] for m in manifest)))


# =============================================================== step 4: denoise + CLAHE
def wavelet_denoise(bgr):
    from skimage.restoration import denoise_wavelet
    # convert2ycbcr expects RGB order; denoising Y, Cb, Cr separately avoids colour noise blotches
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    den = denoise_wavelet(rgb, channel_axis=-1, convert2ycbcr=True, **WAVELET)
    return cv2.cvtColor((np.clip(den, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def clahe(bgr):
    # lightness only: equalising R,G,B separately would shift food colours
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _denoise_clahe_one(rel):
    src, dst = OUT_DIR / "01_original" / rel, OUT_DIR / "02_denoise_clahe" / rel
    if dst.exists():                        # resume after an interrupted run
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), clahe(wavelet_denoise(cv2.imread(str(src)))), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def step_denoise_clahe():
    if not MANIFEST.exists():
        raise SystemExit("[denoise_clahe] manifest.csv missing: run --steps build first")
    rels = [m["relpath"] for m in read_csv(MANIFEST)]
    with Pool() as pool:
        for i, _ in enumerate(pool.imap_unordered(_denoise_clahe_one, rels, chunksize=32)):
            if i % 2000 == 0:
                print(f"[denoise_clahe] {i}/{len(rels)}", flush=True)
    out = OUT_DIR / "02_denoise_clahe"
    missing = [r for r in rels if not (out / r).exists()]
    extra = {str(f.relative_to(out)) for f in out.glob("*/*/*.jpg")} - set(rels)
    if missing or extra:
        raise SystemExit(f"[denoise_clahe] STOP: {len(missing)} missing, {len(extra)} not in manifest")
    print(f"[denoise_clahe] done: {len(rels)} images")


# =============================================================== step 5: Grounded-SAM
def apply_mask(rgb, mask):
    """RGB array + bool mask -> (masked and cropped to the food, area, fallback)."""
    area = float(mask.mean()) if mask is not None else 0.0
    if mask is None or area < MIN_AREA:
        return rgb, area, True
    ys, xs = np.where(mask)
    h, w = mask.shape
    my, mx = int(h * CROP_MARGIN), int(w * CROP_MARGIN)
    y0, y1 = max(ys.min() - my, 0), min(ys.max() + my + 1, h)
    x0, x1 = max(xs.min() - mx, 0), min(xs.max() + mx + 1, w)
    masked = np.where(mask[..., None], rgb, np.array(FILL_RGB, dtype=rgb.dtype))
    return masked[y0:y1, x0:x1], area, False


def load_gsam(device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    dino = (AutoProcessor.from_pretrained(DINO_ID),
            AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval())
    sam = (SamProcessor.from_pretrained(SAM_ID), SamModel.from_pretrained(SAM_ID).to(device).eval())
    return dino, sam


def detect_boxes(img, dino, device):
    import torch
    proc, model = dino
    inp = proc(images=img, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    kw = dict(text_threshold=TEXT_THR, target_sizes=[img.size[::-1]])
    try:  # argument name differs across transformers versions
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=BOX_THR, **kw)[0]
    except TypeError:
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=BOX_THR, **kw)[0]
    return [[round(v, 1) for v in b] for b in res["boxes"].tolist()]


def segment(img, boxes, sam, device):
    """Union of the SAM masks of all food boxes, or None when there is no box."""
    import torch
    if not boxes:
        return None
    proc, model = sam
    inp = proc(img, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    m = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
    return m.any(dim=0)[0].numpy().astype(bool)


def _write_atomic(path, write):
    """Only a finished file gets the final name, so a dead session never leaves half a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _part_path(start):
    return CKPT_DIR / f"gsam_{start:05d}.zip"


def _save_part(start, chunk, out):
    def write(tmp):
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:   # JPEG is already compressed
            z.writestr("stats.json", json.dumps(chunk))
            for s in chunk:
                z.write(out / s["relpath"], f"{out.name}/{s['relpath']}")
    _write_atomic(_part_path(start), write)


def _restore_parts():
    """Unpack finished parts in order and return their stats; stops at the first missing or broken part."""
    stats = []
    while True:
        try:
            with zipfile.ZipFile(_part_path(len(stats))) as z:
                if z.testzip() is not None:
                    return stats
                chunk = json.loads(z.read("stats.json"))
                z.extractall(OUT_DIR, [n for n in z.namelist() if n != "stats.json"])
        except (FileNotFoundError, zipfile.BadZipFile):
            return stats
        stats += chunk


def step_gsam(device):
    if (META_DIR / "gsam_stats.csv").exists():
        print("[gsam] gsam_stats.csv exists, skipping")
        return
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("[gsam] no CUDA GPU -> run '--steps gsam' on Colab (doc/pipeline/COLAB_GUIDE.md)")
        return
    src, out = OUT_DIR / "02_denoise_clahe", OUT_DIR / "03_denoise_clahe_gsam"
    rows = read_csv(MANIFEST)
    missing = [r["relpath"] for r in rows if not (src / r["relpath"]).exists()]
    if missing:
        raise SystemExit(f"[gsam] STOP: {len(missing)} dataset-2 images missing, e.g. {missing[:2]}")
    dino, sam = load_gsam(device)
    stats = _restore_parts()
    if stats:
        print(f"[gsam] resumed at {len(stats)}/{len(rows)}")
    for n, r in enumerate(rows):
        if n < len(stats):
            continue
        img = Image.open(src / r["relpath"]).convert("RGB")
        boxes = detect_boxes(img, dino, device)
        result, area, fallback = apply_mask(np.array(img), segment(img, boxes, sam, device))
        (out / r["relpath"]).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(out / r["relpath"], quality=JPEG_QUALITY)
        stats.append({"relpath": r["relpath"], "split": r["split"], "boxes": boxes,
                      "mask_area": round(area, 4), "fallback": int(fallback)})
        if n % 500 == 0:
            print(f"[gsam] {n}/{len(rows)}", flush=True)
        if (n + 1) % CKPT_EVERY == 0 or n + 1 == len(rows):
            start = n // CKPT_EVERY * CKPT_EVERY
            _save_part(start, stats[start:], out)

    _write_atomic(META_DIR / "gsam_boxes.json",
                  lambda tmp: tmp.write_text(json.dumps({s["relpath"]: s["boxes"] for s in stats})))
    write_csv(META_DIR / "gsam_stats.csv",
              [{k: s[k] for k in ("relpath", "split", "mask_area", "fallback")} | {"n_boxes": len(s["boxes"])}
               for s in stats], ["relpath", "split", "n_boxes", "mask_area", "fallback"])
    rate = sum(s["fallback"] for s in stats) / len(stats)
    by_split = {sp: sum(s["fallback"] for s in stats if s["split"] == sp) for sp in ("train", "val", "test")}
    print(f"[gsam] done: {len(stats)} images, fallback rate {rate:.1%} {by_split}")


# =============================================================== step 6: zip for the team
DATASETS = {
    "01_original": "ภาพต้นฉบับที่ resize แล้ว (baseline) ยังไม่ผ่าน preprocessing ใด ๆ",
    "02_denoise_clahe": "dataset 1 + wavelet denoise (BayesShrink) แล้วตามด้วย CLAHE บนช่อง L ของ LAB",
    "03_denoise_clahe_gsam": "dataset 2 + Grounded-SAM: เก็บเฉพาะอาหาร พื้นหลังเป็นสีเทา (124,116,104) แล้ว crop ให้เหลือแต่อาหาร",
}
ZIP_META = ["manifest.csv", "class_counts.csv", "removed.csv"]


def _dataset_readme(name, rows, digest):
    per = Counter((r["split"], r["label"]) for r in rows)
    split = " · ".join(f"{sp} {per[(sp, CLASSES[0])]}" for sp in ("train", "val", "test"))
    total = " · ".join(f"{sp} {sum(v for (s_, _), v in per.items() if s_ == sp):,}" for sp in ("train", "val", "test"))
    lines = [
        f"# {name}", "", DATASETS[name], "",
        f"- {len(rows):,} ภาพ, 10 คลาส: {', '.join(CLASSES)}",
        f"- ต่อคลาส: {split}", f"- รวม: {total}",
        "- ทุก dataset มีภาพชุดเดียวกันใน split เดียวกัน ต่างกันแค่ preprocessing",
        "- ตัด noise (label ผิด, ไม่ใช่อาหาร, ภาพซ้ำ) ออกแล้ว รายการอยู่ใน `meta/removed.csv`", "",
        "## วิธีใช้", "",
        "แตก zip ที่โฟลเดอร์ `data/dataset/` จะได้", "",
        "```", f"data/dataset/{name}/{{train,val,test}}/<class>/<id>.jpg", "data/dataset/meta/manifest.csv   relpath,label,split,source",
        "```", "",
        "ตรวจว่าได้ข้อมูลชุดเดียวกับทุกคน:", "",
        "```", "sha256sum data/dataset/meta/manifest.csv", digest, "```", "",
        "`source` ใน manifest บอกว่าภาพมาจาก Food-101 official train หรือ official test (รวมกันแล้วแบ่งใหม่ 60/20/20, seed 42)", "",
        "## กติกา", "",
        "1. ห้ามแบ่ง train/val/test ใหม่ ใช้ตามโฟลเดอร์",
        "2. ภายในโมเดลเดียวกัน ทุก dataset ต้องใช้ค่าตั้งเดียวกัน",
        "3. เลือก epoch และ hyper-parameter ด้วย val macro-F1 แล้วประเมิน test ครั้งเดียว รายงาน accuracy และ macro P/R/F1", "",
        "## dataset ทั้ง 3 ชุด", "", "| dataset | preprocessing |", "|---|---|",
        *[f"| `{n}` | {d} |" for n, d in DATASETS.items()],
    ]
    if name == "03_denoise_clahe_gsam":
        st = read_csv(META_DIR / "gsam_stats.csv")
        rate = {sp: sum(int(r["fallback"]) for r in st if r["split"] == sp) / max(1, sum(r["split"] == sp for r in st))
                for sp in ("train", "val", "test")}
        lines += ["", "## Fallback", "",
                  "ภาพที่ mask เล็กกว่า 5% ของภาพถือว่า segment ไม่สำเร็จ และใช้ภาพจาก dataset 2 แทน (`fallback = 1` ใน `meta/gsam_stats.csv`)", "",
                  "fallback rate: " + " · ".join(f"{sp} {v:.1%}" for sp, v in rate.items()), "",
                  "mask บางภาพอาจผิดโดยไม่ถูกนับเป็น fallback เช่น ตัดอาหารสีเข้มทิ้งแต่เก็บพื้นหลังไว้ ให้ระวังตอนอ่านผล"]
    lines += ["", "รายละเอียดทั้งหมด: `doc/pipeline/pipeline_02.md` ใน repo", ""]
    return "\n".join(lines)


def step_zip():
    rows = read_csv(MANIFEST)
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    for name in DATASETS:
        d = OUT_DIR / name
        if not d.is_dir():
            print(f"[zip] {name} not built, skipping")
            continue
        meta = ZIP_META + (["gsam_stats.csv"] if name == "03_denoise_clahe_gsam" else [])
        missing = [r["relpath"] for r in rows if not (d / r["relpath"]).exists()]
        missing += [f"meta/{m}" for m in meta if not (META_DIR / m).exists()]
        if missing:
            raise SystemExit(f"[zip] STOP: {name}: {len(missing)} files missing, e.g. {missing[:2]}")

        def write(tmp):
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:   # JPEG is already compressed
                z.writestr(f"README_{name}.md", _dataset_readme(name, rows, digest))
                for m in meta:
                    z.write(META_DIR / m, f"meta/{m}")
                for r in rows:
                    z.write(d / r["relpath"], f"{name}/{r['relpath']}")
        out = OUT_DIR / "zips" / f"{name}.zip"
        _write_atomic(out, write)
        print(f"[zip] {out}  {out.stat().st_size / 1e6:.0f} MB  {len(rows)} images + {len(meta)} meta files")


# =============================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=["brightness", "flag", "build", "denoise_clahe", "gsam"],
                    choices=["brightness", "flag", "build", "denoise_clahe", "gsam", "zip"])
    ap.add_argument("--src", default=str(RAW_DATA_DIR))
    ap.add_argument("--device", default="cuda", help="gsam only; 'cpu' is far too slow for the full set")
    a = ap.parse_args()

    META_DIR.mkdir(parents=True, exist_ok=True)
    if "brightness" in a.steps:
        step_brightness(a.src)
    if "flag" in a.steps:
        step_flag(a.src)
    if "build" in a.steps:
        step_build(a.src)
    if "denoise_clahe" in a.steps:
        step_denoise_clahe()
    if "gsam" in a.steps:
        step_gsam(a.device)
    if "zip" in a.steps:
        step_zip()


if __name__ == "__main__":
    main()
