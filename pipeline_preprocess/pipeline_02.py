"""Preprocessing pipeline 02: dark-class datasets. Plan and reasons: pipeline_02.md.

    python pipeline_preprocess/pipeline_02.py --steps brightness flag   # CPU, then review by hand
    python pipeline_preprocess/pipeline_02.py --steps build             # after review

Input   data/raw_data/                     Food-101 as downloaded (any nesting)
Output  data/dataset/
          meta/brightness.csv              mean grey level of every Food-101 image (class selection)
          meta/clip_flags.csv              CLIP + duplicate flags; reviewer fills `decision`
          meta/duplicates.csv              near-duplicate pairs
          meta/review/*.jpg                contact sheets of flagged images
          meta/removed.csv                 images removed, with reason
          meta/class_counts.csv            images per class after cleaning; N = min
          meta/manifest.csv                relpath,label,split,source (same for every dataset)
          01_original/{train,val,test}/<class>/<id>.jpg

Steps
  brightness  record mean brightness of all 101 classes (the class list below is fixed, not re-ranked)
  flag        CLIP zero-shot flags on official train, near-duplicates on the whole pool, review sheets
  build       apply review decisions, balance to N per class, split 60/20/20, resize
"""
import argparse
import csv
import glob
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
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
DUP_COS = 0.95                    # CLIP embedding cosine above this = near-duplicate

SEED = 42
RATIOS = {"train": 0.6, "val": 0.2}   # test = the rest
STORE_SHORT_SIDE = 256
JPEG_QUALITY = 95


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
    import torch
    from transformers import CLIPModel, CLIPProcessor
    prompts = [f"a photo of {c.replace('_', ' ')}, a type of food." for c in CLASSES]
    prompts += [f"a photo of {n}." for n in NONFOOD]
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
    return np.concatenate(probs), np.concatenate(embs), prompts


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


def _sheet(items, img_dir, path, tile=160):
    """items: [(class, id, caption)] -> numbered grid, 10 per row."""
    cols, rows = 10, (len(items) + 9) // 10
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (cls, img_id, cap) in enumerate(items):
        im = Image.open(os.path.join(img_dir, cls, f"{img_id}.jpg")).convert("RGB")
        im.thumbnail((tile, tile))
        x, y = (n % cols) * tile, (n // cols) * (tile + 28)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + tile), f"{n + 1}. {img_id}", fill="black")
        draw.text((x + 2, y + tile + 13), cap[:26], fill="red")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


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

    # checks: counts per split/class, no key in two splits, no kept duplicate pair across splits
    got = Counter((m["split"], m["label"]) for m in manifest)
    if any(got[(s, c)] != sizes[s] for s in sizes for c in CLASSES):
        raise SystemExit(f"[build] STOP: unexpected counts {dict(got)}")
    split_of = {f"{m['label']}/{Path(m['relpath']).stem}": m["split"] for m in manifest}
    if len(split_of) != len(manifest):
        raise SystemExit("[build] STOP: an image appears twice")
    for d in read_csv(META_DIR / "duplicates.csv"):
        a, b = split_of.get(d["drop"]), split_of.get(d["keep"])
        if a and b and a != b:
            raise SystemExit(f"[build] STOP: duplicate pair across splits {d['drop']} ({a}) / {d['keep']} ({b})")
    write_csv(MANIFEST, manifest, ["relpath", "label", "split", "source"])
    print("[build] done:", dict(Counter(m["split"] for m in manifest)))


# =============================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=["brightness", "flag", "build"],
                    choices=["brightness", "flag", "build"])
    ap.add_argument("--src", default=str(RAW_DATA_DIR))
    a = ap.parse_args()

    META_DIR.mkdir(parents=True, exist_ok=True)
    if "brightness" in a.steps:
        step_brightness(a.src)
    if "flag" in a.steps:
        step_flag(a.src)
    if "build" in a.steps:
        step_build(a.src)


if __name__ == "__main__":
    main()
