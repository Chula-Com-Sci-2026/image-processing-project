"""CLIP zero-shot on the test split of each dataset: which preprocessing helps CLIP most?

No training: CLIP picks the class whose text prompt is closest to the image. Same model,
same prompts, same 1,920 test images for every dataset, so the only thing that changes is
the preprocessing. Method and caveats: doc/analysis/clip_zeroshot.md.

    .venv/bin/python analyst/clip_zeroshot.py              # every dataset found in data/dataset/
    .venv/bin/python analyst/clip_zeroshot.py --datasets 01_original 02_denoise_clahe

Predictions are cached per dataset in analyst/results/clip_zeroshot/<dataset>_test.csv,
so adding dataset 3 later only scores dataset 3. Delete a csv to re-score it.
"""
import argparse
import csv
import sys
import time
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline_preprocess"))
from pipeline_02 import CLASSES, CLIP_ID, MANIFEST, OUT_DIR  # one source of truth for classes/model

DATASETS = ["01_original", "02_denoise_clahe", "03_denoise_clahe_gsam"]
# same wording as the flag step, without the non-food prompts: a plain 10-way choice
PROMPTS = [f"a photo of {c.replace('_', ' ')}, a type of food." for c in CLASSES]
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "clip_zeroshot"
BATCH = 64


def load_test():
    with open(MANIFEST) as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def score(ds, rows):
    """Run CLIP on one dataset, return rows with pred + p_label. Cached as csv."""
    out = RESULTS_DIR / f"{ds}_test.csv"
    if out.exists():
        print(f"[{ds}] using cached {out.name}")
        with open(out) as f:
            return list(csv.DictReader(f))
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model, proc = CLIPModel.from_pretrained(CLIP_ID).eval(), CLIPProcessor.from_pretrained(CLIP_ID)
    # transformers >= 5 returns an output object (pooler_output), older versions a tensor
    unit = lambda o: (lambda x: x / x.norm(dim=-1, keepdim=True))(getattr(o, "pooler_output", o))
    probs, t0 = [], time.time()
    with torch.no_grad():
        txt = unit(model.get_text_features(**proc(text=PROMPTS, return_tensors="pt", padding=True)))
        for i in range(0, len(rows), BATCH):
            ims = [Image.open(OUT_DIR / ds / r["relpath"]).convert("RGB") for r in rows[i:i + BATCH]]
            img = unit(model.get_image_features(**proc(images=ims, return_tensors="pt")))
            probs.append((model.logit_scale.exp() * img @ txt.T).softmax(-1).numpy())
            print(f"[{ds}] {min(i + BATCH, len(rows))}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    probs = np.concatenate(probs)
    res = [{"relpath": r["relpath"], "label": r["label"], "source": r["source"],
            "pred": CLASSES[p.argmax()], "p_label": f"{p[CLASSES.index(r['label'])]:.4f}"}
           for r, p in zip(rows, probs)]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    return res


def metrics(res):
    """Accuracy and macro precision / recall / F1 (a class never predicted gets precision 0)."""
    y = np.array([r["label"] for r in res])
    p = np.array([r["pred"] for r in res])
    P, R, F = [], [], []
    for c in CLASSES:
        tp = np.sum((p == c) & (y == c))
        prec = tp / max(np.sum(p == c), 1)
        rec = tp / max(np.sum(y == c), 1)
        P.append(prec)
        R.append(rec)
        F.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc = np.mean(p == y)
    return {"n": len(y), "acc": acc, "se": np.sqrt(acc * (1 - acc) / len(y)),
            "P": np.mean(P), "R": np.mean(R), "F1": np.mean(F), "per_class_f1": dict(zip(CLASSES, F))}


def mcnemar(a, b):
    """Exact two-sided McNemar on paired correctness. Returns (only_a_right, only_b_right, p)."""
    ca = np.array([r["pred"] == r["label"] for r in a])
    cb = np.array([r["pred"] == r["label"] for r in b])
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p


def report(results):
    subsets = {"all test (1,920)": lambda r: True,
               "official_test only (not CLIP-filtered)": lambda r: r["source"] == "official_test"}
    lines = [f"CLIP zero-shot ({CLIP_ID}), test split\n"]
    for name, keep in subsets.items():
        lines += [f"## {name}\n", "| dataset | n | accuracy | ±SE | macro P | macro R | macro F1 |",
                  "|---|---|---|---|---|---|---|"]
        sub = {ds: [r for r in res if keep(r)] for ds, res in results.items()}
        for ds, res in sub.items():
            m = metrics(res)
            lines.append(f"| {ds} | {m['n']} | {100 * m['acc']:.2f} | {100 * m['se']:.2f} | "
                         f"{100 * m['P']:.2f} | {100 * m['R']:.2f} | {100 * m['F1']:.2f} |")
        lines += ["", "McNemar (exact, paired): `A only` = images A gets right and B gets wrong\n",
                  "| A vs B | A only | B only | p |", "|---|---|---|---|"]
        for a, b in combinations(sub, 2):
            n01, n10, p = mcnemar(sub[a], sub[b])
            lines.append(f"| {a} vs {b} | {n01} | {n10} | {p:.4f} |")
        lines.append("")
    lines += ["## per-class F1, all test\n", "| class | " + " | ".join(results) + " |",
              "|---|" + "---|" * len(results)]
    pcf = {ds: metrics(res)["per_class_f1"] for ds, res in results.items()}
    for c in CLASSES:
        lines.append(f"| {c} | " + " | ".join(f"{100 * pcf[ds][c]:.1f}" for ds in results) + " |")
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary.md").write_text(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, help="default: every dataset folder that exists")
    args = ap.parse_args()
    todo = args.datasets or [d for d in DATASETS if (OUT_DIR / d).is_dir()]
    rows = load_test()
    print(f"{len(rows)} test images, datasets: {todo}")
    report({ds: score(ds, rows) for ds in todo})


if __name__ == "__main__":
    main()
