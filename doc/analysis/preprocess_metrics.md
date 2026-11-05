# Preprocessing metrics

Image statistics for each preprocessing candidate, measured on the same 10 dark images. Source: `analyst/preprocess_compare.ipynb` (2026-11-05).

These numbers show what each step does to the pixels. They were used only to rule out clearly bad steps. They do **not** say which step classifies best: see `doc/pipeline/pipeline_02.md`, section 3.

## Setup

| item | value |
|---|---|
| Images | 10, one per class, drawn with `SEED = 0` |
| Pool | official train only, mean grey < 80 (dark) |
| Size | resized to short side 256 px, like `01_original` |
| Channel | every step except wavelet changes only L of LAB, so food colours (a, b) stay the same |
| Values | mean over the 10 images |

## Steps

| step | in the lecture notes (PDF page) | parameters |
|---|---|---|
| contrast stretch | taught: §3.3.4 (p.50) | 1st–99th percentile of L → 0–255 |
| hist-eq | taught: §3.4 (p.55–58) | global equalization of L |
| CLAHE | taught: §3.4.2 (p.58–60) | clip 2.0, 8×8 tiles |
| gamma | taught: §3.3.3 (p.49) | `s = r^γ`, γ per image so mean L → 0.5, γ in [0.4, 1] |
| unsharp | named only, as a frequency-domain example (p.97) | `L + 1·(L − GaussianBlur(L, σ=2))` |
| high-boost | exercise 8 (p.100), with solution code (p.393) | `L + 2·(L − GaussianBlur(L, σ=2))` |
| wavelet | idea only: one line, "wavelet thresholding removes high-frequency noise" (p.146) | BayesShrink, soft threshold, YCbCr, `rescale_sigma=True` |

`A → B` means A first, then B.

The notes don't cover the wavelet details. Cite these instead:
- Soft thresholding: D. L. Donoho, "De-noising by soft-thresholding," *IEEE Trans. Information Theory*, 41(3), 1995.
- BayesShrink (picks a threshold per wavelet sub-band from the estimated noise and signal variance): S. G. Chang, B. Yu, M. Vetterli, "Adaptive wavelet thresholding for image denoising and compression," *IEEE Trans. Image Processing*, 9(9), 2000.
- Implementation: `skimage.restoration.denoise_wavelet` (scikit-image).

The high-boost solution on p.393 also notes that high-boost amplifies noise, which matches the results below.

## Metrics

All metrics except noise σ are computed on L of LAB (0–255).

| metric | code | what it measures | higher means |
|---|---|---|---|
| Brightness | `L.mean()` | overall lightness | brighter image |
| Contrast | `L.std()` (RMS contrast) | how widely lightness spreads over the whole image | more global contrast |
| Laplacian variance | `cv2.Laplacian(L, cv2.CV_64F).var()` | high-frequency energy: edges, texture **and noise** | more edges or more noise |
| Noise σ | `skimage.restoration.estimate_sigma(rgb, channel_axis=-1, average_sigmas=True)` | estimated Gaussian noise level, mean over RGB | more noise |

### How to read them

**Laplacian variance**
- The Laplacian is the second derivative, with the 3×3 kernel `[[0,1,0],[1,-4,1],[0,1,0]]`. It is about 0 on flat areas and large at edges.
- The variance is taken over all pixels. The mean would be useless, because positive and negative values cancel out.
- It cannot tell real edges from noise grains. A high value is not "sharper is better". Always read it together with noise σ.
- It grows with the **square** of contrast: stretching lightness 2× makes it 4× larger. So contrast stretch and hist-eq raise it without adding any new detail.
- Its unit is (grey level)². Compare values between steps only, not as absolute numbers.

**Noise σ**
- Estimated from the finest wavelet band with the median absolute deviation, which is robust to edges. It separates noise from edges better than Laplacian variance does.
- On JPEG photos it also picks up some fine texture, so it is an estimate, not ground truth.

## Results

| step | brightness | contrast | Laplacian variance | noise σ |
|---|---|---|---|---|
| original | 64.6 | 55.8 | 613 | 1.50 |
| contrast stretch | 79.9 | 72.1 | 918 | 2.26 |
| hist-eq | 127.0 | 72.4 | 1505 | 3.51 |
| CLAHE | 83.5 | 62.1 | 1334 | 3.15 |
| gamma | 109.6 | 59.2 | 650 | 2.24 |
| unsharp | 65.0 | 59.5 | 2114 | 3.13 |
| high-boost | 65.7 | 62.6 | 3914 | 4.30 |
| stretch → CLAHE | 93.9 | 71.1 | 1672 | 3.75 |
| unsharp → CLAHE | 84.7 | 64.4 | 3750 | 5.64 |
| CLAHE → unsharp | 83.9 | 68.5 | 4637 | 6.03 |
| high-boost → CLAHE | 84.6 | 67.1 | 6279 | 7.59 |
| wavelet → CLAHE | 83.3 | 62.2 | 1199 | 2.36 |

## Findings

- **CLAHE doubles the noise** (σ 1.50 → 3.15) while brightening dark images (64.6 → 83.5).
- **Wavelet before CLAHE** cuts that noise by about half (3.15 → 2.36) with the same brightness. Part of the drop is lost fine texture (Laplacian variance 1334 → 1199).
- **Stretch before CLAHE** gives the brightest and highest-contrast CLAHE result (93.9, 71.1), with a little more noise (3.75).
- **Unsharp and high-boost** don't brighten at all (65.0, 65.7). Combined with CLAHE, noise σ grows 4–5× (5.64–7.59). Order matters: CLAHE → unsharp (6.03) is worse than unsharp → CLAHE (5.64), because it sharpens the noise CLAHE just amplified.
- **Hist-eq and gamma** brighten the most (127.0, 109.6), but on near-black areas they expose the 8×8 JPEG blocks (visible in the "dark patch" figure of the notebook, rows paella and chicken_wings). The table can't show this. CLAHE's clip limit prevents it.

**Outcome:** dataset 2 = wavelet denoise → CLAHE (decided 2026-11-05, `doc/pipeline/pipeline_02.md` section 3). Unsharp, high-boost, hist-eq and gamma are ruled out.

## Check on the built datasets (1,000 train images)

After building dataset 2, the same four metrics were measured on 1,000 train images (2026-11-05). "CLAHE only" was computed on the fly from `01_original` with the pipeline's own `clahe()`, so the comparison isolates the wavelet step.

| item | value |
|---|---|
| Images | 1,000 drawn at random from the **train** split of the manifest (`pandas.sample(1000, random_state=0)`) |
| Pool | all brightness levels, not only dark images |
| Pairs | the same 1,000 relpaths in every row |
| Values | mean over the 1,000 images |

| | brightness | contrast | Laplacian variance | noise σ |
|---|---|---|---|---|
| `01_original` | 111.4 | 63.8 | 878 | 1.99 |
| CLAHE only | 121.6 | 66.7 | 1956 | 3.70 |
| `02_denoise_clahe` (wavelet → CLAHE) | 121.5 | 66.8 | 1773 | 2.85 |

Split by the original brightness (thirds of the same 1,000 images):

| third | brightness: original | CLAHE only | wavelet → CLAHE | noise σ: original | CLAHE only | wavelet → CLAHE |
|---|---|---|---|---|---|---|
| darkest | 80.8 | 99.0 | 98.8 | 1.89 | 3.59 | 2.76 |
| middle | 111.9 | 122.9 | 122.8 | 1.99 | 3.70 | 2.86 |
| brightest | 141.6 | 143.0 | 143.1 | 2.10 | 3.82 | 2.92 |

- **CLAHE alone almost doubles the noise** (σ 1.99 → 3.70, ×1.9).
- **Denoising first halves the noise CLAHE adds** (+1.71 → +0.86) at the same brightness (121.6 vs 121.5) and contrast (66.7 vs 66.8). So the wavelet step doesn't weaken the enhancement.
- **CLAHE adapts to each image:** the darkest third gains +18 brightness, the brightest only +1.5. The pattern is the same in every third.
- **Laplacian variance:** CLAHE alone 1956, wavelet → CLAHE 1773. The wavelet step removes about 9% of the high-frequency energy, and part of that is noise. Both are about 2× the original, so edges and texture still get stronger overall.
- **Same direction as the 10 dark images** (σ 1.50 → 3.15 CLAHE only, → 2.36 wavelet → CLAHE). There, brightness gains were larger because every image was dark (mean 64.6, here 111.4).

## Limits

- 10 images, one draw (`SEED = 0`). The direction of each effect is clear in the figures, but the exact numbers would change with another draw. The 1,000-image check above confirms the direction for denoise + CLAHE.
- Pixel statistics do not measure classification accuracy. Whether denoise + CLAHE helps is measured when the team trains on datasets 1 and 2.
