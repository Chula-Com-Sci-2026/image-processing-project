CLIP zero-shot (openai/clip-vit-base-patch16), test split

## all test (1,920)

| dataset | n | accuracy | ±SE | macro P | macro R | macro F1 |
|---|---|---|---|---|---|---|
| 01_original | 1920 | 97.45 | 0.36 | 97.51 | 97.45 | 97.44 |
| 02_denoise_clahe | 1920 | 96.20 | 0.44 | 96.38 | 96.20 | 96.19 |

McNemar (exact, paired): `A only` = images A gets right and B gets wrong

| A vs B | A only | B only | p |
|---|---|---|---|
| 01_original vs 02_denoise_clahe | 39 | 15 | 0.0015 |

## official_test only (not CLIP-filtered)

| dataset | n | accuracy | ±SE | macro P | macro R | macro F1 |
|---|---|---|---|---|---|---|
| 01_original | 515 | 96.50 | 0.81 | 96.81 | 96.40 | 96.49 |
| 02_denoise_clahe | 515 | 94.95 | 0.96 | 95.58 | 94.96 | 94.99 |

McNemar (exact, paired): `A only` = images A gets right and B gets wrong

| A vs B | A only | B only | p |
|---|---|---|---|
| 01_original vs 02_denoise_clahe | 12 | 4 | 0.0768 |

## per-class F1, all test

| class | 01_original | 02_denoise_clahe |
|---|---|---|
| bibimbap | 98.2 | 97.6 |
| miso_soup | 99.5 | 99.0 |
| mussels | 97.7 | 95.0 |
| prime_rib | 95.7 | 92.6 |
| edamame | 99.5 | 99.5 |
| baby_back_ribs | 94.7 | 91.3 |
| paella | 96.0 | 93.4 |
| chicken_wings | 98.7 | 98.7 |
| guacamole | 97.1 | 97.6 |
| nachos | 97.4 | 97.2 |
