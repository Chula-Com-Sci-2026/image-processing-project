# Reference map: course textbook ↔ papers

Maps each technique used in the project to the course textbook section and to the primary paper (or standard) to cite.

**Verification column**
- ✅ = checked against an online source during preparation (arXiv / CVF / ICLR page)
- ⚠️ = written from memory; confirm the year, venue and authors before the final report
- — = no paper needed (standard or library behaviour)

## 1. Prepare and light correction

| What we use | Textbook section | Page | Paper / standard to cite | Check | Note |
|---|---|---|---|---|---|
| Resize (`cv2.INTER_AREA`, short side 256) | 2.3.3 Interpolation Techniques | 22 | OpenCV documentation (`cv::resize`, `INTER_AREA`) | — | Area resampling suits downscaling; no paper needed |
| CIELAB (used by `clahe`) | 2.4.7 CIELAB colour model | 29 | CIE 1976 L\*a\*b\* colour space (CIE standard) | — | Standard, not a paper |
| Histogram equalization (`histeq`) | 3.4.1 Histogram Equalization | 42 | Pizer et al., "Adaptive histogram equalization and its variations", *CVGIP* 39, 1987 | ⚠️ | Also covers AHE, the basis of CLAHE |
| CLAHE and clip limit (`clahe`) | 3.4.2 CLAHE | 45 | Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", *Graphics Gems IV*, 1994 | ⚠️ | The standard citation for CLAHE |
| Y/Cr/Cb (channel used by `histeq`) | 7.6 JPEG | 157 | ITU-R BT.601 (YCbCr definition) | — | The textbook covers YCbCr only for JPEG; we use it to separate brightness from colour |
| JPEG quality 95 | 7.5.3 Quantization and 7.6 | 155, 157 | Wallace, "The JPEG still picture compression standard", *Communications of the ACM* 34(4), 1991 | ⚠️ | |

## 2. Segmentation and detection

| What we use | Textbook section | Page | Paper to cite | Check | Note |
|---|---|---|---|---|---|
| SAM ViT-B (`sam`, `gsam`) | 12.4.5 Segment Anything Model | 303 | Kirillov et al., "Segment Anything", *ICCV* 2023 (arXiv 2304.02643) | ✅ | The most directly relevant section |
| SlimSAM (`gsam_light`) | 12.4.5 | 304 | Chen, Fang, Ma & Wang, "SlimSAM: 0.1% Data Makes Segment Anything Slim", *NeurIPS* 2024 | ⚠️ | Weights `Zigeng/SlimSAM-uniform-77` |
| Grounded-SAM (`gsam`, `gsam_light`) | 12.4 opening paragraph | 290 | Ren et al., "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks", arXiv 2401.14159, 2024 | ✅ | The textbook only mentions it by name, so this paper carries the explanation |
| Grounding DINO tiny | 13.5.3.1 DETR, 13.5.4 YOLOE, 13.8 | 330, 331, 337 | Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", *ECCV* 2024 (arXiv 2303.05499) | ✅ arXiv / ⚠️ venue | Explain via DETR → open-vocabulary detection |
| DETR (background for Grounding DINO) | 13.5.3.1 | 330 | Carion et al., "End-to-End Object Detection with Transformers", *ECCV* 2020 | ⚠️ | |
| GrabCut (method dropped in the trial) | — | — | Rother, Kolmogorov & Blake, "GrabCut", *ACM TOG (SIGGRAPH)* 2004 | ⚠️ | Not in the table; used in the Segmentation Methods slide |
| U-Net (optional extension) | — | — | Ronneberger, Fischer & Brox, "U-Net", *MICCAI* 2015 | ⚠️ | Not in the table |
| IoU and Dice (FoodSeg103 extension) | 12.3.1, 12.3.2 | 288, 289 | Wu et al., "A Large-Scale Benchmark for Food Image Segmentation" (FoodSeg103), *ACM MM* 2021 | ⚠️ | The dataset we would measure mask IoU against |
| SAM on food (related work) | 12.4.5 | 303 | Lan et al., "FoodSAM: Any Food Segmentation", *IEEE TMM* 2023 (arXiv 2308.05938) | ✅ | |
| | | | Alahmari, Gardner & Salem, "Segment Anything in Food Images", *CVPR Workshops* 2024 | ✅ | |

## 3. Models and training

| What we use | Textbook section | Page | Paper to cite | Check | Note |
|---|---|---|---|---|---|
| HSV colour histogram (MLP features) | 2.4.6 and 10.2.1 | 28, 230 | Smith, "Color gamut transform pairs", *SIGGRAPH* 1978 | ⚠️ | Origin of HSV; optional to cite |
| HOG (MLP features) | 10.3 | 231 | Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection", *CVPR* 2005 | ⚠️ | |
| **LBP (MLP features)** | **not in the table** | — | Ojala, Pietikäinen & Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns", *TPAMI* 24(7), 2002 | ⚠️ | We use `uniform` LBP; look for an LBP section in the textbook |
| MLP | 10.4.4 Multilayer Perceptron | 237 | Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors", *Nature* 1986 | ⚠️ | |
| ResNet-50 (`resnet50.a1_in1k`) | 11.3.5 ResNet | 263 | He et al., "Deep Residual Learning for Image Recognition", *CVPR* 2016 | ⚠️ | |
| | | | Wightman, Touvron & Jégou, "ResNet strikes back", arXiv 2110.00476, 2021 | ⚠️ | Source of the `a1` training recipe for the timm weights |
| DeiT-Small (`deit_small_patch16_224.fb_in1k`) | 11.4.1 ViT and 11.5 | 267, 276 | Touvron et al., "Training data-efficient image transformers & distillation through attention", *ICML* 2021 | ⚠️ | |
| | | | Dosovitskiy et al., "An Image is Worth 16x16 Words", *ICLR* 2021 | ⚠️ | ViT, the base of DeiT |
| Augmentation (rotation, flip) | 11.2 | 258 | Shorten & Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning", *Journal of Big Data* 2019 | ⚠️ | Survey that covers every transform we use |
| RandomResizedCrop | 11.2 | 258 | Szegedy et al., "Going Deeper with Convolutions", *CVPR* 2015 | ⚠️ | Origin of the Inception-style random-sized crop |
| Transfer learning and fine-tuning | 11.6 | 277 | Yosinski et al., "How transferable are features in deep neural networks?", *NeurIPS* 2014 | ⚠️ | |
| | | | Kornblith, Shlens & Le, "Do Better ImageNet Models Transfer Better?", *CVPR* 2019 | ⚠️ | Supports full fine-tuning from ImageNet |
| Train / val / test | 10.6.1 | 244 | Bossard, Guillaumin & Van Gool, "Food-101: Mining Discriminative Components with Random Forests", *ECCV* 2014 | ✅ | Source of the official train/test split we follow |
| Adam | 10.6.3 Optimizer | 246–247 | Kingma & Ba, "Adam: A Method for Stochastic Optimization", *ICLR* 2015 | ⚠️ | |
| **AdamW (what we actually use)** | **not in the textbook** | — | Loshchilov & Hutter, "Decoupled Weight Decay Regularization", *ICLR* 2019 | ⚠️ | Fills the textbook gap; explain how decoupled weight decay differs from L2 in Adam |
| **Cosine learning-rate schedule** | not in the table | — | Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", *ICLR* 2017 | ⚠️ | We use `CosineAnnealingLR` |
| **Label smoothing 0.1** | not in the table | — | Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", *CVPR* 2016 | ⚠️ | Used in the CNN / Transformer notebooks |
| ImageNet mean/std normalisation, grey fill | not in the table | — | Deng et al., "ImageNet: A large-scale hierarchical image database", *CVPR* 2009 | ⚠️ | Source of the mean colour (124, 116, 104) |

## 4. Evaluation

| What we use | Textbook section | Page | Paper to cite | Check | Note |
|---|---|---|---|---|---|
| Confusion matrix | 10.7.1 | 248 | Sokolova & Lapalme, "A systematic analysis of performance measures for classification tasks", *Information Processing & Management* 45(4), 2009 | ⚠️ | Covers both the confusion matrix and the metrics |
| Accuracy, precision, recall, macro-F1 | 10.7.2 | 248–249 | (same as above) | ⚠️ | Explain why we use **macro** averaging (equal weight per class) |
| **McNemar test (comparing two variants)** | not in the table | — | Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", *Neural Computation* 10(7), 1998 | ⚠️ | Paired test on the same test photos, using `_preds.csv` |

## 5. Background effect (the motivation for the research question)

Not covered by the textbook. These come from the literature review.

| Topic | Paper to cite | Check |
|---|---|---|
| Models rely on the image background | Xiao, Engstrom, Ilyas & Madry, "Noise or Signal: The Role of Image Backgrounds in Object Recognition", *ICLR* 2021 | ✅ |
| Fusing object + context features | Zhu, Xie & Yuille, "Object Recognition with and without Objects", *IJCAI* 2017 | ⚠️ |
| Food classification with ViT | Gao et al., AlsmViT, *Journal of Food Engineering* 2024, doi:10.1016/j.jfoodeng.2023.111833 | ✅ (from the teammate's review) |
| Food classification with attention | Deng et al., MAMS-Net, *Computers, Materials & Continua* 2024, doi:10.32604/cmc.2024.052174 | ✅ (from the teammate's review) |

## Gaps the textbook does not cover (papers are needed here)

1. **Grounded-SAM / Grounding DINO**: the textbook only names them or explains them via DETR, so cite Ren et al. 2024 and Liu et al. 2024.
2. **LBP**: we use it but it is not in the table; check whether the textbook has a section on it.
3. **AdamW, cosine schedule, label smoothing**: the textbook stops at Adam, so cite Loshchilov & Hutter and Szegedy et al.
4. **McNemar test**: needed when we claim a variant is significantly better; cite Dietterich 1998.
5. **The background effect**: the core of the research question; cite Xiao et al. 2021.

Before the final report, check every ⚠️ row on Google Scholar or the paper page, especially the venue and year of SlimSAM, GrabCut, Zhu et al. and Grounding DINO.