# References for the model exploration

Status: ⚠️ = not yet checked against the paper itself. Check each claim (especially the Food-101 numbers) before it goes on a slide,
then change it to ✅.

## Core papers (one per model choice)

| model | paper | status | why we cite it |
|---|---|---|---|
| SigLIP zero-shot | Zhai, Mustafa, Kolesnikov & Beyer, "Sigmoid Loss for Language Image Pre-Training", *ICCV* 2023 | ⚠️ | Replaces CLIP's softmax contrastive loss with a pairwise sigmoid loss. Reports stronger zero-shot ImageNet accuracy than CLIP at the same model size. Check whether Food-101 is among its zero-shot benchmarks. |
| Dense SIFT BoVW + SVM | Lazebnik, Schmid & Ponce, "Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories", *CVPR* 2006 | ⚠️ | The classical recipe we follow: dense SIFT on a grid → k-means visual words → histograms over a spatial pyramid → kernel SVM. |
| EfficientNet-B0 transfer | Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", *ICML* 2019 | ⚠️ | High ImageNet accuracy at ~5.3M parameters. The paper reports transfer results on several datasets; check that Food-101 is one of them and note its number. |

## Supporting papers

| topic | paper | status | why we cite it |
|---|---|---|---|
| ResNet-18 from scratch | He, Zhang, Ren & Sun, "Deep Residual Learning for Image Recognition", *CVPR* 2016 | ⚠️ | The architecture we train from random weights. Also cited in the team's `references_map.md`. |
| Fine-tune vs freeze | Kornblith, Shlens & Le, "Do Better ImageNet Models Transfer Better?", *CVPR* 2019 | ⚠️ | Compares fixed features vs fine-tuning across 12 datasets, Food-101 among them. Supports our unfreeze grid. Also in the team's `references_map.md`. |
| Bag of visual words | Csurka, Dance, Fan, Willamowski & Bray, "Visual Categorization with Bags of Keypoints", *ECCV Workshop on Statistical Learning in Computer Vision* 2004 | ⚠️ | Origin of the bag-of-keypoints classifier. |
| SIFT | Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", *IJCV* 60(2), 2004 | ⚠️ | The local descriptor. |
| RootSIFT | Arandjelović & Zisserman, "Three Things Everyone Should Know to Improve Object Retrieval", *CVPR* 2012 | ⚠️ | L1-normalise then square-root SIFT, so Euclidean distance behaves like the Hellinger kernel. |
| χ² kernel SVM | Zhang, Marszałek, Lazebnik & Schmid, "Local Features and Kernels for Classification of Texture and Object Categories: A Comprehensive Study", *IJCV* 73(2), 2007 | ⚠️ | χ² kernel works well for histogram features such as bag-of-words. |
| Dataset | Bossard, Guillaumin & Van Gool, "Food-101 – Mining Discriminative Components with Random Forests", *ECCV* 2014 | ⚠️ | The dataset. Its classical baselines include bag-of-words with SURF features, which is close to our classical model. |
