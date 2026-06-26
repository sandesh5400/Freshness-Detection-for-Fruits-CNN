# Fruit Freshness Detection — Custom CNN vs. Transfer Learning with Explainable AI

A comparative deep-learning study that classifies fruit images as **fresh** or **rotten** across three fruit types (apple, banana, orange), benchmarks a **custom CNN** against two transfer-learning backbones (**MobileNetV2**, **ResNet50**), and explains every model's decisions with **Grad-CAM** and **SHAP**.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.19-orange">
<img alt="Keras" src="https://img.shields.io/badge/Keras-3-red">
<img alt="Platform" src="https://img.shields.io/badge/Platform-Kaggle%20(GPU)-20BEFF">
</p>

---

## Overview

The goal of this project is not only to find the best-performing classifier, but to understand **what visual evidence each model relies on** when it makes a decision. Three models are trained on the same data under the same conditions, evaluated with standard classification metrics, and then interrogated with two explainability methods:

- **Grad-CAM** — highlights the image regions that drove a prediction (class-discriminative localisation).
- **SHAP** — estimates the per-pixel contribution (positive/negative evidence) toward the predicted class.

Every figure is exported as **PDF** and every table as **CSV**, so the results are fully reproducible and ready to drop into a report.

---

## Headline Results

Evaluated on the held-out test set of **2,698 images** (macro-averaged precision/recall/F1):

| Model | Accuracy | Precision | Recall | F1-Score | Parameters | Training Time |
|---|---|---|---|---|---|---|
| Custom CNN | 95.89% | 95.72% | 96.54% | 96.01% | 0.26 M | ~500 s |
| MobileNetV2 | 99.56% | 99.54% | 99.51% | 99.52% | 2.27 M | ~726 s |
| **ResNet50** | **99.78%** | **99.79%** | **99.76%** | **99.78%** | 23.6 M | ~831 s |

**Takeaways:** ResNet50 is the most accurate by a narrow margin, but **MobileNetV2 offers the best performance-to-cost trade-off** — it reaches within ~0.26% F1 of ResNet50 using roughly **1/10th the parameters** and shorter training time. The from-scratch custom CNN is already strong (96% F1) with a tiny footprint (0.26 M parameters).

---

## Dataset

**[Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)** (Sriram R., Kaggle) — **6 classes**, **13,599 images** total, pre-split into `train/` and `test/`.

| Class | Train | Test | Total |
|---|---|---|---|
| freshapples | 1,693 | 395 | 2,088 |
| freshbanana | 1,581 | 381 | 1,962 |
| freshoranges | 1,466 | 388 | 1,854 |
| rottenapples | 2,342 | 601 | 2,943 |
| rottenbanana | 2,224 | 530 | 2,754 |
| rottenoranges | 1,595 | 403 | 1,998 |
| **Total** | **10,901** | **2,698** | **13,599** |

All images are resized to **224 × 224 × 3**. A **20% validation split** is carved from the training set (8,721 train / 2,180 validation), giving a clean train / validation / test separation.

---

## Project Structure

```
.
├── fruit-freshness-detection.ipynb   # End-to-end notebook (Phases 0–8)
├── README.md
└── outputs/
    ├── figures/                      # All figures, as PDF
    │   ├── fig01_dataset_samples.pdf
    │   ├── fig02_workflow.pdf
    │   ├── fig03_training_curves.pdf
    │   ├── fig04_confusion_matrices.pdf
    │   ├── fig05_gradcam_correct.pdf
    │   ├── fig06_gradcam_misclassified.pdf
    │   └── fig07_shap.pdf
    ├── tables/                       # All tables, as CSV
    │   ├── dataset_description.csv
    │   ├── model_architecture.csv
    │   ├── evaluation_results.csv
    │   ├── final_comparison.csv
    │   ├── classification_report_<model>.csv
    │   └── confusion_matrix_<model>.csv
    └── models/                       # Saved models + text summaries
        ├── Custom_CNN.keras
        ├── MobileNetV2.keras
        └── ResNet50.keras
```

---

## Methodology

### 1. Per-model preprocessing
Each architecture expects a different input normalisation, so each gets its own pipeline (mixing these up silently degrades accuracy):

| Model | Preprocessing |
|---|---|
| Custom CNN | Rescale to `[0, 1]` |
| MobileNetV2 | `mobilenet_v2.preprocess_input` → `[-1, 1]` |
| ResNet50 | `resnet50.preprocess_input` → caffe-style BGR mean subtraction |

Training data is augmented with random horizontal flips, small rotations, and zoom; validation and test data are left un-augmented.

### 2. Architectures

| Model | Total Params | Trainable (head) | Layers |
|---|---|---|---|
| Custom CNN | 259,526 | 258,822 | 17 |
| MobileNetV2 | 2,265,670 | 7,686 | 157 |
| ResNet50 | 23,600,006 | 12,294 | 178 |

- **Custom CNN** — four `Conv → BatchNorm → MaxPool` blocks → Global Average Pooling → Dropout → Dense → softmax(6).
- **MobileNetV2 / ResNet50** — ImageNet-pretrained backbone (`include_top=False`) with a frozen base, followed by Global Average Pooling → Dropout → softmax(6). Built functionally so the convolutional layers stay top-level for Grad-CAM.

### 3. Training setup
- Input 224×224, batch size 32, fixed seed for fair comparison.
- **Head training:** Adam (lr 1e-3), categorical cross-entropy, up to 10 epochs.
- **Fine-tuning (transfer models):** top layers unfrozen at a low learning rate (1e-5), **BatchNorm layers kept frozen**.
- Callbacks: `EarlyStopping` (restore best weights) and `ReduceLROnPlateau`.

### 4. Explainability
- **Grad-CAM** uses each model's last convolutional layer (`last_conv`, `out_relu`, `conv5_block3_out`), with each image fed through that model's own preprocessing. Generated for both correctly classified and misclassified images.
- **SHAP** uses the image masker (`inpaint_telea`) in a model-agnostic, version-robust setup, explaining the top predicted class for selected test images.

---

## Results & Figures

| Figure | Contents |
|---|---|
| `fig01_dataset_samples.pdf` | One sample per class |
| `fig02_workflow.pdf` | Model-design workflow diagram |
| `fig03_training_curves.pdf` | Train/validation accuracy & loss per model |
| `fig04_confusion_matrices.pdf` | Confusion matrix per model |
| `fig05_gradcam_correct.pdf` | Grad-CAM on correct predictions (3 models side-by-side) |
| `fig06_gradcam_misclassified.pdf` | Grad-CAM on misclassifications, with true/predicted labels |
| `fig07_shap.pdf` | SHAP explanations for selected test images |

Per-class precision/recall/F1 is in `classification_report_<model>.csv`, raw confusion counts in `confusion_matrix_<model>.csv`, and the consolidated scoreboard in `evaluation_results.csv` / `final_comparison.csv`.

---

## Research Questions

- **RQ1 — How well does a custom CNN classify the dataset?** Very well for its size: **96.01% F1** (95.89% accuracy) with only 0.26 M parameters.
- **RQ2 — Does transfer learning improve over the custom CNN?** Yes. MobileNetV2 (99.52% F1) and ResNet50 (99.78% F1) both clearly outperform the custom CNN, cutting the error rate by roughly an order of magnitude.
- **RQ3 — Best trade-off between performance and cost?** **MobileNetV2** — near-ResNet50 accuracy at ~1/10th the parameters and lower training time.
- **RQ4 — Where does each model focus (Grad-CAM)?** See `fig05`; heatmaps concentrate on the fruit body and spoilage cues. Per-model qualitative ratings are recorded in `final_comparison.csv`.
- **RQ5 — Do attention patterns differ between custom CNN and transfer models?** Compared side-by-side in `fig05`; transfer backbones generally produce more focused localisation than the shallow custom CNN.
- **RQ6 — What evidence do SHAP explanations reveal?** See `fig07` for the positive/negative pixel contributions toward each predicted class.
- **RQ7 — Can explainability diagnose misclassifications?** `fig06` pairs each error's true/predicted label with its Grad-CAM, showing which misleading regions the model attended to.

> The qualitative XAI scores (Grad-CAM quality, SHAP interpretability, overall finding) in `final_comparison.csv` are intended to be filled in from direct inspection of Figures 5–7 — they are deliberately not auto-generated.

---

## How to Run (Kaggle)

1. Open the notebook in a Kaggle kernel.
2. In the right-hand panel, set:
   - **Accelerator → GPU** (T4 / P100)
   - **Internet → ON** (needed to download ImageNet weights)
   - **Add Input →** attach the *fruits-fresh-and-rotten-for-classification* dataset
3. **Run All.** Outputs are written to `/kaggle/working/outputs/`.

To manage runtime, adjust the config cell: `EPOCHS`, `DO_FINETUNE`, `FINE_TUNE_EPOCHS`, and `SHAP_MAX_EVALS`. A download cell at the end bundles everything into a single ZIP.

---

## Environment

Developed and run on Kaggle with **2 × NVIDIA Tesla T4** GPUs.

```
tensorflow == 2.19    (Keras 3)
numpy, pandas, matplotlib
scikit-learn
shap
opencv-python
Pillow
```

---

## Acknowledgements

- Dataset: *Fruits fresh and rotten for classification* by Sriram R. (Kaggle).
- Pretrained backbones: MobileNetV2 and ResNet50 (ImageNet weights via `keras.applications`).
- Explainability: Grad-CAM and the SHAP library.

## License

Released for academic and educational use. Please respect the original dataset's license terms when redistributing data or trained weights.
