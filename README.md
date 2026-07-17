# SCANNER: Explainable Multi-Class Skin Lesion Classification
A ResNet50 classifier for 7 types of skin lesions that doesn't just predict — it shows its work, with lesion boundaries and gradient saliency maps for every diagnosis.

Stack: TensorFlow / Keras · ResNet50 (transfer learning) · OpenCV · scikit-learn · HAM10000

> Research/educational project. Not a diagnostic tool, and not validated for clinical use.

## The Problem
Skin lesion datasets like HAM10000 are heavily imbalanced — benign nevi outnumber rare classes like dermatofibroma by more than 10:1 — so a naively trained classifier collapses toward predicting the majority class, which is exactly backwards from what matters clinically: catching the rare, dangerous cases. On top of that, most classifiers are black boxes — a prediction with no indication of *why*, which is a hard sell in a medical context where every diagnosis needs to be checkable.

## What SCANNER Does
SCANNER trains a transfer-learning classifier over 7 diagnostic classes from the HAM10000 dataset, explicitly correcting for class imbalance through targeted oversampling and weighted loss, then evaluates itself with clinically meaningful metrics rather than raw accuracy alone. Every prediction can be visually audited through two explainability layers: a classical CV lesion boundary overlay, and a gradient-based saliency map showing exactly which pixels drove the model's decision.

```
Dermatoscopic image
        ↓
ResNet50 (ImageNet-pretrained) → classifier head
        ↓
Predicted diagnosis + confidence
        ↓
Lesion boundary overlay (HSV + contours)  ←  "where is the lesion?"
Saliency heatmap (gradient-based)         ←  "what drove the call?"
```

## System Architecture
```
HAM10000 images + metadata.csv
        │
        ▼
┌────────────────────┐
│  Path Resolution    │  ← match image_id → .jpg, drop unmatched records
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│  Stratified Split   │  ← 80/20 train/test, stratified by diagnosis
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│  Class Balancing    │  ← per-class oversampling (500-800 target) + √-scaled class weights
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│  Augmentation       │  ← rotation, shift, zoom, flips, brightness, shear, channel shift
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│  ResNet50 (frozen)  │  ← ImageNet weights, transfer learning
│  + Classifier Head  │  ← GAP → BN → Dense(512) → Dropout → BN → Dense(256) → Dropout → Softmax(7)
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│  Training Loop      │  ← Adam, class-weighted CE, EarlyStopping + ReduceLROnPlateau + Checkpoint
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│  Evaluation Suite   │  ← per-class P/R/F1, balanced accuracy, Cohen's Kappa, MCC, confusion matrix
└─────────┬───────────┘
          │
          ▼
┌────────────────────┐
│  Explainability      │  ← HSV/contour lesion boundary + gradient saliency heatmaps
└────────────────────┘
```

## Quickstart
### Prerequisites
```bash
git clone https://github.com/OmPatel1891/SCANNER.git
cd SCANNER
pip install -r requirements.txt
```

Place the dataset alongside the script:
```
SCANNER/
├── HAM10000_images_part_1/
├── HAM10000_metadata - HAM10000_metadata.csv
└── scanner.py
```

### Run
```bash
python scanner.py
```

Outputs land in `model_outputs/`:
| File | Contents |
|---|---|
| `best_model.h5` / `enhanced_skin_lesion_classifier.h5` | Trained model checkpoints |
| `confusion_matrix.png` | Normalized confusion matrix across all 7 classes |
| `performance_metrics_by_class.png` | Precision / Recall / F1 per class |
| `training_history.png` | Accuracy and loss curves |
| `class_distribution.png` | Original vs. balanced vs. test set class counts |
| `overall_metrics_summary.png` | Accuracy, balanced accuracy, F1, Kappa at a glance |
| `sample_predictions_with_lesions.png` | Predictions with lesion boundary overlays |
| `saliency_map_sample_*.png` | Gradient saliency maps for individual predictions |

## Diagnosis Classes
| Code | Condition |
|---|---|
| `akiec` | Actinic Keratosis |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevus |
| `vasc` | Vascular Lesion |

## Evaluation
Raw accuracy is misleading on an imbalanced clinical dataset, so SCANNER reports a fuller picture:

| Metric | Why it's here |
|---|---|
| Per-class Precision / Recall / F1 | Surfaces classes the model quietly ignores |
| Balanced Accuracy | Accuracy adjusted so majority classes can't dominate the score |
| Cohen's Kappa | Agreement with ground truth beyond what chance would predict |
| Matthews Correlation Coefficient | Single robust score across all classes, imbalance-resistant |

## Explainability
Two independent layers, each answering a different question about a prediction:

| Layer | Method | Answers |
|---|---|---|
| Lesion boundary | HSV color thresholding + contour detection (OpenCV) | Where is the lesion in the image? |
| Saliency map | Vanilla gradients — max \|∂prediction/∂pixel\| per pixel | Which pixels drove that diagnosis? |

## Tech Stack
| Layer | Tools |
|---|---|
| Model | ResNet50 (ImageNet-pretrained, frozen base) + custom dense head |
| Framework | TensorFlow / Keras |
| Data handling | pandas, `ImageDataGenerator` |
| Evaluation | scikit-learn, seaborn, matplotlib |
| Explainability | OpenCV (lesion boundary), custom gradient-based saliency |
| Dataset | HAM10000 (10,000+ dermatoscopic images, 7 diagnostic classes) |

## Author
Om Patel · MS Data Science, University of Michigan
GitHub · LinkedIn
