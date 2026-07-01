# 🫁 COVID-19 CT Scan Classification Using VGG16 Transfer Learning

<p align="center">
  <img src="https://img.shields.io/badge/Model-VGG16%20Transfer%20Learning-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-FF6F00?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF?style=for-the-badge&logo=kaggle" />
  <img src="https://img.shields.io/badge/Test%20Accuracy-96.48%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-purple?style=for-the-badge" />
</p>

---

## 📌 Overview

A comprehensive deep learning study on **COVID-19 CT scan image classification** using VGG16 transfer learning. This project fine-tunes a pre-trained VGG16 convolutional neural network on the SARS-CoV-2 CT Scan Dataset to classify CT scan images as **COVID-19** or **Non-COVID-19**.

The key result: our optimized VGG16 pipeline achieves **96.48% test accuracy** — surpassing the base paper's VGG16 result of 94.96% by **+1.52 percentage points**, and matching within 0.29 points of the base paper's complex three-model ensemble (ResNet18 + GoogleNet + ShuffleNet = 96.77%).

> **Course:** Deep Learning — Semester Assignment  
> **Instructor:** Engr. Umair Ayaz Kamangar  
> **Institution:** Sukkur IBA University, Department of Computer Systems Engineering

---

## 🏆 Key Results at a Glance

| Metric | Value |
|---|---|
| Training Accuracy | 98.97% |
| Validation Accuracy | 96.09% |
| **Test Accuracy** | **96.48%** |
| Macro F1-Score | 0.97 |
| COVID-Positive Recall | 0.98 |
| COVID-Positive Precision | 0.96 |

### Comparison vs. Base Paper (Amouzegar et al., 2022)

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| VGG16 (Base Paper) | 94.96% | 94.02% | 95.43% | 94.97% |
| ResNet18 (Base Paper) | 92.74% | 92.03% | 94.78% | 93.38% |
| GoogleNet (Base Paper) | 93.55% | 92.75% | 95.52% | 94.12% |
| ShuffleNet (Base Paper) | 93.95% | 93.28% | 93.28% | 94.34% |
| **Ensemble (Base Paper Best)** | **96.77%** | 95.40% | 97.65% | 96.51% |
| **VGG16 Optimized (Ours)** | **96.48%** | **97.00%** | **98.00%** | **97.00%** |

> Our single VGG16 model achieves **higher recall than the base paper's 3-model ensemble** — a clinically critical advantage for minimizing missed COVID diagnoses.

---

## 🗂️ Repository Structure

```
COVID19-VGG16-Classification/
│
├── Covid19_Detection.ipynb          # Main training notebook (model definition, training, evaluation)
├── Covid19_Detection_Eval.ipynb     # Evaluation notebook (metrics, plots, confusion matrix)
├── Covid19_Training_Results/        # Saved model weights, training logs, result plots
├── COVID19_DL_Report_Asma_Usman.pdf # Full academic report (26 pages)
├── DL_Base_Paper.pdf                # Base paper: Amouzegar et al., 2022
└── README.md
```

---

## 🧠 Model Architecture

The VGG16 base (pre-trained on ImageNet) is frozen, and a custom classification head is appended for binary COVID-19 classification:

```
Input (224×224×3)
    ↓
VGG16 Convolutional Base — FROZEN (ImageNet weights)
  ├── Block 1: 2× Conv(64) + MaxPool  → 112×112×64
  ├── Block 2: 2× Conv(128) + MaxPool → 56×56×128
  ├── Block 3: 3× Conv(256) + MaxPool → 28×28×256
  ├── Block 4: 3× Conv(512) + MaxPool → 14×14×512
  └── Block 5: 3× Conv(512) + MaxPool → 7×7×512
    ↓
Flatten → 25088 features
    ↓
Dense(256, activation='relu')     ← TRAINABLE
    ↓
Dense(2, activation='softmax')    ← TRAINABLE
    ↓
Output: [Non-COVID, COVID]
```

---

## 📊 Dataset

**SARS-CoV-2 CT Scan Dataset** — sourced from Kaggle (`plameneduardo/sarscov2-ctscan-dataset`)

| Attribute | COVID Class | Non-COVID Class |
|---|---|---|
| Number of Images | 2,481 | 2,481 |
| Image Format | JPEG/PNG | JPEG/PNG |
| Resized Resolution | 224×224 px | 224×224 px |
| Color Space | RGB (3-channel) | RGB (3-channel) |

**Dataset Split (Stratified)**

| Split | Percentage | Approx. Images | Purpose |
|---|---|---|---|
| Training | 72% | ~3,374 | Model Learning |
| Validation | 13% | ~844 | Hyperparameter Tuning |
| Testing | 15% | ~744 (373 used) | Final Evaluation |

---

## ⚙️ Methodology

### Data Preprocessing Pipeline

| Step | Description |
|---|---|
| Image Loading | OpenCV `cv2.imread` in BGR → converted to RGB |
| Resizing | All images resized to 224×224 px (bilinear interpolation) |
| Normalization | Pixel values scaled from [0, 255] → [0.0, 1.0] (÷ 255.0) |
| Label Encoding | Integer labels → one-hot encoded via `to_categorical()` |
| Augmentation | Horizontal flipping via `ImageDataGenerator` (training only) |

### Hyperparameter Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 128 |
| Epochs | 20 (max) |
| Early Stopping Patience | 3 |
| Input Shape | 224 × 224 × 3 |
| Dense Units (Hidden) | 256 |
| Activation (Hidden) | ReLU |
| Activation (Output) | Softmax |
| VGG16 Base | Frozen (trainable=False) |

### Training Environment

| Component | Specification |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA Tesla T4 (2 × 13,757 MB VRAM) |
| Framework | TensorFlow 2.x + Keras |
| Python | 3.12 |
| Key Libraries | TensorFlow/Keras, NumPy, OpenCV, scikit-learn, Matplotlib, Seaborn |

---

## 📈 Training History

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 49.25% | 5.2526 | 54.03% | 0.6631 |
| 2 | 63.81% | 0.6305 | 79.15% | 0.4933 |
| 5 | 88.32% | 0.3256 | 88.63% | 0.2970 |
| 10 | 93.77% | 0.1820 | 93.36% | 0.1826 |
| 15 | 97.83% | 0.1047 | 94.55% | 0.1408 |
| 18 | 98.46% | 0.0858 | 95.26% | 0.1201 |
| 20 | 99.11% | 0.0695 | 95.97% | 0.1152 |

---

## 🔬 Classification Report (Test Set — 373 Images)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Non-COVID (0) | 0.98 | 0.96 | 0.97 | 196 |
| COVID (1) | 0.96 | 0.98 | 0.97 | 177 |
| **Macro Avg** | **0.97** | **0.97** | **0.97** | **373** |
| Weighted Avg | 0.97 | 0.97 | 0.97 | 373 |

**Confusion Matrix (Test Set)**

```
                  Predicted
                Non-COVID   COVID
Actual Non-COVID   188        8
       COVID         4      173
```

- ✅ 188 True Negatives (non-COVID correctly identified)
- ✅ 173 True Positives (COVID correctly identified)
- ❌ 8 False Positives (non-COVID misclassified as COVID)
- ❌ 4 False Negatives (COVID missed — minimized by design)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy opencv-python scikit-learn matplotlib seaborn
```

### Run Training

Open and run `Covid19_Detection.ipynb` in Kaggle Notebooks or locally with GPU support.

```python
# Key model definition snippet
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### Run Evaluation

Open `Covid19_Detection_Eval.ipynb` to reproduce all metrics, plots, and the confusion matrix.

### Dataset Setup

Download from Kaggle: [plameneduardo/sarscov2-ctscan-dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)

```
dataset/
├── COVID/       # 2,481 CT scan images
└── non-COVID/   # 2,481 CT scan images
```

---

## 💡 Why This Outperforms the Base Paper's VGG16

Three pipeline improvements drove the +1.52% accuracy gain over the base paper's VGG16:

1. **Horizontal data augmentation** — increases effective training diversity without flipping clinically relevant lateral patterns
2. **Frozen convolutional base** — preserves ImageNet representations, prevents catastrophic forgetting on a ~3,374-image training set
3. **3-way stratified split (72/13/15) with early stopping** — proper validation prevents overfitting and yields an unbiased test evaluation

---

## 🔮 Future Work

- Progressive fine-tuning of upper VGG16 blocks (potential +1–2% accuracy gain)
- Grad-CAM visualization for explainable AI (highlighting infected CT regions)
- Multi-class extension: COVID-19 vs. pneumonia vs. normal
- Test on EfficientNet-B4 and Vision Transformers (ViT)
- Federated learning for multi-hospital training without sharing patient data
- Validation on independent multi-institutional datasets

---

## 👥 Team

| Name | Contributions | GitHub |
|---|---|---|
| **Asma Channa** (133-22-0042) | Data preprocessing pipeline, model architecture design, training implementation, evaluation metrics, report writing (Introduction, Methodology, Results) | [@asma-13](https://github.com/asma-13) |
| **Muhammad Usman** (133-22-0016) | Literature review, dataset curation & analysis, visualization (accuracy/loss plots, confusion matrix), hyperparameter tuning, report writing (Abstract, Literature Review, Conclusion, References) | [@Usmansarwar143](https://github.com/Usmansarwar143) |

> **Instructor:** Engr. Umair Ayaz Kamangar  
> **Course:** Deep Learning — Semester Assignment  
> **Institution:** Sukkur IBA University, Department of Computer Systems Engineering

---

## 📄 Report

The full 26-page academic report is available in [`COVID19_DL_Report_Asma_Usman.pdf`](./COVID19_DL_Report_Asma_Usman.pdf), covering:

- Comprehensive literature review (15 recent papers, 2022–2026)
- VGG16 architecture deep-dive with layer-by-layer breakdown
- Transfer learning strategy and frozen-base rationale
- Full preprocessing pipeline documentation
- Training history, confusion matrix, classification report
- Comparison against Amouzegar et al. (2022) base paper benchmarks
- Limitations and future research directions

---

## 📚 References

- **Base Paper:** F. Amouzegar et al., *"Diagnosis of COVID-19 disease using CT scan images and pre-trained models,"* arXiv:2208.07829, 2022.
- **Dataset:** E. Soares et al., *"SARS-CoV-2 CT-scan dataset,"* medRxiv, 2020. [Kaggle](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)
- K. Simonyan and A. Zisserman, *"Very Deep Convolutional Networks for Large-Scale Image Recognition,"* ICLR 2015.

---

## 📜 License

Developed as an academic Deep Learning semester assignment at Sukkur IBA University. Free to reference or build upon with proper attribution.

---

<p align="center">VGG16 Transfer Learning · TensorFlow/Keras · Kaggle GPU · 96.48% Test Accuracy &nbsp;|&nbsp; Sukkur IBA University &nbsp;|&nbsp; 2026</p>
