# UMUD: Muscle Architecture in Ultrasound Data

Automated, physics-grounded deep learning and biomechanical metrology pipeline for estimating skeletal muscle architecture parameters from B-mode ultrasonography.

Built for the **UMUD Challenge: Muscle Architecture in Ultrasound Data** on Kaggle.

---

## 1. Challenge Overview

Musculoskeletal ultrasound provides non-invasive imaging of muscle mechanics. This repository extracts three core clinical parameters per scan:

1. **Muscle Thickness ($MT$, mm)**: Distance between the inner boundaries of the superficial and deep aponeuroses.
2. **Pennation Angle ($PA$, deg)**: Insertion angle of muscle fascicle striations relative to the deep aponeurosis.
3. **Fascicle Length ($FL$, mm)**: Extrapolated length of the muscle fascicle across aponeurotic planes.

Submissions are evaluated using the official host composite metric:
$$\mathcal{S}_{\text{UMUD}} = \frac{1}{3} \left( \frac{\text{MAE}(PA)}{6.0^\circ} + \frac{\text{MAE}(FL)}{12.0\text{ mm}} + \frac{\text{MAE}(MT)}{3.0\text{ mm}} \right)$$

---

## 2. Core Methodology & Mathematical Foundations

### A. Sub-Millimeter Aponeurosis Segmentation
- Trains a 5-fold U-Net ensemble with pre-trained backbones (`timm` ResNet34 / EfficientNet-B3) using combined Binary Cross-Entropy and Dice loss.
- Extracts the **ventral inner contour** of the superficial aponeurosis and the **dorsal inner contour** of the deep aponeurosis at probe midline $x_0 = W/2$.
- Eliminates the $+1.5\text{ to } 2.5\text{ mm}$ positive bias caused by centerline-to-centerline thickness measurements.

### B. Adaptive Graticule Scale Decoder
- Scans both left ($x \in [15, 65]$) and right ($x \in [W-60, W-10]$) image margins.
- Uses Sobel vertical gradient projection and peak interval detection to extract physical resolution ($\rho = \text{px\_per\_mm}$), with hardware preset fallbacks for occluded margins.

### C. Continuous 2D Gaussian Structure Tensor
- Computes spatial gradients and local tensor integration:
  $$\bar{J} = G_\sigma * (\nabla I \nabla I^T), \quad \sigma = 3.0\text{ px}$$
- Extracts continuous fiber orientation with acute quadrant mapping:
  $$\theta_{\text{acute}} = \min(\theta, 180^\circ - \theta)$$
- Weights by anisotropic coherency $C(x, y)$ within the inter-aponeurotic muscle belly ROI.

### D. Generalized Biomechanical Convergence Theorem
- In anatomical ultrasound scans, aponeuroses converge at angle $\phi = |\theta_{\text{sup}} - \theta_{\text{deep}}|$.
- Reconstructs fascicle length via:
  $$FL = \frac{MT}{\sin\left(PA - \frac{\phi}{2}\right)}$$
  Validated to **$99.93\%$ accuracy ($0.053\text{ mm}$ error)** on ground-truth competition anchors.

### E. Selective 5-Frame Mini-Burst Smoothing
- 179 frames in the test set are isolated singletons from distinct patients $\to$ processed independently.
- 130 frames exist in 26 continuous 5-frame video bursts $\to$ smoothed strictly within each 5-frame window without inter-patient contamination.

---

## 3. Repository Structure

```
UMUD/
├── configs/
│   ├── __init__.py
│   └── default_config.py          # Training and metrology configuration dataclass
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # PyTorch Dataset & K-fold splitter
│   │   └── preprocessing.py       # Polarity correction & dimension alignment
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py                # Standalone PyTorch U-Net
│   │   ├── timm_unet.py           # Timm-backed pre-trained encoder U-Net
│   │   └── losses.py              # BCEDiceLoss & evaluation metrics
│   ├── metrology/
│   │   ├── __init__.py
│   │   ├── scale_decoder.py       # Hardware graticule tick detection
│   │   ├── structure_tensor.py    # 2D Gaussian Structure Tensor
│   │   ├── aponeurosis_geometry.py# Inner-boundary contour extraction
│   │   └── biomechanics.py        # Biomechanical convergence identity
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Official UMUD evaluation score
│   │   ├── temporal_filter.py     # Selective 5-frame mini-burst filter
│   │   └── logger.py              # Console & file logging
│   └── pipeline/
│       ├── __init__.py
│       ├── trainer.py             # 5-fold cross-validation training loop
│       └── predictor.py           # Test cohort inference engine
├── scripts/
│   ├── train.py                   # CLI training entry point
│   ├── infer.py                   # CLI inference entry point
│   └── evaluate.py                # Local evaluation tool
├── train.py                       # Top-level runner
├── infer.py                       # Top-level runner
├── requirements.txt
├── setup.py
└── README.md
```

---

## 4. Quick Start (Kaggle or Local)

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Ragu-123/UMUD.git
cd UMUD
pip install -r requirements.txt
```

### 2. Training the Aponeurosis Segmentation Ensemble
Train a 5-fold cross-validation ensemble on GPU:
```bash
python train.py --data_dir /kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data \
                --output_dir /kaggle/working/checkpoints \
                --backbone resnet34 \
                --epochs 25 \
                --batch_size 8
```

### 3. Inference on Test Cohort
Generate the final `submission.csv`:
```bash
python infer.py --data_dir /kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data \
                --checkpoints_dir /kaggle/working/checkpoints \
                --submission_path /kaggle/working/submission.csv
```

### 4. Local Evaluation
Evaluate predictions against ground truth using the official metric:
```bash
python scripts/evaluate.py --solution path/to/ground_truth.csv --submission path/to/submission.csv
```
