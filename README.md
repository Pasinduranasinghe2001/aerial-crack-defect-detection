# Aerial Crack & Surface Defect Detection using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Computer Vision pipeline for autonomous detection, localization, and classification of structural cracks and surface anomalies from aerial drone (UAV) imagery using the **YOLOv8** deep learning architecture.

---

## 📌 Problem Statement

Automated aerial visual inspection of civil structures (bridge decks, commercial roofs, concrete infrastructure) requires identifying hairline cracks, surface spalling, and material degradation from variable flight altitudes and camera angles. 

Key technical challenges addressed:
- **Scale Variance:** High variance in defect sizes due to fluctuating UAV ground sampling distances (GSD).
- **Perspective & Blur:** Gimbal motion blur and oblique viewing angles.
- **Real-Time Efficiency:** Balancing high spatial detection accuracy with low-latency inference suitable for edge deployment.

---

## 🏗️ Architecture & Methodology

- **Backbone:** Modified CSPDarknet53 with C2f (cross-stage partial with 2 convolutions) modules for rich multi-scale feature representation.
- **Neck:** Path Aggregation Network (PANet) / Feature Pyramid Network (FPN) for cross-scale feature fusion, ensuring robust detection of small structural fractures.
- **Head:** Anchor-free decoupled detection head separating classification and bounding-box regression loss (Task-Aligned Assigner).
- **Loss Functions:**
  - Complete IoU (CIoU) / Distribution Focal Loss (DFL) for precise bounding box localization.
  - Binary Cross-Entropy (BCE) for multi-class classification.

---

## 📂 Repository Structure

```
aerial-crack-defect-detection/
├── data.yaml            # Dataset configuration & class label mapping
├── train.py             # Model training script with configurable hyperparameters
├── evaluate.py          # Quantitative benchmark script (mAP@50, mAP@50:95, P, R)
├── predict.py           # Inference script for images, directories, or video streams
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore for weights, datasets, and run outputs
```

---

## 📊 Dataset Format

The pipeline uses the standard **YOLO bounding-box format**:
- Each image corresponds to a `.txt` label file in normalized coordinates:
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```
- **Directory Layout:**
  ```
  dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
  ```

### Defect Classes:
- `0: concrete_crack` (Structural longitudinal and transverse fractures)
- `1: surface_spalling` (Concrete surface peeling and exposed reinforcement)
- `2: roof_damage` (Tile displacement, punctures, surface wear)

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/aerial-crack-defect-detection.git
cd aerial-crack-defect-detection
pip install -r requirements.txt
```

### 2. Model Training
Train YOLOv8 on your labeled aerial dataset:
```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640
```

### 3. Model Evaluation
Evaluate validation metrics (mAP@0.5, mAP@0.5:0.95, Precision, Recall):
```bash
python evaluate.py --weights runs/train/aerial_defect_yolov8/weights/best.pt --data data.yaml
```

### 4. Inference & Visual Prediction
Run predictions on new aerial inspection frames:
```bash
python predict.py --source path/to/aerial_image.jpg --weights runs/train/aerial_defect_yolov8/weights/best.pt --conf 0.40
```

---

## 📈 Evaluation Metrics

The model is evaluated using standard COCO/PASCAL visual object detection metrics:
- **Precision (P):** Ratio of correct positive predictions to total predicted positives.
- **Recall (R):** Ratio of correctly detected defects to ground truth defects.
- **mAP@0.5:** Mean Average Precision at IoU threshold of 0.50.
- **mAP@0.5:0.95:** Primary metric averaging mAP across IoU thresholds from 0.50 to 0.95 (0.05 step).
