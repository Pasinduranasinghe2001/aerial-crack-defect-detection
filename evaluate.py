"""
Model Evaluation and Quantitative Metrics Script.
Calculates mAP@0.5, mAP@0.5:0.95, Precision, Recall, and per-class metrics.
"""

import argparse
import os
from ultralytics import YOLO


def evaluate_model(weights_path: str, data_config: str = "data.yaml", img_size: int = 640):
    if not os.path.exists(weights_path):
        print(f"[!] Warning: Specified weights file not found: {weights_path}")
        print("[*] Falling back to base YOLOv8n weights for benchmark demonstration...")
        weights_path = "yolov8n.pt"

    print("=" * 60)
    print("      QUANTITATIVE MODEL EVALUATION BENCHMARK")
    print("=" * 60)
    print(f"[*] Model Weights : {weights_path}")
    print(f"[*] Dataset Config: {data_config}")
    print(f"[*] Input Size    : {img_size}x{img_size}")

    model = YOLO(weights_path)

    # Run validation
    metrics = model.val(
        data=data_config,
        imgsz=img_size,
        split="val",
        plots=True,
        verbose=True
    )

    print("\n" + "-" * 40)
    print("  OVERALL VALIDATION SUMMARY")
    print("-" * 40)
    print(f"  Precision (P)       : {metrics.box.mp:.4f}")
    print(f"  Recall (R)          : {metrics.box.mr:.4f}")
    print(f"  mAP @ 0.5           : {metrics.box.map50:.4f}")
    print(f"  mAP @ 0.5:0.95      : {metrics.box.map:.4f}")
    print(f"  Mean Fitness        : {metrics.fitness:.4f}")
    print("-" * 40)

    # Per-class metrics
    if hasattr(metrics.box, "maps") and len(metrics.box.maps) > 0:
        print("\n[+] Per-Class Performance (mAP@0.5:0.95):")
        names = model.names
        for cls_id, map_val in enumerate(metrics.box.maps):
            cls_name = names.get(cls_id, f"Class {cls_id}")
            print(f"    - {cls_name:<20}: {map_val:.4f}")

    print("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv8 model")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    evaluate_model(args.weights, args.data, args.imgsz)

