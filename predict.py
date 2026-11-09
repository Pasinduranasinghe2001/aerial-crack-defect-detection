"""
Inference & Visual Prediction Script.
Runs object detection on single images, directories, or video frames,
drawing bounding boxes, class labels, and confidence scores.
"""

import argparse
import os
from ultralytics import YOLO


def run_inference(
    source_path: str,
    weights_path: str = "yolov8n.pt",
    conf_thresh: float = 0.35,
    img_size: int = 640,
    save_dir: str = "runs/predict"
):
    print(f"[*] Loading model from: {weights_path}")
    model = YOLO(weights_path)

    print(f"[*] Processing source: {source_path}")
    print(f"[*] Confidence threshold: {conf_thresh:.2f}")

    results = model.predict(
        source=source_path,
        conf=conf_thresh,
        imgsz=img_size,
        save=True,
        project=save_dir,
        name="output",
        exist_ok=True
    )

    print(f"\n[✓] Inference complete. Visualizations saved in: {os.path.join(save_dir, 'output')}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on images or video")
    parser.add_argument("--source", type=str, required=True, help="Path to input image, directory, or video")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    args = parser.parse_args()

    run_inference(
        source_path=args.source,
        weights_path=args.weights,
        conf_thresh=args.conf,
        img_size=args.imgsz
    )
