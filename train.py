"""
Model Training Script for Aerial Crack and Surface Defect Detection.
Uses Ultralytics YOLOv8 architecture fine-tuned on aerial drone inspection datasets.
"""

import argparse
import os
import torch
from ultralytics import YOLO


def train_model(
    data_config: str = "data.yaml",
    model_type: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    learning_rate: float = 0.01,
    project_dir: str = "runs/train",
    name: str = "aerial_defect_yolov8"
):
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing training on device: {device}")
    print(f"[*] Base model: {model_type}")
    print(f"[*] Dataset config: {data_config}")
    print(f"[*] Image resolution: {img_size}x{img_size} | Epochs: {epochs} | Batch size: {batch_size}")

    # Load pretrained model
    model = YOLO(model_type)

    # Train model
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        device=device,
        project=project_dir,
        name=name,
        optimizer="AdamW",
        save=True,
        plots=True,
        verbose=True
    )

    print("\n[✓] Training complete.")
    print(f"[✓] Best weights saved at: {os.path.join(project_dir, name, 'weights', 'best.pt')}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on aerial defect dataset")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained weights (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    args = parser.parse_args()

    train_model(
        data_config=args.data,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        learning_rate=args.lr
    )
