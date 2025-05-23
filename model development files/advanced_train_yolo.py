from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

def train():
    model = YOLO("yolo11s.pt")  # Load small pre-trained YOLOv11 model

    model.train(
        data="C:/Users/aswan/College Project/new/dataset.yaml",  # Path to dataset.yaml
        epochs=200,                # Full 200 epochs as planned
        patience=50,               # Early stopping patience
        batch=16,                  # Manually set batch size
        imgsz=640,                 # Input image size
        device="cuda",             # Use GPU
        cache=False,               # Don't cache images into RAM

        # Learning Rate Scheduling
        lr0=1e-3,                  # Lower initial learning rate (papers suggestion)
        lrf=0.01,                  # Final learning rate fraction

        optimizer="SGD",           # SGD optimizer (papers used SGD + momentum)
        momentum=0.937,            # Standard YOLO momentum
        weight_decay=0.0005,       # Standard YOLO weight decay
        dropout=0.1,                     # add dropout

        # Data Augmentation
        warmup_epochs=5,           # Warmup training for first 5 epochs
        close_mosaic=10,           # Disable mosaic augmentation in last 10 epochs
        hsv_h=0.015,               # Light color jittering
        scale=0.5,                 # Image scaling (0.5 = good augmentation)
        degrees=0.0,               # No random rotations (optional for better stability)
        translate=0.1,             # Light translation

        # Other Training Settings
        single_cls=False,          # Multi-class (falls vs normal)
        pretrained=True,           # Continue fine-tuning from weights
        val=True,                  # Validate after every epoch
    )

if __name__ == '__main__':
    train()
