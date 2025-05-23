from ultralytics import YOLO
import torch


MODEL_PATH = "runs/detect/train23/weights/best.pt"  # best model
HARD_YAML = "C:/Users/aswan/College Project/hard_samples/train23_new/dataset.yaml"  # Fine-tune config


def finetune_on_hard_samples():
    torch.cuda.empty_cache()
    model = YOLO(MODEL_PATH)  

    model.train(
    data=HARD_YAML,
    epochs=20,
    lr0=1e-5,
    optimizer="AdamW",
    freeze=10,          
    imgsz=640,
    device="cuda",
    batch=16,
    cache=True,
    close_mosaic=0,
    patience=5,
    resume=False,
    name="hard_sample_train23_ft"
    )


if __name__ == "__main__":
    finetune_on_hard_samples()

