import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2


MODEL_PATH = r"runs/detect/train24/weights/best.pt"
DATASET_YAML = r"C:/Users/aswan/College Project/YOLOv11_dataset_ALL/dataset.yaml"
VAL_IMAGES_DIR = Path(r"C:/Users/aswan/College Project/YOLOv11_dataset_ALL/images/val")
VAL_LABELS_DIR = Path(r"C:/Users/aswan/College Project/YOLOv11_dataset_ALL/labels/val")

FT_IMAGES_DIR = Path(r"C:/Users/aswan/College Project/hard_samples/train24_new/images")
FT_LABELS_DIR = Path(r"C:/Users/aswan/College Project/hard_samples/train24_new/labels")
SAVE_ANNOTATED_DIR = Path(r"C:/Users/aswan/College Project/hard_samples/annotated_new")

IOU_THRESHOLD = 0.75
CONFIDENCE_THRESHOLD = 0.001  # Detect every tiny prediction


def iou(box1, box2):
    """Calculate IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / (box1Area + box2Area - interArea + 1e-6)

def yolo_to_xyxy(box, w, h):
    """Convert [class, x_center, y_center, width, height] to [class, x1, y1, x2, y2]."""
    cls, x, y, bw, bh = box
    x1 = (x - bw / 2) * w
    y1 = (y - bh / 2) * h
    x2 = (x + bw / 2) * w
    y2 = (y + bh / 2) * h
    return [int(cls), x1, y1, x2, y2]


def find_false_predictions():
    FT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    FT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    torch.cuda.empty_cache()

    all_val_imgs = list(VAL_IMAGES_DIR.glob("*.jpg"))
    count = 0

    for img_path in all_val_imgs:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        label_file = VAL_LABELS_DIR / (img_path.stem + ".txt")
        if not label_file.exists():
            continue

        # Load ground truth boxes
        with open(label_file, 'r') as f:
            gt_boxes = [yolo_to_xyxy(list(map(float, line.strip().split())), w, h) for line in f.readlines()]

        # Run model
        results = model.predict(source=img_path, conf=CONFIDENCE_THRESHOLD, iou=0.45, save=False, verbose=False)
        pred = results[0]
        pred_boxes = pred.boxes.xyxy.cpu().numpy()
        pred_classes = pred.boxes.cls.cpu().numpy()

        # Format predictions
        preds = [[int(cls), *box] for cls, box in zip(pred_classes, pred_boxes)]

        matched = set()
        false_found = False

        for gt in gt_boxes:
            found_match = False
            for i, pred_box in enumerate(preds):
                if pred_box[0] == gt[0] and iou(pred_box[1:], gt[1:]) >= IOU_THRESHOLD:
                    matched.add(i)
                    found_match = True
                    break
            if not found_match:
                false_found = True  # Missed ground truth

        # Now check for false positives
        for i, pred_box in enumerate(preds):
            if i in matched:
                continue
            false_found = True  # Extra prediction not matching any GT

        if false_found:
            shutil.copy(img_path, FT_IMAGES_DIR)
            shutil.copy(label_file, FT_LABELS_DIR)

            # Draw predicted boxes
            for cls_id, x1, y1, x2, y2 in preds:
                color = (0, 0, 255) if cls_id == 1 else (255, 0, 0)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, f"{int(cls_id)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imwrite(str(SAVE_ANNOTATED_DIR / img_path.name), img)
            count += 1

    print(f"Saved {count} hard sample images for fine-tuning at:\n📁 {FT_IMAGES_DIR}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    find_false_predictions()
