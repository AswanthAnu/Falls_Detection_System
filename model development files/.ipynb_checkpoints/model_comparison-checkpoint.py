import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


# Paths to your fine-tuned models
MODEL1_PATH = "runs/detect/train23/weights/best.pt"
MODEL2_PATH = "runs/detect/hard_sample_train23_ft/weights/best.pt"

# Input images folder
IMAGES_DIR = Path(r"C:/Users/aswan/College Project/Unseen_data_for_model_comparison")
# Output folder to save visualized images
OUTPUT_DIR = Path(r"C:/Users/aswan/College Project/Unseen_data_model _comparison_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors for bounding boxes (different for each model)
COLOR_MODEL1 = (0, 0, 255)    # Red for Model 1
COLOR_MODEL2 = (0, 255, 0)    # Green for Model 2


def load_models():
    print("Loading models...")
    model1 = YOLO(MODEL1_PATH)
    model2 = YOLO(MODEL2_PATH)
    print("Both models loaded successfully.")
    return model1, model2

def predict_and_draw(image_path, model1, model2):
    img = cv2.imread(str(image_path))

    # Predict with Model 1
    result1 = model1.predict(source=img, conf=0.5, save=False, verbose=False)[0]
    boxes1 = result1.boxes

    # Predict with Model 2
    result2 = model2.predict(source=img, conf=0.5, save=False, verbose=False)[0]
    boxes2 = result2.boxes

    # Draw Model 1 predictions
    for box in boxes1:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"Model1: {result1.names[cls_id]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_MODEL1, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MODEL1, 2)

    # Draw Model 2 predictions
    for box in boxes2:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"Model2: {result2.names[cls_id]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_MODEL2, 2)
        cv2.putText(img, label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MODEL2, 2)

    return img

def process_images(images_dir, model1, model2):
    all_images = list(images_dir.glob("*.png"))
    print(f"Found {len(all_images)} images to process...")

    for img_path in all_images:
        print(f"Processing: {img_path.name}")
        combined_img = predict_and_draw(img_path, model1, model2)
        save_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(save_path), combined_img)

    print(f"Done! Input gets from : {IMAGES_DIR}")
    print(f"Done! Combined outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    model1, model2 = load_models()
    process_images(IMAGES_DIR, model1, model2)
