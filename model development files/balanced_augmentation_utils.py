import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import random
import os


def load_yolo_labels(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x, y, w, h = map(float, line.strip().split())
            boxes.append([x, y, w, h, int(class_id)])
    return boxes

def save_yolo_labels(save_path, bboxes):
    with open(save_path, 'w') as f:
        for box in bboxes:
            x, y, w, h, class_id = box
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def yolo_to_albumentations(bboxes, img_width, img_height):
    return [
        [
            (x - w / 2) * img_width,
            (y - h / 2) * img_height,
            (x + w / 2) * img_width,
            (y + h / 2) * img_height,
            class_id
        ]
        for x, y, w, h, class_id in bboxes
    ]

def albumentations_to_yolo(bboxes, img_width, img_height):
    results = []
    for x_min, y_min, x_max, y_max, class_id in bboxes:
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        results.append([x_center, y_center, width, height, class_id])
    return results


AUGMENTATIONS = A.Compose(
    [
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.Affine(scale=(0.9, 1.1), shear={"x": (-15, 15), "y": (-15, 15)}, p=0.4),

        # Photometric
        A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
        A.HueSaturationValue(15, 20, 20, p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.ToGray(p=0.1),

        # Degradations
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.Downscale(scale_min=0.3, scale_max=0.5, interpolation=0, p=0.2),

        # Occlusion & crop
        A.RandomSizedBBoxSafeCrop(512, 512, p=0.4),
        A.CoarseDropout(max_holes=1, max_height=50, max_width=50, min_holes=1, min_height=30, min_width=30, fill_value=0, p=0.3),
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
)


def augment_to_target(
    image_dir,
    label_dir,
    output_image_dir,
    output_label_dir,
    target_fall,
    target_nonfall
):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_image_dir = Path(output_image_dir)
    output_label_dir = Path(output_label_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    fall_samples = []
    nonfall_samples = []

    for lbl_file in os.listdir(label_dir):
        if not lbl_file.endswith('.txt'):
            continue
        lbl_path = label_dir / lbl_file
        img_path = image_dir / (lbl_file.replace('.txt', '.jpg'))
        if not img_path.exists():
            img_path = img_path.with_suffix('.png')
        if not img_path.exists():
            continue

        boxes = load_yolo_labels(lbl_path)
        class_ids = set(box[4] for box in boxes)
        if 1 in class_ids:
            fall_samples.append((img_path, lbl_path))
        else:
            nonfall_samples.append((img_path, lbl_path))

    print(f"Original Fall: {len(fall_samples)}, Non-Fall: {len(nonfall_samples)}")

    def do_augmentation(samples, count_needed, class_name):
        if count_needed <= 0:
            print(f"No need to augment {class_name}")
            return
        for i in range(count_needed):
            tries = 0
            while tries < 5:
                image_path, label_path = random.choice(samples)
                image = cv2.imread(str(image_path))
                if image is None:
                    tries += 1
                    continue

                h, w = image.shape[:2]
                boxes = load_yolo_labels(label_path)
                alb_boxes = yolo_to_albumentations(boxes, w, h)
                class_labels = [b[4] for b in alb_boxes]

                try:
                    augmented = AUGMENTATIONS(image=image, bboxes=[b[:4] for b in alb_boxes], class_labels=class_labels)
                    aug_img = augmented['image']
                    aug_boxes = [list(b) + [cls] for b, cls in zip(augmented['bboxes'], class_labels)]
                    yolo_boxes = albumentations_to_yolo(aug_boxes, aug_img.shape[1], aug_img.shape[0])

                    # Save
                    base_name = image_path.stem + f'_aug_{class_name}_{i}'
                    out_img = output_image_dir / f"{base_name}.jpg"
                    out_lbl = output_label_dir / f"{base_name}.txt"
                    cv2.imwrite(str(out_img), aug_img)
                    save_yolo_labels(out_lbl, yolo_boxes)

                    print(f"Augmented {class_name}: {base_name}")
                    break
                except Exception as e:
                    print(f"Skip: {image_path.name} -> {e}")
                    tries += 1

    do_augmentation(fall_samples, target_fall - len(fall_samples), 'fall')
    do_augmentation(nonfall_samples, target_nonfall - len(nonfall_samples), 'nonfall')

    print("🎉 Augmentation complete!")



if __name__ == "__main__":
    # humanpose dataset
    augment_to_target(
    image_dir=r"C:\Users\aswan\College Project\Humanpose dataset\images\train",
    label_dir=r"C:\Users\aswan\College Project\Humanpose dataset\labels\train",
    output_image_dir=r"C:\Users\aswan\College Project\Agumented_dataset_yolo\images\train",
    output_label_dir=r"C:\Users\aswan\College Project\Agumented_dataset_yolo\labels\train",
    target_fall=20000,
    target_nonfall=19500
)
    


    # caucafall dataset
    augment_to_target(
    image_dir=r"C:\Users\aswan\College Project\CaucaFall dataset\images\train",
    label_dir=r"C:\Users\aswan\College Project\CaucaFall dataset\labels\train",
    output_image_dir=r"C:\Users\aswan\College Project\Agumented_dataset_yolo\images\train",
    output_label_dir=r"C:\Users\aswan\College Project\Agumented_dataset_yolo\labels\train",
    target_fall=13500,
    target_nonfall=13500
)