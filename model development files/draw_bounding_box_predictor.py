import cv2
import os
from pathlib import Path
from ultralytics import YOLO 

# Load your trained model
model = YOLO("runs/detect/hard_sample_ft/weights/best.pt")

def predict_and_save_images(input_folder, output_folder):
    try:
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Loop through all the images in the folder
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)

            # Check if it's an image file
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image = cv2.imread(image_path)

                # Run prediction
                results = model.predict(
                    source=image,
                    device='cuda',
                    conf=0.50,  
                    iou=0.45,    
                    max_det=5,  # Avoid clutter
                    verbose=False
                )

                # Draw bounding boxes and labels
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        label = "Fall" if class_id == 1 else "Non-Fall"
                        color = (0, 0, 255) if class_id == 1 else (0, 255, 0)

                        # Draw bounding box and label
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(image, f'{label} {confidence*100:.1f}%',
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, color, 2)

                # Save the output image
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, image)

                print(f"Processed and saved: {output_image_path}")

        print(f"All images processed and saved in: {output_folder}")
    except Exception as e:
        print(f"Error during image prediction: {e}")


input_folder = r"C:\Users\aswan\College Project\hard_samples\train_hard_new\images"  
output_folder = r"C:\Users\aswan\College Project\Output on prediction"  

predict_and_save_images(input_folder, output_folder)
