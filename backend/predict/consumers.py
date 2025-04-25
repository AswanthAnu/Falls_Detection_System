import json

import base64
import torch
import cv2
import numpy as np
from channels.generic.websocket import WebsocketConsumer
from ultralytics import YOLO  # Required even for torchscript load
from pathlib import Path

# Path to the directory containing this file
# This is used to load the YOLOv11 model from the correct path
BASE_DIR = Path(__file__).resolve().parent.parent

class PredictConsumer(WebsocketConsumer):

    def connect(self):
        self.accept()

        try:
            model_path = BASE_DIR / "best.torchscript"
            self.model = YOLO(model_path, task='detect')
            print("✅ YOLO model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

        self.send(json.dumps({
            'type': 'connection_established',
            'message': '🔗 WebSocket connected!'
        }))

    def receive(self, text_data):
        try:
            try:
                data = json.loads(text_data)
            except json.JSONDecodeError:
                self.send(json.dumps({'type': 'error', 'message': 'Invalid JSON format.'}))
                return
            base64_image = data.get('image')


            if not base64_image:
                self.send(json.dumps({'type': 'error', 'message': 'No image received.'}))
                return

            img_data = base64.b64decode(base64_image.split(',')[1])
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            prediction, annotated = self.predict_yolo(frame)

            _, buffer = cv2.imencode('.jpg', annotated)
            encoded = base64.b64encode(buffer).decode('utf-8')

            self.send(json.dumps({
                'type': 'prediction',
                'prediction': prediction,
                'frame': f"data:image/jpeg;base64,{encoded}"
            }))

        except Exception as e:
            self.send(json.dumps({'type': 'error', 'message': f'Processing error: {e}'}))

    def predict_yolo(self, frame):
        """
        Predict fall using YOLOv11 model and annotate frame.
        """
        results = self.model.predict(frame, conf=0.4, iou=0.5, verbose=False)
        print("Model prediction completed.")

        fall_detected = False
        annotated = frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = "Fall" if cls_id == 1 else "Non-Fall"
                color = (0, 0, 255) if cls_id == 1 else (0, 255, 0)

                if cls_id == 1:
                    fall_detected = True

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return ("Fall Detected!" if fall_detected else "No Fall Detected"), annotated
