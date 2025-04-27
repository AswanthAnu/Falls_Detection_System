import json
import cv2
import base64
import time
import threading
from channels.generic.websocket import WebsocketConsumer
from ultralytics import YOLO
from pathlib import Path
from .fall_detection_logic import FallDetection

BASE_DIR = Path(__file__).resolve().parent.parent

class PredictConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        print("🔌 WebSocket connected, waiting for RTSP URL...")

        # Load YOLO model once
        try:
            model_path = BASE_DIR / "best.torchscript"
            self.model = YOLO(model_path, task='detect')
            print("✅ YOLO model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")

        # Initialize fall detection system
        self.fall_detection = FallDetection()

        self.running = False
        self.capture_thread = None

    def disconnect(self, close_code):
        print(f"🔌 WebSocket disconnected with code {close_code}")
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join()
            print("🛑 Frame capturing thread stopped cleanly.")

    def receive(self, text_data):
        try:
            data = json.loads(text_data)

            if 'rtspUrl' in data:
                rtsp_url = data['rtspUrl']
                print(f"🎯 Received RTSP URL: {rtsp_url}")

                self.running = True
                self.capture_thread = threading.Thread(target=self.capture_frames, args=(rtsp_url,))
                self.capture_thread.start()

            else:
                self.send(json.dumps({'type': 'error', 'message': 'RTSP URL not provided.'}))
                print("⚠️ RTSP URL not found in message.")

        except json.JSONDecodeError:
            self.send(json.dumps({'type': 'error', 'message': 'Invalid JSON format.'}))
            print("❌ Failed to decode JSON.")

    def capture_frames(self, rtsp_url):
        while self.running:
            try:
                cap = cv2.VideoCapture(rtsp_url)

                if not cap.isOpened():
                    print("❌ Unable to open RTSP stream. Retrying in 5 seconds...")
                    self.send(json.dumps({'type': 'error', 'message': 'Cannot connect to RTSP stream.'}))
                    time.sleep(5)
                    continue

                print("🎥 RTSP stream opened successfully.")

                last_frame_time = 0
                desired_fps = 2  # FPS limit to reduce server load

                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠️ Failed to read frame from RTSP stream. Retrying...")
                        break  # Try reconnecting

                    # FPS Limiting
                    now = time.time()
                    if now - last_frame_time < 1.0 / desired_fps:
                        continue
                    last_frame_time = now

                    prediction, annotated_frame = self.predict_yolo(frame)

                    # Encode annotated frame
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                    # Process the fall detection logic
                    if prediction == "Fall Detected!":
                        self.fall_detection.add_frame(True)
                    else:
                        self.fall_detection.add_frame(False)

                    if self.fall_detection.can_detect_fall():
                        if self.fall_detection.should_trigger_alarm():
                            self.fall_detection.record_fall()
                            self.send(json.dumps({
                                'type': 'fall_alarm',
                                'message': "ALERT: Fall Detected!",
                                'frame': f"data:image/jpeg;base64,{jpg_as_text}"
                            }))

                    self.send(json.dumps({
                        'type': 'prediction',
                        'prediction': prediction,
                        'frame': f"data:image/jpeg;base64,{jpg_as_text}"
                    }))

                cap.release()
                print("🛑 RTSP stream closed, attempting to reconnect...")

            except Exception as e:
                print(f"❌ Exception in capture thread: {e}")
                time.sleep(5)

    def predict_yolo(self, frame):
        results = self.model.predict(frame, conf=0.4, iou=0.5, verbose=False)

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
