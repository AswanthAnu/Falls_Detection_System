# 🧓 Real-Time Fall Detection System using YOLOv11
> A final year MSc in Data Analytics project submitted to Dublin Business School  
> Developed by **Aswanth Manoharan**

## 📌 Project Overview

This project presents a real-time fall detection system for elderly home care using **YOLOv11** object detection. It is designed to **maximize fall sensitivity (high recall)** while operating efficiently on live video streams (e.g., RTSP from mobile/IP camera). The system is implemented using a combination of **Python, PyTorch, Ultralytics YOLOv11, OpenCV, and Django** for web deployment.

## 🧠 Core Features

- Fine-tuned YOLOv11 model with high recall on falls
- Buffered fall detection logic to avoid false alerts
- Real-time RTSP camera integration using Django WebSocket
- Bounding box prediction with confidence overlay
- Easy-to-integrate and modular code

## 📁 Repository Structure

```
📦FallDetectionSystem
 ┣ 📂notebooks
 ┃ ┗ 📜Fall_Detection_Modeling_using_YOLOV11.ipynb
 ┣ 📂scripts
 ┃ ┣ 📜advanced_train_yolo.py
 ┃ ┣ 📜balanced_augmentation_utils.py
 ┃ ┣ 📜caucafall_dataset_structuring.ipynb
 ┃ ┣ 📜false_prediction_finder.py
 ┃ ┣ 📜hard_sample_finetuning_on_train23.py
 ┃ ┣ 📜invert_falls_label.py
 ┃ ┗ 📜model_comparison.py
 ┣ 📂Falldetection_web
 ┃ ┣ 📜Django backend and frontend code
 ┣ 📂sample_results
 ┃ ┗ 📜Sample annotated prediction images
 ┣ 📜README.md
 ┗ 📜requirements.txt
```

## 🔧 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/FallDetectionSystem.git
cd FallDetectionSystem
```

### 2️⃣ Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

> Make sure **PyTorch with GPU** is installed if you want real-time performance:  
[Get PyTorch with CUDA →](https://pytorch.org/get-started/locally/)

## 🧪 Running the Model

### ➤ Training the YOLOv11 Model
Use the scripts in the `/scripts` folder to train:
```bash
python scripts/advanced_train_yolo.py
```

### ➤ Fine-tuning on Hard Samples
```bash
python scripts/hard_sample_finetuning_on_train23.py
```

## 🌐 Running the Web App (Real-Time Detection)

### 1. Move to the Django app folder:
```bash
cd Falldetection_web
```

### 2. Run the server
```bash
python manage.py runserver
```

### 3. Open your browser and go to:
```
http://localhost:8000
```

Paste the RTSP link from your mobile/IP camera (e.g., IP Webcam app) to begin live prediction.

## 🎯 Key Results

| Metric               | Train23 | Train23 Fine-Tune |
|---------------------|---------|-------------------|
| mAP@0.5             | 0.9866  | 0.9835            |
| mAP@0.5:0.95        | 0.8096  | 0.9212            |
| Precision           | 0.9482  | 0.9583            |
| Recall              | 0.9518  | 0.9891            |
| False Negatives     | 10      | 7                 |
| Inference Time/frame| ~2.3ms  | ~2.3ms            |

## 📦 Datasets

> 📁 **Dataset not included due to size.**  
However, all datasets are publicly available and links are provided below:

- **CAUCAFall**: [Mendeley Data](https://data.mendeley.com/datasets/7w7fccy7ky/4)
- **HumanPose**: [Roboflow](https://universe.roboflow.com/weile-tech/humanpose-prf6h)
- **LE2I (Augmented)**: [Roboflow](https://universe.roboflow.com/new-workspace-qfcus/le2i)

## 📸 Screenshots

UI Screenshot
![web1](https://github.com/user-attachments/assets/ed89b1f2-0983-430c-86a0-7d59954678bc)
![web2](https://github.com/user-attachments/assets/8bc13609-9c8d-499f-9394-4c36b018f6f8)
![web15](https://github.com/user-attachments/assets/9c88662f-8a49-4f27-a2ef-3eb10f0f78aa)
![web23](https://github.com/user-attachments/assets/6b7365f1-b2cb-4f26-9944-09174eed825c)

Prediction Comparison
![comparison_model](https://github.com/user-attachments/assets/45fa3fa2-bc3d-4f18-b65d-1e7fbe2370d0)


## 🙏 Acknowledgements

- Supervisor: Dr. Baidyanath Biswas, Dublin Business School
- Open Source Datasets: CAUCAFall, HumanPose, LE2I
- Frameworks: PyTorch, Ultralytics YOLOv11, Django, OpenCV
- Community Support: AI/ML forums, Roboflow, Medium
- My friends and family for real-world testing support

## 📜 License

This project is for academic research purposes. Attribution required for reuse.
