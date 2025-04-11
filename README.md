# 🧠 Neural Ninjas - HackMol 6.0

## 🚧 YOLO-Based Road Quality Analysis System

This project is a deep learning-powered road monitoring tool that detects **potholes**, measures **area**, tracks movement across frames, and scores **road quality** using video input. Built with YOLOv8 and other computer vision tools.

---

## 📌 Overview

Road infrastructure plays a crucial role in public safety. Our system automates **road inspection** using a video-based approach to:

- 🔍 Detect potholes in real-time
- 🎯 Measure the area of damage
- 📊 Track detected potholes across frames
- 🧮 Score road quality based on density, area, and location of potholes

---

## 🛠 Tech Stack

| Layer          | Tools Used                                           |
|----------------|------------------------------------------------------|
| Frontend       | Streamlit / HTML (optional interface for demo)       |
| Backend        | Python, OpenCV, NumPy, Pandas                        |
| Deep Learning  | YOLOv8 (Ultralytics), PyTorch                        |
| Visualization  | Matplotlib, Seaborn (for plots, charts)             |
| Deployment     | Jupyter Notebook / Streamlit                        |

---

## 🚀 Setup Instructions

### 1. Clone the Repo
git clone https://github.com/mee-vishal/neural_ninjas_hackmol6.O.git
cd neural_ninjas_hackmol6.O


### 2. Create a Virtual Environment (recommended)
bash
Copy
Edit
python -m venv venv
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate


### 3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt


⚠️ Make sure you have Python 3.8+ and torch, opencv-python, ultralytics installed.

### 4. Run the YOLOv8 Detection Notebook

ml code/yolov8-roadpothole-detection-main/yolov8_instance_segmentation_on_custom_dataset.ipynb

### 🧪 Features
✅ YOLOv8 Instance Segmentation

📏 Area Estimation for detected potholes

📍 Tracking using object IDs across frames

🧠 Road Quality Scoring based on analyzed metrics

📹 Video and Image Input Support


### 📂 Folder Structure
.
neural_ninjas_hackmol6.O/ ├── ml code/ │ └── yolov8-roadpothole-detection-main/ │ └── yolov8_instance_segmentation_on_custom_dataset.ipynb ├── assets/ │ └── sample images for demo ├── README.md └── requirements.txt


### 💡 Future Improvements
🔌 Real-time deployment via camera stream

📱 Android App integration

📡 Cloud-based dashboard for municipality data collection

---

## 🙌 Thank You!

Thank you for checking out our project! We hope it inspires innovative solutions for smart and safe road infrastructure. If you found this useful or interesting, feel free to ⭐ the repo, contribute, or reach out to the team.

Together, let's make roads smarter and safer! 🚗🛣️

— Team Neural Ninjas 💡

