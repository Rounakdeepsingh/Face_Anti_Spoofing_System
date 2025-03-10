# Face_Anti_Spoofing_System
Face Anti-Spoofing System
🚀 Overview

The Face Anti-Spoofing System is an advanced deep learning-based application designed to detect and prevent face spoofing attacks such as photographs, video replays, and mask attempts. It distinguishes between real and fake faces in real-time to ensure secure facial authentication, making it suitable for integration into high-security biometric systems.

🛠️ Tech Stack & Tools

Python 3.x
YOLOv8 (Ultralytics)
OpenCV (Image and Video Processing)
NumPy (Numerical operations)
TensorFlow / PyTorch (Model training - specify if used)
Google Colab (For training on cloud GPU)
Matplotlib (Optional - Visualization)
Streamlit / Flask (Optional for GUI - specify if added)

✨ Features

✅ Real-time face anti-spoofing detection using webcam or video feed.
✅ Detects spoof attempts like printed photos, video replays, and masks.
✅ Dataset collection module for real and fake samples.
✅ Custom dataset support and YOLOv8 model training.
✅ Pre-trained model for immediate testing and validation.
✅ Highly accurate, lightweight, and ready for deployment.
✅ Extendable for mobile and web-based deployment (optional future scope).

📂 Project Structure

Face_Anti_Spoofing_System/
│
├── datacollect/          # Scripts & data for dataset collection
├── real/                # Folder to store real face images/videos
├── fake/                # Folder to store fake face images/videos
├── trained_model/       # Pre-trained YOLOv8 weights/models
├── dataset.yaml         # Dataset configuration for YOLO training
├── main.py              # Main application for detection
├── train.py             # Training pipeline for custom datasets
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation (this file)

⚙️ Installation & Setup

1. Clone the Repository
git clone https://github.com/Rounakdeepsingh/Face_Anti_Spoofing_System.git
cd Face_Anti_Spoofing_System

3. Create and Activate Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate        # For Windows

4. Install Dependencies
pip install -r requirements.txt

5. Download Pre-trained Models (If not present)
Place YOLOv8 trained model weights (e.g., best.pt) into the trained_model/ directory.
🚀 Usage
1. Real-Time Face Anti-Spoofing Detection
   
python main.py
This will activate the webcam and start real-time spoof detection.
Detected faces will be labeled as "Real" or "Fake" along with a confidence score.

2. Collect Dataset (Real or Fake Faces)
python datacollect.py
Follow on-screen instructions to capture real or fake face datasets for training purposes.

3. Train YOLOv8 Model on Custom Dataset
python train.py
Make sure dataset.yaml is properly configured to point to training and validation datasets.
📈 Dataset Structure
bash
Copy
Edit
dataset/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
└── labels/
    ├── train/          # YOLO format annotations for training
    └── val/            # YOLO format annotations for validation
dataset.yaml Example:

data.yaml
path: ./dataset
train: images/train
val: images/val

nc: 2
names: ['real', 'fake']
✅ Example Outputs
Live Feed Detection:
Displays real-time bounding boxes labeled as "Real" or "Fake".
Shows detection confidence percentages.
Training & Evaluation:
Training logs including loss, precision, and recall graphs.

📷 Screenshots 
Real Face Detected
![WhatsApp Image 2025-03-01 at 21 54 45_ce6d3be8](https://github.com/user-attachments/assets/bca0a7d5-8116-4482-9d4a-b1795f234f9d)
Fake Face Detected
![image](https://github.com/user-attachments/assets/ce8473af-8c1a-48ec-8522-4b4963764bca)

💡 Future Work

 Add mobile app integration (Android/iOS).
 Integrate web interface using Streamlit/Flask.
 Optimize model size for edge devices (e.g., Raspberry Pi).
 Extend to 3D mask attack detection.

 🤝 Contributors
Rounakdeep Singh - Lead Developer
Saran MS 

📜 License
This project is licensed under the MIT License — feel free to use and modify for personal and commercial use.

📬 Contact
For issues or queries, please reach out at:
📧 rounakdeepsingh54@gmail.com 

⭐ Support
If you find this project helpful, give it a ⭐ on GitHub to support ongoing improvements!

⚙️ Optional Badges 
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red)
