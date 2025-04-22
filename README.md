# 🚯 Littering Detection & Vehicle Identification System

An AI-powered solution to detect littering behavior from vehicles in real-time, identify the vehicle number using license plate recognition, and log incidents for further action.


## 🧠 Project Overview

This project aims to automate the process of identifying littering activities from moving vehicles using CCTV or mounted cameras. It uses a combination of object detection (YOLOv8), number plate detection (OpenCV), and OCR (Tesseract) to:

- Detect if trash is being thrown from a vehicle
- Capture and recognize the vehicle's license plate
- Log the event for record-keeping or law enforcement purposes

---

## ✨ Features

- 📹 Real-time object detection using **YOLOv8**
- 🚗 Vehicle detection and tracking
- 🗑️ Trash object classification
- 🔍 License plate detection and OCR with **Tesseract**
- 📁 Incident logging with timestamp and image evidence
- 🔒 Privacy-focused and efficient

---

## 🛠️ Tech Stack

| Component        | Technology          |
|------------------|---------------------|
| 🧠 AI Model       | YOLOv8 (Ultralytics) |
| 🧪 OCR            | Tesseract OCR       |
| 🔧 Backend        | Python              |
| 🖼️ Image Processing | OpenCV              |
| 💾 Database (optional) | SQLite or Firebase |
| ☁️ Deployment     | Localhost / Cloud VM |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/littering-detection.git
cd littering-detection
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Linux/Mac
.venv\Scripts\activate         # On Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 Weights

Download the pre-trained model weights from [Ultralytics](https://github.com/ultralytics/ultralytics) and place them in the `models/` directory.

```bash
models/
├── vehicle/
│   └── yolov8s.pt
├── trash/
│   └── yolov8s.pt
```

### 5. Run the App

```bash
python main.py
```

---

## 🧰 How it Works

1. **Video Feed Input**  
   - Either live camera or pre-recorded footage.

2. **Object Detection**  
   - Detects vehicles and litter using YOLOv8 models.

3. **Littering Check**  
   - Identifies if trash is being thrown outside of a detected vehicle's bounding box.

4. **License Plate Recognition**  
   - Localizes the number plate region and extracts text using Tesseract OCR.

5. **Logging**  
   - Stores images and OCR results with timestamp for each littering event.

---


## 📁 Folder Structure

```
littering-detection/
│
├── models/
│   ├── vehicle/
│   └── trash/
├── static/
│   └── logs/         # Detected incident screenshots
├── main.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## 🔮 Future Enhancements

- 🗺️ GPS tagging for each incident
- 🌐 Dashboard for analytics and reporting
- 📧 Email alerts or SMS notifications
- 🤖 Integration with city surveillance systems
- 🧠 Custom-trained models for improved accuracy

---

## 🙌 Contributors

- **Shashank Gowda R** – [GitHub](https://github.com/itsshashankr14)\
- **Mohan** – [GitHub](https://github.com/mohan1345)

---

## 📜 License

MIT License. Feel free to use and improve this project for educational or civic purposes.
