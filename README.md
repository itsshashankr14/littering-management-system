# Littering Detection & Vehicle Number Plate Recognition System
---
## Overview

In many cities, the problem of littering from moving vehicles is growing. Traditional surveillance systems struggle to catch these events in real-time and identify the culprits. This project aims to solve this issue using a computer vision pipeline that performs the following:

1. **Plastic Waste Detection** – A YOLOv8 model trained to detect plastic items such as bottles and wrappers thrown from vehicles.
2. **Vehicle Detection** – Identifies and localizes the vehicle responsible for the action using a separate YOLOv8 model.
3. **License Plate Detection and OCR** – The number plate is detected from the vehicle region, cropped, and passed to an OCR module to extract the alphanumeric registration number.
4. **Data Logging and Evidence Storage** – Cropped images of the plastic waste, vehicle, and number plate are saved along with the recognized text for further verification or legal action.
5. **Real-time Feed Processing** – The system works on a live video stream or recorded footage for continuous monitoring.

This solution can be used by municipal bodies and smart city projects to automate litter detection, issue fines, and build a cleaner environment.
---
## Features

- Plastic Waste Detection using YOLOv8
- Vehicle Detection
- License Plate Detection
- OCR for extracting vehicle number
- Cropping and saving evidence images
- Organized storage by date and time
- Real-time camera feed analysis
---
## 🛠️ Tech Stack

| Component        | Technology          |
|------------------|---------------------|
|  AI Model        | YOLOv8 (Ultralytics) |
|  OCR             | Tesseract OCR       |
|  Backend         | Python              |
|  Image Processing | OpenCV              |
|  Database (optional) | SQLite or Firebase |
|  Deployment      | Localhost / Cloud VM |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/itsshashankr14/littering-management-system.git
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
## 🎥 Demo
![Dashboard](image.png)
![Trash and vehicle detection](image-1.png)

---

## 🔮 Future Enhancements

- GPS tagging for each incident
- Email alerts or SMS notifications
- Integration with city surveillance systems
- Custom-trained models for improved accuracy

---

## 🙌 Contributors

- **Shashank Gowda R** – [Folow on Github](https://github.com/itsshashankr14)
- **Mohan** – [Follow on Github](https://github.com/mohan1345)

---

## 📜 License

MIT License. Feel free to use and improve this project for educational or civic purposes.
