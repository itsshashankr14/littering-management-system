import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context
import sqlite3

import re
import cv2
import easyocr
import numpy as np
import supervision as sv
from ultralytics import YOLO
from datetime import datetime
from supervision.detection.core import Detections
from supervision.detection.utils import merge_data

import numpy as np
import math
import time

conn = sqlite3.connect('detections.db')
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_file TEXT NOT NULL,
        car_number TEXT NOT NULL,
        time TEXT NOT NULL,
        output_file TEXT NOT NULL
    );"""
)

conn.commit()

reader = easyocr.Reader(['en'])

detected_cars = {}
detected_license_plates = {}

def calculate_centroid(xyxy):
    xmin, ymin, xmax, ymax = xyxy
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    return (cx, cy)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def detect_trash_thrown(car_detections, trash_detections, edge_to_edge_distance):
    whole_car_detections = car_detections
    whole_trash_detections = trash_detections
    try:
        car_detections = car_detections.xyxy
        trash_detections = trash_detections.xyxy

        for trash_detection in trash_detections:
            trash_centroid = calculate_centroid(trash_detection)

            for car_detection in car_detections:
                car_centroid = calculate_centroid(car_detection)
                distance = calculate_distance(car_centroid, trash_centroid)
                if distance < edge_to_edge_distance:
                    
                    return True, (int(trash_centroid[0]), int(trash_centroid[1])), (int(car_centroid[0]), int(car_centroid[1])), distance, whole_car_detections.tracker_id[0], whole_trash_detections.tracker_id[0]
    except Exception as e:
        pass    
    return False, (0, 0), (0, 0), 0, -1, -1


def check_format(input_string):
    cleaned_string = re.sub(r'[^A-Za-z0-9]', '', input_string)
    if 9 < len(cleaned_string) < 14:
        return cleaned_string
    else:
        return ""


def extract_the_ROI(detection, frame):
    xmin, ymin, xmax, ymax = detection
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    roi = frame[ymin:ymax, xmin:xmax]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    results = reader.readtext(binary)

    for (bbox, text, prob) in results:
        return check_format(text) 

    
vehical_model = YOLO("models/vehical/yolov8s.pt", verbose=False)
trash_model = YOLO("models/train10/weights/last.pt", verbose=False)
license_plate_model = YOLO("models/license_plate/weights/best.pt", verbose=False)


def insert_detection(conn, detection):
    sql = ''' INSERT INTO detections(input_file, car_number, time, output_file)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, detection)
    conn.commit()
    return cur.lastrowid


def start_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    codec='mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_name = 'output/output_{}.mp4'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    out = cv2.VideoWriter(output_name, fourcc, int(fps), (width, height))
    edge_to_edge_distance = calculate_distance((0, 0), (width, height)) // 3
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    vehical_classes = vehical_model.names
    trash_classes = trash_model.names 
    license_plate_classes = license_plate_model.names


    while cap.isOpened():
        success, frame = cap.read()

        if success:

            vehical_results = vehical_model.track(frame, persist=True, conf=0.7, classes=[1, 2, 3, 5, 6, 7], verbose=False)
            vehical_detections = sv.Detections.from_ultralytics(vehical_results[0])
            vehical_labels = [f"{vehical_classes[class_id]} {confidence:0.2f} Tracker ID:{tracker_id}" for _, _, confidence, class_id, tracker_id, _ in vehical_detections]

            annotated_image = bounding_box_annotator.annotate(scene=frame, detections=vehical_detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=vehical_detections, labels=vehical_labels)


            if vehical_detections.class_id is not None and len(list(vehical_detections.class_id)) > 0:
                trash_results = trash_model.track(frame, persist=True, conf=0.6, verbose=False)
                trash_detections = sv.Detections.from_ultralytics(trash_results[0])
                trash_labels = [f"{trash_classes[class_id]} {confidence:0.2f} Tracker ID:{tracker_id}" for _, _, confidence, class_id, tracker_id, _ in trash_detections]
                
                is_trash_trown, trash_centroid, car_centroid, distance, car_id, trash_id = detect_trash_thrown(vehical_detections, trash_detections, edge_to_edge_distance)

                if is_trash_trown:
                    annotated_image = cv2.line(annotated_image, trash_centroid, car_centroid, (255, 255, 0) , 5)
                    if car_id is not None and car_id not in detected_cars.keys() and car_id in detected_license_plates.keys():
                        detected_cars[car_id] = {
                            'license_plate': detected_license_plates[car_id],
                            'time': datetime.now()
                        }

                annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=trash_detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=trash_detections, labels=trash_labels)

            if vehical_detections.class_id is not None and len(list(vehical_detections.class_id)) > 0:
                license_plate_results = license_plate_model.track(frame, persist=True, conf=0.7, verbose=False)
                license_plate_detections = sv.Detections.from_ultralytics(license_plate_results[0])
                license_plate_labels = []
                
                for xyxy, _, confidence, class_id, tracker_id, _ in license_plate_detections:
                    plate_number = extract_the_ROI(xyxy, frame)
                    license_plate_labels.append(f"{plate_number} {license_plate_classes[class_id]} {confidence:0.2f} Tracker ID:{tracker_id}")
                    for vehical_detection in vehical_detections:
                        if vehical_detection[0][0] < xyxy[0] < vehical_detection[0][2] and vehical_detection[0][0] < xyxy[2] < vehical_detection[0][2] and vehical_detection[0][1] < xyxy[1] < vehical_detection[0][3] and vehical_detection[0][1] < xyxy[3] < vehical_detection[0][3]:
                            if plate_number and len(plate_number) > 8:
                                detected_license_plates[vehical_detection[4]] = plate_number

                annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=license_plate_detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=license_plate_detections, labels=license_plate_labels)
                

            cv2.imshow('Resized_Window', annotated_image)
            out.write(annotated_image)
        
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    out.release()

    for x, y in detected_cars.items():
        detection_data = (video_path, y['license_plate'], y['time'], output_name)
        insert_detection(conn, detection_data)


if __name__ == "__main__":
    video_path = "videos/throw-waste.mp4"
    start_detection(video_path)
    conn.close()
    print("Detection completed and data inserted into the database.")
    print("Detected Cars: ", detected_cars)
    print("Detected License Plates: ", detected_license_plates)
    print("Output video saved as: ", 'output/output_{}.mp4'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    print("Database file: detections.db")
    print("Detection data inserted into the database.")