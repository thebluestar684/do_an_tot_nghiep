import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

VIDEO_PATH = "new1.mp4"
MODEL_PATH = "yolov8n.pt"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]

model = YOLO(MODEL_PATH)
tracker = Sort(max_age=70, min_hits=1, iou_threshold=0.15)

cap = cv2.VideoCapture(VIDEO_PATH)

total_vehicles = 0
tracked_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))

    results = model.predict(source=frame, save=False, conf=0.25, verbose=False)
    
    detections = np.empty((0, 5))
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if cls in VEHICLE_CLASSES:
                current_array = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack((detections, current_array))

    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        
        if obj_id not in tracked_ids:
            tracked_ids.add(obj_id)
            total_vehicles += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Tong so xe: {total_vehicles}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("UrbanFlow - Local Execution", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()