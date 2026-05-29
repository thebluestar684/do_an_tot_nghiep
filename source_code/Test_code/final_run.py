import cv2
import numpy as np
from ultralytics import RTDETR
from collections import defaultdict
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time

detector = RTDETR("rtdetr-l.pt")
gru_model = load_model("gru_traffic_model.h5", compile=False)

video_path = "new1.mp4"
cap = cv2.VideoCapture(video_path)

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

track_history = defaultdict(list)

roi_points = np.array([
    [180, 180],
    [1150, 180],
    [1250, 720],
    [80, 720]
], dtype=np.int32)

TIMESTEPS = 12
traffic_sequence = []

scaler = MinMaxScaler()
dummy_data = np.array([
    [0, 0, 0],
    [826, 0.5264, 77.3]
])
scaler.fit(dummy_data)

def detect_complex_direction(track):
    p_start = track[0]
    p_end = track[-1]
    
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    
    MIN_MOTION_THRESHOLD = 30 
    if (dx**2 + dy**2) < MIN_MOTION_THRESHOLD**2:
        return "Queueing/Stopped"

    angle = np.degrees(np.arctan2(dy, dx))
    
    if -22.5 <= angle < 22.5:
        return "West -> East"
    elif 22.5 <= angle < 67.5:
        return "Turn Right: W->S"
    elif 67.5 <= angle < 112.5:
        return "North -> South"
    elif 112.5 <= angle < 157.5:
        return "Turn Left: E->S"
    elif angle >= 157.5 or angle < -157.5:
        return "East -> West"
    elif -157.5 <= angle < -112.5:
        return "Turn Right: E->N"
    elif -112.5 <= angle < -67.5:
        return "South -> North"
    elif -67.5 <= angle < -22.5:
        return "Turn Left: W->N"

    return "Proceeding"

def calculate_occupancy(vehicle_count):
    return min(vehicle_count / 50.0, 1.0)

def calculate_speed(vehicle_count):
    base_speed = 45
    reduced_speed = base_speed - (vehicle_count * 0.4)
    return max(reduced_speed, 10)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (1280, 720))

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    results = detector.track(roi_frame, persist=True, conf=0.4)

    direction_count = {
        "North -> South": 0, "South -> North": 0,
        "West -> East": 0, "East -> West": 0,
        "Turn Left: W->N": 0, "Turn Right: W->S": 0,
        "Turn Left: E->S": 0, "Turn Right: E->N": 0
    }

    total_vehicle = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):
            cls = int(cls)
            if cls not in vehicle_classes:
                continue

            total_vehicle += 1
            x1, y1, x2, y2 = map(int, box)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            track = track_history[int(track_id)]
            track.append((center_x, center_y))

            if len(track) > 30: 
                track.pop(0)

            direction = "Detecting..."
            if len(track) >= 5:
                direction = detect_complex_direction(track)
                if direction in direction_count:
                    direction_count[direction] += 1

            box_color = (0, 165, 255) if "Turn" in direction else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

            label = f"{vehicle_classes[cls]} | {direction}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            for i in range(1, len(track)):
                cv2.line(frame, track[i - 1], track[i], (255, 0, 0), 2)

    flow = total_vehicle
    occupy = calculate_occupancy(total_vehicle)
    speed = calculate_speed(total_vehicle)

    traffic_sequence.append([flow, occupy, speed])

    if len(traffic_sequence) > TIMESTEPS:
        traffic_sequence.pop(0)

    predicted_flow = flow
    density_ratio = 0.0
    green_light_time = 20

    if len(traffic_sequence) == TIMESTEPS:
        input_data = scaler.transform(np.array(traffic_sequence))
        X_input = np.array([input_data])

        predicted_scaled = gru_model.predict(X_input, verbose=0)[0][0]
        flow_min, flow_max = 0, 826
        predicted_flow = (predicted_scaled * (flow_max - flow_min)) + flow_min

        density_ratio = max(0, min(predicted_flow / flow_max, 1))
        GREEN_MIN, GREEN_MAX = 15, 80
        green_light_time = int(GREEN_MIN + (GREEN_MAX - GREEN_MIN) * density_ratio)

    total_ns = (direction_count["North -> South"] + direction_count["South -> North"] + 
                direction_count["Turn Right: E->N"] + direction_count["Turn Left: W->N"])
                
    total_ew = (direction_count["West -> East"] + direction_count["East -> West"] + 
                direction_count["Turn Right: W->S"] + direction_count["Turn Left: E->S"])

    priority = "North-South" if total_ns > total_ew else ("East-West" if total_ew > total_ns else "Balanced")

    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_points], (0, 255, 255))
    frame = cv2.addWeighted(overlay, 0.10, frame, 0.90, 0)

    metrics = [
        (f"Vehicles in ROI: {flow}", (0, 255, 0)),
        (f"Predicted Flow: {predicted_flow:.2f}", (255, 255, 0)),
        (f"Traffic Density: {density_ratio:.2f}", (0, 255, 255)),
        (f"Priority Phase: {priority}", (0, 0, 255)),
        (f"Green Light Alloc: {green_light_time}s", (255, 255, 255))
    ]
    for idx, (text, color) in enumerate(metrics):
        cv2.putText(frame, text, (30, 40 + idx*35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, "[ STRAIGHT PHASE ]", (1000, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    cv2.putText(frame, f"N->S: {direction_count['North -> South']} | S->N: {direction_count['South -> North']}", (1000, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"W->E: {direction_count['West -> East']} | E->W: {direction_count['East -> West']}", (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "[ TURNING PHASE ]", (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(frame, f"Left W->N: {direction_count['Turn Left: W->N']}", (1000, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Right W->S: {direction_count['Turn Right: W->S']}", (1000, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Left E->S: {direction_count['Turn Left: E->S']}", (1000, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Right E->N: {direction_count['Turn Right: E->N']}", (1000, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Smart Traffic Control System (8-Way Adaptive)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()