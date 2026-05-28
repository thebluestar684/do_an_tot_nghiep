import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_file, redirect, url_for
from ultralytics import YOLO
from sort.sort import Sort
import matplotlib.pyplot as plt

# Disable interactive plotting
plt.switch_backend("Agg")

app = Flask(__name__)

# Initialize YOLOv8 model and SORT tracker
model = YOLO("train.pt")
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Global variables
uploaded_video_path = "uploaded_videos/new.mp4"
unique_vehicle_ids = set()
total_vehicle_count = 0
vehicle_counts = []
speeds = []
predicted_traffic = []
congestion_frames = 0
non_congestion_frames = 0

# Ensure directories exist
os.makedirs("uploaded_videos", exist_ok=True)
os.makedirs("results", exist_ok=True)


@app.route("/")
def login():
    """
    A simple login page for show-off purposes.
    """
    return render_template("login.html")



@app.route("/authenticate", methods=["POST"])
def authenticate():
    """
    Dummy authentication for the login page.
    """
    username = request.form.get("username")
    password = request.form.get("password")

    if username == "user" and password == "123":
        return redirect(url_for("index"))
    else:
        return render_template("login.html", error="Invalid username or password")


@app.route("/index")
def index():
    """
    Main page of the app.
    """
    return render_template("index.html")


def clear_results_directory():
    """
    Deletes all files in the 'results' directory before starting a new analysis.
    """
    for file in os.listdir("results"):
        file_path = os.path.join("results", file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def process_video(video_path):
    """
    Processes the video, detects vehicles, calculates speed, and updates graphs.
    """
    global unique_vehicle_ids, total_vehicle_count, vehicle_counts, speeds, predicted_traffic, congestion_frames, non_congestion_frames
    unique_vehicle_ids.clear()
    total_vehicle_count = 0
    vehicle_counts = []
    speeds = []
    predicted_traffic = []
    congestion_frames = 0
    non_congestion_frames = 0

    # Clear previous results
    clear_results_directory()

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 480p
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_width = int(480 * aspect_ratio)
        frame = cv2.resize(frame, (new_width, 480))

        # Perform detection
        results = model.predict(source=frame, save=False, conf=0.5)

        # Collect detections
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                if confidence > 0.5:
                    detections.append([x1, y1, x2, y2, confidence])

        # Update tracker
        tracked_objects = tracker.update(np.array(detections))

        # Draw bounding boxes and calculate speed
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            unique_vehicle_ids.add(track_id)

            # Example: Calculate speed based on bounding box dimensions (pseudo-speed calculation)
            speed = abs(x2 - x1) / 10  # Replace with a real-world calibration
            speeds.append(speed)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Update total vehicle count and traffic prediction
        total_vehicle_count = len(unique_vehicle_ids)
        vehicle_counts.append(total_vehicle_count)
        predicted_traffic.append(total_vehicle_count * 1440)

        # Update congestion status
        if total_vehicle_count > 30:  # Threshold for congestion
            congestion_frames += 1
        else:
            non_congestion_frames += 1

        # Save graphs every 10 frames
        if frame_index % 10 == 0:
            generate_prediction_graph()
            generate_speed_graph()
            generate_congestion_graph()

        # Yield the frame for video streaming
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        frame_index += 1

    cap.release()


def generate_prediction_graph():
    """
    Generates a live traffic prediction graph.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_traffic, label="24-Hour Traffic Prediction", color="red")
    plt.title("Traffic Prediction Graph")
    plt.xlabel("Frame Index")
    plt.ylabel("Predicted Vehicles in 24 Hours")
    plt.legend()
    plt.grid()
    plt.savefig("results/prediction_graph.png")
    plt.close()


def generate_speed_graph():
    """
    Generates a graph showing the average speed of vehicles over time.
    """
    plt.figure(figsize=(10, 5))
    if speeds:
        plt.plot(range(len(speeds)), speeds, label="Average Speed (km/h)", color="blue")
    plt.title("Average Vehicle Speed Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Speed (Pseudo Units)")
    plt.legend()
    plt.grid()
    plt.savefig("results/speed_graph.png")
    plt.close()


def generate_congestion_graph():
    """
    Generates a pie chart showing congestion vs. clear traffic.
    """
    plt.figure(figsize=(8, 8))
    labels = ["Congested", "Clear"]
    values = [congestion_frames, non_congestion_frames]
    colors = ["red", "green"]

    plt.pie(values, labels=labels, colors=colors, autopct="%1.1f%%")
    plt.title("Congestion Analysis")
    plt.savefig("results/congestion_graph.png")
    plt.close()


@app.route("/video_feed")
def video_feed():
    return Response(process_video(uploaded_video_path), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/upload", methods=["POST"])
def upload_video():
    global uploaded_video_path
    file = request.files.get("video")
    if file:
        uploaded_video_path = f"uploaded_videos/{file.filename}"
        file.save(uploaded_video_path)
        return jsonify({"success": True, "message": "Video uploaded successfully!"})
    return jsonify({"success": False, "message": "No video file provided."})


@app.route("/speed_graph")
def speed_graph():
    return send_file("results/speed_graph.png", mimetype="image/png")


@app.route("/congestion_graph")
def congestion_graph():
    return send_file("results/congestion_graph.png", mimetype="image/png")


@app.route("/prediction_graph")
def prediction_graph():
    return send_file("results/prediction_graph.png", mimetype="image/png")


@app.route("/total_vehicle_count")
def total_vehicle_count():
    global total_vehicle_count
    return jsonify({"count": total_vehicle_count})


@app.route("/congestion_status")
def congestion_status():
    """
    Returns the current congestion status as a string.
    """
    global total_vehicle_count
    if total_vehicle_count > 30:  # Threshold for congestion
        return jsonify({"status": "Congested"})
    elif total_vehicle_count > 15:
        return jsonify({"status": "Moderate Traffic"})
    else:
        return jsonify({"status": "Clear Traffic"})

@app.route("/logout")
def logout():
    """
    Logs out the user, clears old results, and redirects to the login page.
    """
    # Clear old results
    clear_results_directory()

    # Redirect to login page
    return redirect(url_for("login"))





if __name__ == "__main__":
    app.run(debug=True, threaded=True)
