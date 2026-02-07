import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only

from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from eye_contact import eye_contact_score

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("video_autism_model.h5")

# ---------------- PLACEHOLDER BEHAVIOR FUNCTIONS ----------------
def head_movement_score(video_path):
    return 65.0  # placeholder

def emotion_stability_score(video_path):
    return 70.0  # placeholder

# ---------------- VIDEO PREPROCESS ----------------
def preprocess_video(video_path, num_frames=10, img_size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, img_size)
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array([frames])

# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = False
    label = None
    confidence = 0
    video_path = None
    eye_score = None
    behavior = None
    risk_score = None
    risk_level = None

    if request.method == "POST":
        if "video" not in request.files:
            return render_template("index.html")

        video = request.files["video"]
        if video.filename == "":
            return render_template("index.html")

        # Check file size
        video.seek(0, os.SEEK_END)
        if video.tell() > MAX_VIDEO_SIZE:
            return "Video too large. Max 50MB allowed.", 400
        video.seek(0)

        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        video.save(video_path)

        data = preprocess_video(video_path)
        if data is None:
            return render_template("index.html")

        pred = model.predict(data)[0][0]
        label = "Autism" if pred > 0.5 else "Non-Autism"
        confidence = round(pred if pred > 0.5 else 1 - pred, 2)
        prediction = True

        # -------- BEHAVIOR ANALYSIS --------
        eye = eye_contact_score(video_path)
        head = head_movement_score(video_path)
        emotion = emotion_stability_score(video_path)

        behavior = {
            "eye": eye,
            "head": head,
            "emotion": emotion,
            "attention": "High" if eye >= 70 else "Medium" if eye >= 40 else "Low"
        }

        # -------- RISK ASSESSMENT --------
        risk_score = round(
            (0.4 * confidence * 100) +
            (0.3 * (100 - eye)) +
            (0.3 * (100 - head)),
            2
        )

        if risk_score >= 70:
            risk_level = "High"
        elif risk_score >= 40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        eye_score = eye

    return render_template(
        "index.html",
        prediction=prediction,
        label=label,
        confidence=confidence,
        video_path=video_path,
        eye_score=eye_score,
        behavior=behavior,
        risk_score=risk_score,
        risk_level=risk_level
    )

# ---------------- RENDER ENTRY POINT ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
