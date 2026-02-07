import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only

from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

# Custom behavior module
from eye_contact import eye_contact_score

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("video_autism_model.h5")
print("Model input shape:", model.input_shape)  # For debugging

NUM_FRAMES = model.input_shape[1]  # e.g., 20
IMG_SIZE = (model.input_shape[2], model.input_shape[3])  # e.g., (64,64)

# ---------------- PLACEHOLDER BEHAVIOR FUNCTIONS ----------------
def head_movement_score(video_path):
    return 65.0

def emotion_stability_score(video_path):
    return 70.0

# ---------------- VIDEO PREPROCESS ----------------
def preprocess_video(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    frame_idxs = np.linspace(0, total_frames - 1, min(num_frames, total_frames)).astype(int)

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, img_size)
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    if not frames:
        return None

    # Pad frames if fewer than num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array([frames], dtype=np.float32)  # Shape: (1, num_frames, H, W, 3)

# ---------------- BATCH PREDICTION FUNCTION ----------------
def batch_predict(dataset_path="newdataset"):
    results = []
    for label in ["autism", "non_autism"]:
        folder_path = os.path.join(dataset_path, label)
        if not os.path.exists(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".mp4"):
                video_path = os.path.join(folder_path, file)
                data = preprocess_video(video_path)
                if data is not None:
                    pred = model.predict(data, verbose=0)[0][0]
                    predicted_label = "Autism" if pred > 0.5 else "Non-Autism"
                    results.append({
                        "file": file,
                        "actual": label,
                        "predicted": predicted_label,
                        "confidence": float(pred)
                    })
    return results

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
            return render_template("index.html", error="No video uploaded.")
        video = request.files["video"]
        if video.filename == "":
            return render_template("index.html", error="No video selected.")

        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        video.save(video_path)

        data = preprocess_video(video_path)
        if data is None:
            return render_template("index.html", error="Could not read video.")

        # -------- PREDICTION --------
        pred = model.predict(data, verbose=0)[0][0]
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

# ---------------- BATCH PREDICTION ROUTE ----------------
@app.route("/batch", methods=["GET"])
def batch():
    results = batch_predict()
    return {"results": results}

# ---------------- RENDER ENTRY POINT ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
