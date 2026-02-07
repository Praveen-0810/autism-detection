import cv2
import numpy as np
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("video_autism_model.h5")
print("Model input shape:", model.input_shape)  # e.g., (None, 20, 112, 112, 3)

# ---------------- ADAPTIVE PREPROCESS FUNCTION ----------------
def preprocess_video_auto(video_path, model):
    """
    Preprocesses a video to match the input shape of the model automatically.

    Args:
        video_path (str): Path to the uploaded video.
        model (tf.keras.Model): Loaded TensorFlow Keras model.

    Returns:
        np.array: Preprocessed video ready for model.predict()
    """
    input_shape = model.input_shape  # e.g., (None, frames, height, width, channels)
    num_frames = input_shape[1]
    img_height = input_shape[2]
    img_width = input_shape[3]
    channels = input_shape[4]

    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    # Choose evenly spaced frames
    frame_idxs = np.linspace(0, total_frames - 1, min(num_frames, total_frames)).astype(int)

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Resize and normalize
        frame = cv2.resize(frame, (img_width, img_height))
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis=-1)
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if not frames:
        return None

    # Pad if fewer frames than expected
    while len(frames) < num_frames:
        frames.append(frames[-1])

    video_array = np.array([frames], dtype=np.float32)  # batch size 1
    return video_array

# ---------------- TEST ----------------
# Example usage:
# video_data = preprocess_video_auto("sample_video.mp4", model)
# pred = model.predict(video_data)
