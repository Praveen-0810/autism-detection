import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_videos_from_folder(folder, label, num_frames=20, img_size=(64,64)):
    data = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            path = os.path.join(folder, file)
            cap = cv2.VideoCapture(path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idxs = np.linspace(0, total_frames-1, num_frames).astype(int)
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, img_size)
                    frame = frame / 255.0
                    frames.append(frame)
            cap.release()
            if len(frames) == num_frames:
                data.append(frames)
                labels.append(label)
    return np.array(data), np.array(labels)

autism_data, autism_labels = load_videos_from_folder("newdataset/autismdataset", 1)
nonautism_data, nonautism_labels = load_videos_from_folder("newdataset/nonautismdataset", 0)
X = np.concatenate([autism_data, nonautism_data], axis=0)
y = np.concatenate([autism_labels, nonautism_labels], axis=0)
print("Dataset shape:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu'), input_shape=(20, 64, 64, 3)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=1
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

pred = model.predict(X_test[:1])
label = "Autism" if pred[0][0] > 0.5 else "Non-Autism"
confidence = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]
print(f"Prediction: {label} (Confidence: {confidence:.2f})")

model.save("video_autism_model.h5")

def preprocess_video(video_path, num_frames=20, img_size=(64,64)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames-1, num_frames).astype(int)
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, img_size)
            frame = frame / 255.0
            frames.append(frame)
    cap.release()
    if len(frames) == num_frames:
        return np.array([frames])
    else:
        return None

def predict_and_show(video_path):
    data = preprocess_video(video_path)
    if data is None:
        print("Not enough frames extracted!")
        return
    pred = model.predict(data)
    label = "Autism" if pred[0][0] > 0.5 else "Non-Autism"
    confidence = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

video_path = r"newdataset/autismdataset/WhatsApp Video 2026-01-10 at 11.18.28 AM.mp4"
predict_and_show(video_path)
