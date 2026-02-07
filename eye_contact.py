import cv2

# Load classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def eye_contact_score(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    eye_contact_frames = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # skip frames

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) == 0:
            continue

        total_frames += 1

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)

            if len(eyes) >= 2:
                eye_contact_frames += 1
            break  # only first face

    cap.release()

    if total_frames == 0:
        return 0.0

    score = (eye_contact_frames / total_frames) * 100
    return round(score, 2)
