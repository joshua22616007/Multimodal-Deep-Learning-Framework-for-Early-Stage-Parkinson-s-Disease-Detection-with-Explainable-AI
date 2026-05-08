import cv2
import numpy as np

# Load DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "models/face_dnn/deploy.prototxt",
    "models/face_dnn/res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_face(frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x1,y1,x2,y2) = box.astype("int")
            faces.append((x1,y1,x2-x1,y2-y1))

    return faces


def predict_face():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ Camera not working!")
        return 0.5

    prev_face = None
    movements = []

    print("📷 Face detection (DNN) running")

    for _ in range(150):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_face(frame)

        print("Faces detected:", len(faces))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            face_roi = cv2.resize(face_roi, (100,100))

            if prev_face is not None:
                diff = cv2.absdiff(prev_face, face_roi)
                movements.append(np.mean(diff))

            prev_face = face_roi

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.putText(frame, "Move face (smile/blink)", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Detection (DNN)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(movements) == 0:
        return 0.5

    movement_score = np.mean(movements)

    print("Movement:", movement_score)

# 🔥 Normalize properly (adaptive scaling)
    norm = movement_score / (movement_score + 1)

    score = 1 - norm  # less movement → higher risk

# 🔥 Clamp safety
    score = max(0, min(score, 1))

    return float(score)