import cv2
import numpy as np

def predict_gait():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ Camera not working!")
        return 0.5

    prev_frame = None
    motion_values = []

    print("🚶 Walk in front of camera")

    for _ in range(150):  # ~6 seconds
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            motion_values.append(np.mean(diff))

        prev_frame = gray

        cv2.putText(frame, "Walk naturally", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Gait Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(motion_values) == 0:
        return 0.5

    gait_score = np.std(motion_values) / 255

    # Normalize
    score = 1 - min(gait_score * 10, 1)

    return float(score)