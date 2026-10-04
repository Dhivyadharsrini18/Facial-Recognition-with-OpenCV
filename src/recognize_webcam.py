"""
recognize_webcam.py

What this file does:
    Opens the webcam and performs real-time facial recognition using the
    trained LBPH model.

Input:  Live webcam feed + trained model from models/.
Output: Live video window showing bounding boxes, names, confidence, and FPS.
How it fits into the project:
    Step 5.  The real-time demonstration of FaceVault, suitable for a
    college project live demo.
"""

import cv2
import time

from src.utils import RECOGNITION_THRESHOLD, FACE_SIZE, get_face_detector
from src.recognize_image import load_model, distance_to_confidence


def main():
    """Real-time webcam recognition loop."""
    print("\n=== Real-Time Webcam Recognition ===")

    recogniser, label_map = load_model()
    if recogniser is None:
        return

    try:
        detector = get_face_detector()
    except RuntimeError as e:
        print(e)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("\nInstructions:")
    print("  Q – quit\n")

    # Initialise timing for FPS
    prev_time = time.time()

    # We'll store the latest detection results so we can skip frames
    # for performance while still drawing boxes every frame.
    boxes = []
    names = []
    confidences = []
    process_this_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        if process_this_frame:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            boxes = []
            names = []
            confidences = []

            for (x, y, w, h) in raw_faces:
                face_roi = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, FACE_SIZE)

                label, distance = recogniser.predict(face_resized)
                conf = distance_to_confidence(distance, RECOGNITION_THRESHOLD)

                if distance <= RECOGNITION_THRESHOLD:
                    name = label_map.get(label, "Unknown")
                else:
                    name = "Unknown"

                boxes.append((x, y, w, h))
                names.append(name)
                confidences.append(conf)

        # Toggle to skip every other frame for speed
        process_this_frame = not process_this_frame

        # Draw results
        for (x, y, w, h), name, conf in zip(boxes, names, confidences):
            colour = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

            label_text = f"{name} ({conf}%)"
            cv2.rectangle(frame, (x, y + h), (x + len(label_text) * 12, y + h + 28), colour, cv2.FILLED)
            cv2.putText(
                frame, label_text, (x + 5, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        # FPS
        curr_time = time.time()
        elapsed = curr_time - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        prev_time = curr_time

        cv2.putText(
            frame, f"FPS: {int(fps)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
        )

        cv2.imshow("FaceVault - Real-Time Recognition  |  Q=quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()
