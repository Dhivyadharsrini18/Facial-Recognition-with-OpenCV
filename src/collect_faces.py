"""
collect_faces.py

What this file does:
    Opens the webcam, detects faces in real time, and lets the user
    capture images by pressing SPACE.  Images are saved into
    dataset/raw/<person_name>/.

Input:  Person's name (typed in the console) + webcam video feed.
Output: JPEG images saved to  dataset/raw/<person_name>/.
How it fits into the project:
    This is Step 1.  It gathers the raw training data for each person
    the system should learn to recognise.
"""

import cv2
import re
from src.utils import RAW_DATA_DIR, get_face_detector, SUPPORTED_IMAGE_EXTENSIONS


def _sanitise_name(name: str) -> str:
    """
    Remove characters that are unsafe for folder names.
    Replace spaces with underscores, strip leading/trailing whitespace.
    """
    name = name.strip().replace(" ", "_")
    name = re.sub(r'[^A-Za-z0-9_-]', '', name)
    return name


def _next_image_number(person_dir):
    """
    Find the next available image number so we never overwrite an
    existing capture, even across multiple collection sessions.
    """
    existing = [
        f for f in person_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ] if person_dir.exists() else []
    return len(existing) + 1


def main():
    """Interactive webcam-based dataset collection."""
    print("\n=== Dataset Collection ===")
    person_name = input("Enter the person's name: ").strip()
    person_name = _sanitise_name(person_name)

    if not person_name:
        print("ERROR: Name cannot be empty or contain only special characters.")
        return

    # Create / reuse directory for this person
    person_dir = RAW_DATA_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Load face detector
    try:
        face_cascade = get_face_detector()
    except RuntimeError as e:
        print(e)
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.  Make sure a camera is connected.")
        return

    count = _next_image_number(person_dir)
    saved = 0

    print(f"\nSaving images to: {person_dir}")
    print("Instructions:")
    print("  SPACE  – capture image (face must be visible)")
    print("  Q      – quit collection")
    print("  Aim for at least 20 images with varied angles, lighting, and expressions.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read a frame from the webcam.")
            break

        # Detect faces on a grayscale copy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # On-screen info
        cv2.putText(
            frame, f"Person: {person_name}  |  Saved: {saved}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        if len(faces) == 0:
            cv2.putText(
                frame, "No face detected", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        cv2.imshow("FaceVault - Collect Faces  |  SPACE=save  Q=quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if len(faces) == 0:
                print("WARNING: No face detected — image NOT saved.")
            elif len(faces) > 1:
                print("WARNING: Multiple faces detected — please keep only one person in frame.")
            else:
                img_path = person_dir / f"{person_name}_{count:03d}.jpg"
                cv2.imwrite(str(img_path), frame)
                print(f"  Saved  {img_path.name}")
                count += 1
                saved += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCollection complete.  {saved} new images saved for '{person_name}'.")


if __name__ == "__main__":
    main()
