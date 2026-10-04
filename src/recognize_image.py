"""
recognize_image.py

What this file does:
    Loads the trained LBPH model, detects faces in a given image,
    predicts identities, and draws annotated results.

Input:  Path to an image file  (via --image argument or from main.py).
Output: Annotated image displayed on-screen and saved to results/recognition_results/.
How it fits into the project:
    Step 4.  Tests the trained model on individual static images before
    moving to real-time webcam recognition.
"""

import cv2
import pickle
import numpy as np
from pathlib import Path
import argparse

from src.utils import (
    ENCODING_FILE, MODELS_DIR, RECOGNITION_RESULTS_DIR,
    RECOGNITION_THRESHOLD, FACE_SIZE,
    get_face_detector,
)


# ---------------------------------------------------------------------------
# Helper functions  (also imported by recognize_webcam.py and evaluate.py)
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the LBPH recogniser and the label-to-name mapping.

    Returns:
        (recogniser, label_map)  or  (None, None) on failure.
    """
    model_path = MODELS_DIR / "lbph_model.yml"

    if not model_path.exists():
        print(f"ERROR: LBPH model not found at {model_path}")
        print("Please run 'Generate Face Encodings' first (Menu option 3).")
        return None, None

    if not ENCODING_FILE.exists():
        print(f"ERROR: Metadata file not found at {ENCODING_FILE}")
        return None, None

    recogniser = cv2.face.LBPHFaceRecognizer_create()
    recogniser.read(str(model_path))

    with open(ENCODING_FILE, "rb") as f:
        data = pickle.load(f)

    return recogniser, data["label_map"]


def distance_to_confidence(distance, threshold=RECOGNITION_THRESHOLD):
    """
    Convert an LBPH distance into a 0–100 % confidence score.

    LBPH distance:  0 = perfect match,  higher = less similar.
    We map [0 … 2*threshold] → [100% … 0%].
    """
    max_distance = threshold * 2.0
    if distance >= max_distance:
        return 0.0
    confidence = (1.0 - distance / max_distance) * 100.0
    return round(confidence, 1)


def recognise_faces_in_image(img, recogniser, label_map, detector):
    """
    Detect and recognise every face in a BGR image.

    Returns:
        results – list of dicts:  {x, y, w, h, name, confidence, distance}
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raw_faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    results = []
    for (x, y, w, h) in raw_faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, FACE_SIZE)

        label, distance = recogniser.predict(face_resized)
        confidence = distance_to_confidence(distance, RECOGNITION_THRESHOLD)

        if distance <= RECOGNITION_THRESHOLD:
            name = label_map.get(label, "Unknown")
        else:
            name = "Unknown"

        results.append({
            "x": x, "y": y, "w": w, "h": h,
            "name": name,
            "confidence": confidence,
            "distance": round(distance, 2),
        })

    return results


def draw_results(img, results):
    """Draw bounding boxes, names, and confidence on the image."""
    for r in results:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        name = r["name"]
        conf = r["confidence"]

        colour = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)

        # Label background
        label = f"{name} ({conf}%)"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(
            img,
            (x, y + h),
            (x + label_size[0] + 10, y + h + label_size[1] + baseline + 10),
            colour, cv2.FILLED,
        )
        cv2.putText(
            img, label,
            (x + 5, y + h + label_size[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    return img


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(image_path=None):
    """Recognise faces in a single static image."""
    if image_path is None:
        parser = argparse.ArgumentParser(description="Recognise faces in an image.")
        parser.add_argument("--image", required=True, help="Path to the input image")
        args = parser.parse_args()
        image_path = args.image

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found at '{image_path}'")
        return

    recogniser, label_map = load_model()
    if recogniser is None:
        return

    try:
        detector = get_face_detector()
    except RuntimeError as e:
        print(e)
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Unable to read image '{image_path}'")
        return

    results = recognise_faces_in_image(img, recogniser, label_map, detector)

    if not results:
        print("WARNING: No faces detected in the image.")
    else:
        print(f"Detected {len(results)} face(s):")
        for r in results:
            print(f"  → {r['name']}  |  Confidence: {r['confidence']}%  |  Distance: {r['distance']}")

    annotated = draw_results(img.copy(), results)

    # Save result
    RECOGNITION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RECOGNITION_RESULTS_DIR / f"result_{image_path.name}"
    cv2.imwrite(str(save_path), annotated)
    print(f"Result saved to: {save_path}")

    # Display
    h, w = annotated.shape[:2]
    if max(w, h) > 900:
        scale = 900 / max(w, h)
        annotated = cv2.resize(annotated, (0, 0), fx=scale, fy=scale)

    cv2.imshow("FaceVault - Recognition Result  |  Press any key to close", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
