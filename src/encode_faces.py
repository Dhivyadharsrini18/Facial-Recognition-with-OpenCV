"""
encode_faces.py

What this file does:
    Trains an LBPH (Local Binary Pattern Histogram) face recogniser on the
    preprocessed dataset and saves the trained model to disk.

    LBPH works by describing each face as a histogram of local texture
    patterns.  It is fast, works well with small datasets, and does NOT
    require dlib or a GPU.

Input:  Grayscale face images in dataset/processed/<person>/.
Output: models/face_encodings.pkl  (pickled dict with the LBPH model,
        label-to-name mapping, and image counts).
How it fits into the project:
    Step 3.  Converts raw pixel data into a compact representation
    (histogram) that can be compared quickly during recognition.
"""

import cv2
import pickle
import numpy as np
from src.utils import (
    PROCESSED_DATA_DIR, ENCODING_FILE, MODELS_DIR,
    SUPPORTED_IMAGE_EXTENSIONS, FACE_SIZE,
)


def main():
    """Train an LBPH model on processed face images and save to disk."""
    print("\n=== Generating Face Encodings (LBPH) ===")

    if not PROCESSED_DATA_DIR.exists() or not any(PROCESSED_DATA_DIR.iterdir()):
        print(f"ERROR: Processed dataset directory '{PROCESSED_DATA_DIR}' is empty.")
        print("Please run Preprocessing first (Menu option 2).")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    faces = []          # list of grayscale numpy arrays
    labels = []         # integer labels (one per image)
    label_map = {}      # {int_label: person_name}

    current_label = 0
    total_people = 0
    total_encoded = 0
    skipped = 0

    for person_dir in sorted(PROCESSED_DATA_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        label_map[current_label] = person_name
        total_people += 1

        print(f"  Encoding: {person_name}  (label {current_label})")

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"    [WARNING] Unreadable: {img_path.name}")
                skipped += 1
                continue

            # Ensure uniform size
            if img.shape[:2] != FACE_SIZE:
                img = cv2.resize(img, FACE_SIZE)

            faces.append(img)
            labels.append(current_label)
            total_encoded += 1

        current_label += 1

    if total_encoded == 0:
        print("ERROR: No faces could be encoded.  Model not saved.")
        return

    # Train the LBPH recogniser
    print("\nTraining LBPH recogniser ...")
    recogniser = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recogniser.train(faces, np.array(labels))

    # Save everything we need for recognition
    data = {
        "label_map": label_map,              # {0: "Alice", 1: "Bob", ...}
        "total_people": total_people,
        "total_encoded": total_encoded,
    }

    # Save the LBPH model to a YAML file, and save the mapping via pickle
    model_yml = MODELS_DIR / "lbph_model.yml"
    recogniser.write(str(model_yml))

    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)

    print("\n--- Encoding Summary ---")
    print(f"  People enrolled       : {total_people}")
    print(f"  Images encoded        : {total_encoded}")
    print(f"  Skipped / unreadable  : {skipped}")
    print(f"  LBPH model saved to   : {model_yml}")
    print(f"  Metadata saved to     : {ENCODING_FILE}")


if __name__ == "__main__":
    main()
