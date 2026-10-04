"""
preprocess.py

What this file does:
    Reads every raw image from dataset/raw/<person>/,
    detects the face, crops it, converts it to grayscale,
    resizes it to a standard size, and saves it into
    dataset/processed/<person>/.

Input:  Raw webcam images in dataset/raw/.
Output: Cropped, resized grayscale face images in dataset/processed/.
How it fits into the project:
    Step 2.  Standardises all face images so the encoding stage
    receives clean, consistent input.
"""

import cv2
import shutil
from src.utils import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SUPPORTED_IMAGE_EXTENSIONS,
    FACE_SIZE, get_face_detector,
)


def main():
    """Preprocess all raw images: detect → crop → resize → save."""
    print("\n=== Dataset Preprocessing ===")

    if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.iterdir()):
        print(f"ERROR: Raw dataset directory '{RAW_DATA_DIR}' is empty or does not exist.")
        print("Please collect images first (Menu option 1).")
        return

    # Load face detector
    try:
        face_cascade = get_face_detector()
    except RuntimeError as e:
        print(e)
        return

    # Stats
    total_images = 0
    successfully_processed = 0
    failed_images = 0
    images_without_faces = 0
    images_multiple_faces = 0

    # Re-create the processed directory (fresh start each time)
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for person_dir in sorted(RAW_DATA_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        out_dir = PROCESSED_DATA_DIR / person_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {person_name}")

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            total_images += 1

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [ERROR]   Corrupted / unreadable: {img_path.name}")
                    failed_images += 1
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )

                if len(faces) == 0:
                    print(f"  [WARNING] No face detected: {img_path.name}")
                    images_without_faces += 1
                    continue

                if len(faces) > 1:
                    print(f"  [INFO]    Multiple faces in {img_path.name} — using largest.")
                    images_multiple_faces += 1
                    # Pick the largest face by area
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

                x, y, w, h = faces[0]
                face_crop = gray[y:y + h, x:x + w]            # grayscale crop
                face_resized = cv2.resize(face_crop, FACE_SIZE) # standard size

                save_path = out_dir / img_path.name
                cv2.imwrite(str(save_path), face_resized)
                successfully_processed += 1

            except Exception as e:
                print(f"  [ERROR]   Failed on {img_path.name}: {e}")
                failed_images += 1

    print("\n--- Preprocessing Summary ---")
    print(f"  Total images scanned   : {total_images}")
    print(f"  Successfully processed : {successfully_processed}")
    print(f"  Failed / corrupt       : {failed_images}")
    print(f"  No face detected       : {images_without_faces}")
    print(f"  Multiple faces (used 1): {images_multiple_faces}")


if __name__ == "__main__":
    main()
