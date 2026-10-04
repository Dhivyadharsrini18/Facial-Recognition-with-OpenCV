"""
utils.py

What this file does:
    Central configuration file for the FaceVault project.
    All paths, thresholds, and constants are defined here so that no
    other module needs to hard-code them.

Input:  None (imported by every other module).
Output: Path objects and configuration constants.
How it fits into the project:
    Every module imports its paths and settings from here, keeping the
    codebase consistent and easy to modify.
"""

from pathlib import Path
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project root  (the directory that contains main.py)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_DIR       = BASE_DIR / "dataset"
RAW_DATA_DIR      = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
TEST_DATA_DIR     = DATASET_DIR / "test"

# ---------------------------------------------------------------------------
# Model / encoding paths
# ---------------------------------------------------------------------------
MODELS_DIR    = BASE_DIR / "models"
ENCODING_FILE = MODELS_DIR / "face_encodings.pkl"

# ---------------------------------------------------------------------------
# Results paths
# ---------------------------------------------------------------------------
RESULTS_DIR              = BASE_DIR / "results"
DETECTED_FACES_DIR       = RESULTS_DIR / "detected_faces"
RECOGNITION_RESULTS_DIR  = RESULTS_DIR / "recognition_results"
EVALUATION_DIR           = RESULTS_DIR / "evaluation"
PLOTS_DIR                = RESULTS_DIR / "plots"

# ---------------------------------------------------------------------------
# Recognition configuration
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD    = 100.0   # LBPH distance: lower = more similar
FACE_SIZE                = (200, 200)  # standard face crop size
MIN_IMAGES_REQUIRED      = 5
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ---------------------------------------------------------------------------
# Haar cascade path  (ships with OpenCV)
# ---------------------------------------------------------------------------
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def setup_project_structure():
    """Create every required directory if it does not already exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TEST_DATA_DIR,
        MODELS_DIR,
        DETECTED_FACES_DIR,
        RECOGNITION_RESULTS_DIR,
        EVALUATION_DIR,
        PLOTS_DIR,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


def _ensure_cascade_exists():
    """
    Some opencv-contrib-python builds ship without the XML cascade files.
    If the file is missing, download it once from the OpenCV GitHub repo.
    """
    cascade_file = Path(HAAR_CASCADE_PATH)
    if cascade_file.exists():
        return

    print("Haar cascade file not found locally — downloading from OpenCV GitHub ...")
    import urllib.request
    url = (
        "https://raw.githubusercontent.com/opencv/opencv/"
        "master/data/haarcascades/haarcascade_frontalface_default.xml"
    )
    try:
        cascade_file.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(cascade_file))
        print(f"  Downloaded to: {cascade_file}")
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Failed to download Haar cascade: {e}\n"
            "Please download it manually from:\n"
            f"  {url}\n"
            f"and place it at:\n"
            f"  {cascade_file}"
        )


def get_face_detector():
    """
    Return a Haar Cascade face detector.
    This is lightweight, beginner-friendly, and ships with OpenCV.
    Downloads the cascade file automatically if it is missing.
    """
    _ensure_cascade_exists()
    cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(
            "ERROR: Could not load Haar cascade classifier.\n"
            "Make sure opencv-python is installed correctly."
        )
    return cascade


def detect_faces(gray_image, detector=None):
    """
    Detect faces in a grayscale image using Haar cascades.

    Args:
        gray_image: numpy array (grayscale).
        detector:   CascadeClassifier (created if None).

    Returns:
        List of (x, y, w, h) tuples.
    """
    if detector is None:
        detector = get_face_detector()
    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    # detectMultiScale may return a numpy array or a tuple
    if isinstance(faces, np.ndarray):
        return faces.tolist()
    return list(faces)


def get_project_status():
    """
    Gather high-level project status for the menu.
    Returns a dictionary of status items.
    """
    status = {}

    # Registered people (raw)
    raw_people = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()] if RAW_DATA_DIR.exists() else []
    status["registered_people"] = len(raw_people)
    status["raw_images"] = sum(
        1 for p in raw_people for f in p.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    # Processed images
    proc_people = [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()] if PROCESSED_DATA_DIR.exists() else []
    status["processed_images"] = sum(
        1 for p in proc_people for f in p.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    # Encoding database
    status["encoding_exists"] = ENCODING_FILE.exists()

    # Test images
    test_people = [d for d in TEST_DATA_DIR.iterdir() if d.is_dir()] if TEST_DATA_DIR.exists() else []
    status["test_people"] = len(test_people)
    status["test_images"] = sum(
        1 for p in test_people for f in p.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    # Evaluation
    status["evaluation_done"] = (EVALUATION_DIR / "report.txt").exists()

    return status
