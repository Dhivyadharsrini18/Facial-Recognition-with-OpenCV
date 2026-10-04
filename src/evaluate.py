"""
evaluate.py

What this file does:
    Runs the trained model against a separate test dataset and calculates
    real performance metrics.

Input:  Test images in dataset/test/<person_name>/  +  trained LBPH model.
Output: results/evaluation/report.txt
        results/evaluation/classification_report.txt
        results/evaluation/confusion_matrix.png
        results/plots/accuracy_chart.png
How it fits into the project:
    Step 6.  Provides actual, measured evidence of how well the model
    performs — required for any college AI/ML submission.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

from src.utils import (
    TEST_DATA_DIR, EVALUATION_DIR, PLOTS_DIR,
    RECOGNITION_THRESHOLD, FACE_SIZE, SUPPORTED_IMAGE_EXTENSIONS,
    get_face_detector, RECOGNITION_RESULTS_DIR,
)
from src.recognize_image import load_model, distance_to_confidence, draw_results


def main():
    """Evaluate the LBPH model on the test dataset."""
    print("\n=== Model Evaluation ===")

    # ------------------------------------------------------------------
    # Guard: test dataset must exist
    # ------------------------------------------------------------------
    if not TEST_DATA_DIR.exists():
        print(f"ERROR: Test dataset directory '{TEST_DATA_DIR}' does not exist.")
        print("Create it and add test images before evaluating.")
        return

    test_dirs = [d for d in TEST_DATA_DIR.iterdir() if d.is_dir()]
    if not test_dirs:
        print(f"ERROR: '{TEST_DATA_DIR}' is empty.  Add test person folders first.")
        return

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    recogniser, label_map = load_model()
    if recogniser is None:
        return

    try:
        detector = get_face_detector()
    except RuntimeError as e:
        print(e)
        return

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RECOGNITION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run predictions
    # ------------------------------------------------------------------
    y_true = []
    y_pred = []
    total_images = 0
    no_face_count = 0

    print("Running evaluation (this may take a moment) ...\n")

    for person_dir in sorted(test_dirs):
        true_name = person_dir.name
        print(f"  Testing: {true_name}")

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            total_images += 1

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    [WARNING] Cannot read {img_path.name}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            predicted_name = "Unknown"

            if len(faces) > 0:
                # Use the largest detected face
                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y_coord, w, h = faces_sorted[0]
                face_roi = gray[y_coord:y_coord + h, x:x + w]
                face_resized = cv2.resize(face_roi, FACE_SIZE)

                label, distance = recogniser.predict(face_resized)
                if distance <= RECOGNITION_THRESHOLD:
                    predicted_name = label_map.get(label, "Unknown")
            else:
                no_face_count += 1

            y_true.append(true_name)
            y_pred.append(predicted_name)

    if total_images == 0:
        print("ERROR: No valid test images found.")
        return

    # ------------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------------
    labels = sorted(set(y_true + y_pred))

    accuracy  = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    correct   = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    incorrect = total_images - correct

    # False Negative: known person labelled as Unknown
    false_negatives = sum(1 for t, p in zip(y_true, y_pred) if p == "Unknown" and t != "Unknown")
    # False Positive: labelled as a specific person, but wrong person  (or Unknown → named)
    false_positives = sum(1 for t, p in zip(y_true, y_pred) if p != "Unknown" and p != t)

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    report_lines = [
        "# Facial Recognition Evaluation",
        "",
        f"Total test images       : {total_images}",
        f"Correct predictions     : {correct}",
        f"Incorrect predictions   : {incorrect}",
        f"Images with no face     : {no_face_count}",
        "",
        f"Accuracy                : {accuracy:.2f}%",
        f"Precision (macro avg)   : {precision:.2f}%",
        f"Recall    (macro avg)   : {recall:.2f}%",
        f"F1 Score  (macro avg)   : {f1:.2f}%",
        "",
        f"False Positives         : {false_positives}",
        f"False Negatives         : {false_negatives}",
        "",
        "--- Definitions ---",
        "False Positive : Model assigned a wrong person name (misidentification).",
        "False Negative : Model said 'Unknown' for a registered person (missed recognition).",
    ]

    report_text = "\n".join(report_lines)
    report_path = EVALUATION_DIR / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nReport saved to: {report_path}")

    # ------------------------------------------------------------------
    # Classification report  (per-class precision / recall)
    # ------------------------------------------------------------------
    cls_report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cls_path = EVALUATION_DIR / "classification_report.txt"
    cls_path.write_text(cls_report, encoding="utf-8")
    print(f"\nPer-class classification report saved to: {cls_path}")

    # ------------------------------------------------------------------
    # Confusion matrix plot
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), max(5, len(labels) * 1.2)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    ax.set_title("FaceVault – Confusion Matrix")
    plt.tight_layout()
    cm_path = EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # ------------------------------------------------------------------
    # Accuracy bar chart
    # ------------------------------------------------------------------
    per_person_acc = {}
    for label in sorted(set(y_true)):
        indices = [i for i, t in enumerate(y_true) if t == label]
        person_correct = sum(1 for i in indices if y_pred[i] == label)
        per_person_acc[label] = (person_correct / len(indices)) * 100 if indices else 0

    fig2, ax2 = plt.subplots(figsize=(max(6, len(per_person_acc) * 1.5), 5))
    bars = ax2.bar(per_person_acc.keys(), per_person_acc.values(), color="#4C72B0")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Person")
    ax2.set_title("FaceVault – Per-Person Recognition Accuracy")
    for bar, val in zip(bars, per_person_acc.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    acc_path = PLOTS_DIR / "accuracy_chart.png"
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"Accuracy chart saved to: {acc_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
