import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if i < len(text.split("\n")) - 1 else line for i, line in enumerate(text.split("\n"))]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(text.split("\n")) - 1 else line for i, line in enumerate(text.split("\n"))]
    })

# 1. Title and Project Overview
add_md("""# FaceVault – AI-Based Facial Recognition System

## 1. Project Overview
Facial recognition is a biometric technology that uses distinctive features of the face to identify or verify a person's identity. 
The objective of this project is to build a complete, end-to-end facial recognition pipeline that works locally without relying on external cloud services.
The system detects faces in images, preprocesses them to extract consistent features, trains a machine-learning model (LBPH - Local Binary Patterns Histograms) to recognize these patterns, and evaluates its performance on test data.

This project is implemented using Python, OpenCV for computer vision, Scikit-learn for metrics, and Matplotlib for visualizations.""")

# 2. Problem Statement
add_md("""## 2. Problem Statement
Manual identification of individuals is slow, error-prone, and difficult to scale. Automated facial recognition systems solve this by providing rapid, contactless authentication, which is critical for applications like attendance tracking, secure access control, and personalized user experiences.""")

# 3. Objectives
add_md("""## 3. Objectives
The core objectives of this project are to:
* **Face Detection**: Accurately locate faces in varied environments.
* **Face Recognition**: Identify registered individuals based on learned facial patterns.
* **Unknown Person Handling**: Correctly classify unregistered people as "Unknown" rather than forcing a false match.
* **Model Evaluation**: Quantitatively measure the system's accuracy, precision, recall, and F1-score on a dedicated test dataset.
* **Performance Analysis**: Visualize misclassifications using a confusion matrix and performance charts.""")

# 4. Technology Stack
add_md("""## 4. Technology Stack
* **Python**: Core programming language.
* **OpenCV (`opencv-contrib-python`)**: For face detection (Haar Cascades) and facial feature extraction/recognition (LBPH).
* **NumPy**: For high-performance array operations and image matrix manipulation.
* **Matplotlib**: For plotting results and confusion matrices.
* **Scikit-learn**: For robust calculation of classification metrics.
* **Pillow (PIL)**: For additional image handling if needed.

*Note: This project relies on OpenCV's built-in LBPH recognizer, which provides robust performance without requiring complex deep-learning setups or C++ compilers (like dlib).*""")

# 5. Import Libraries
add_md("""## 5. Import Libraries
First, we'll import all necessary packages required for the notebook.""")

add_code("""import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import urllib.request
from IPython.display import display, Markdown""")

# 6. Configuration
add_md("""## 6. Configuration
We will define the directory structure for our project. 
The paths are set up using `pathlib` for cross-platform compatibility. The required folders will be created automatically if they don't exist.""")

add_code("""PROJECT_ROOT = Path.cwd()

# Dataset Paths
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Output Paths
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DETECTED_FACES_DIR = RESULTS_DIR / "detected_faces"
RECOGNITION_RESULTS_DIR = RESULTS_DIR / "recognition_results"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"

# Constants
FACE_SIZE = (200, 200)
RECOGNITION_THRESHOLD = 100.0  # Adjust based on LBPH distance (lower distance = better match)

# Create directories
for d in [TRAIN_DIR, TEST_DIR, MODEL_DIR, DETECTED_FACES_DIR, RECOGNITION_RESULTS_DIR, EVALUATION_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    
print("Project Structure Initialized:")
print(f"Training Data Path : {TRAIN_DIR}")
print(f"Testing Data Path  : {TEST_DIR}")
print(f"Models Path        : {MODEL_DIR}")
print(f"Results Path       : {RESULTS_DIR}")""")

# 7. Dataset Instructions
add_md("""## 7. Dataset Instructions
**IMPORTANT: BEFORE PROCEEDING**

Please manually place your photographs in the correct folders. This system does not use a webcam for dataset collection by default, giving you full control over the images used.

**TRAINING PHOTOS GO HERE:**
`dataset/train/<PersonName>/`

**TESTING PHOTOS GO HERE:**
`dataset/test/<PersonName>/`

### Example Structure:
```text
dataset/train/Loga/
    img1.jpg
    img2.jpg
    ...
dataset/train/Person2/
    img1.jpg
    ...

dataset/test/Loga/
    test1.jpg
    test2.jpg
    ...
dataset/test/Person2/
    test1.jpg
    ...
```

### Guidelines for a Good Dataset:
* **Quantity**: 20–30 training images per person is recommended for good results.
* **Separation**: Test images **must** be different from training images to properly evaluate the model.
* **Variation**: Use different expressions, slight angle changes, and varied lighting.
* **Quality**: Avoid extremely blurry, very dark, or heavily occluded images.
* **Privacy**: Only use images for which you have obtained permission.""")

# 8. Dataset Discovery
add_md("""## 8. Dataset Discovery
We will automatically scan the `dataset/train/` and `dataset/test/` directories to discover the registered people and count the available images.""")

add_code("""def scan_dataset(directory):
    if not directory.exists():
        return {}
    
    counts = {}
    for person_dir in sorted(directory.iterdir()):
        if person_dir.is_dir():
            images = [f for f in person_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            counts[person_dir.name] = len(images)
    return counts

train_counts = scan_dataset(TRAIN_DIR)
test_counts = scan_dataset(TEST_DIR)

print("Training Classes:")
print(list(train_counts.keys()) if train_counts else "None found")
print("\\nTraining Images:")
if train_counts:
    for name, count in train_counts.items():
        print(f"{name}: {count}")
else:
    print("0 images found. Please add images to dataset/train/<PersonName>/")

print("\\nTest Images:")
if test_counts:
    for name, count in test_counts.items():
        print(f"{name}: {count}")
else:
    print("0 test images found. Please add images to dataset/test/<PersonName>/")
    
if not train_counts:
    print("\\n⚠️ STOP: No training images found. Please place images in the dataset/train/ folder before continuing.")

# Simple Dataset Distribution Chart
def plot_dataset_distribution(train_counts, test_counts):
    all_people = sorted(list(set(list(train_counts.keys()) + list(test_counts.keys()))))
    if not all_people:
        print("No images found to plot dataset distribution.")
        return
        
    train_vals = [train_counts.get(p, 0) for p in all_people]
    test_vals = [test_counts.get(p, 0) for p in all_people]
    
    x = np.arange(len(all_people))
    width = 0.35
    
    plt.figure(figsize=(max(6, len(all_people) * 1.5), 4))
    bars1 = plt.bar(x - width/2, train_vals, width, label='Training Images', color='#4C72B0')
    bars2 = plt.bar(x + width/2, test_vals, width, label='Testing Images', color='#55A868')
    
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(h), ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(h), ha='center', va='bottom', fontsize=9)
            
    plt.xlabel('Person Name')
    plt.ylabel('Number of Images')
    plt.title('FaceVault - Dataset Distribution per Person')
    plt.xticks(x, all_people, rotation=20 if len(all_people) > 4 else 0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

if train_counts or test_counts:
    plot_dataset_distribution(train_counts, test_counts)
""")

# 9. Visualize Sample Data
add_md("""## 9. Visualize Sample Data
Let's display one sample image for each person from the training dataset to verify everything is loaded correctly.""")

add_code("""def visualize_samples():
    if not train_counts:
        print("No training images found. Please place images inside dataset/train/<PersonName>/.")
        return
        
    num_classes = len(train_counts)
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
    if num_classes == 1:
        axes = [axes]
        
    for ax, person_name in zip(axes, train_counts.keys()):
        person_dir = TRAIN_DIR / person_name
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if images:
            img = cv2.imread(str(images[0]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"{person_name}\\n({len(images)} images)")
            else:
                ax.text(0.5, 0.5, "Could not load image", ha='center', va='center')
        ax.axis('off')
            
    plt.tight_layout()
    plt.show()

visualize_samples()""")

# 10. Face Detection
add_md("""## 10. Face Detection
We use OpenCV's Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) for face detection. It's lightweight, fast, and works well for frontal faces.

The function below downloads the cascade file automatically if it's missing from your OpenCV installation.""")

add_code("""HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def get_face_detector():
    cascade_file = Path(HAAR_CASCADE_PATH)
    if not cascade_file.exists():
        print("Downloading Haar cascade file from OpenCV GitHub...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        cascade_file.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(cascade_file))
        print(f"Downloaded to: {cascade_file}")
        
    cascade = cv2.CascadeClassifier(str(cascade_file))
    if cascade.empty():
        raise RuntimeError("Failed to load Haar cascade classifier.")
    return cascade

face_detector = get_face_detector()

def detect_faces(image):
    \"\"\"
    Detects faces in an image.
    Returns: list of (x, y, w, h) bounding boxes
    \"\"\"
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    return faces""")

# 11. Preprocessing
add_md("""## 11. Preprocessing
To train the LBPH model, all faces must be preprocessed:
1. **Detect** the face.
2. **Crop** the face from the background.
3. **Convert to Grayscale** (LBPH operates on single-channel images).
4. **Resize** to a standard dimension (e.g., 200x200 pixels) to ensure consistent feature extraction.""")

add_code("""def preprocess_image(img_path):
    \"\"\"
    Loads an image, detects the largest face, crops, converts to grayscale, and resizes.
    Returns the preprocessed face image, or None if no face is found.
    \"\"\"
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    
    if len(faces) == 0:
        return None, img
        
    # If multiple faces, pick the largest one for training
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces_sorted[0]
    
    face_roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, FACE_SIZE)
    
    return face_resized, img

# Demonstrate preprocessing on one sample image (if available)
if train_counts:
    first_person = list(train_counts.keys())[0]
    sample_imgs = list((TRAIN_DIR / first_person).glob("*.*"))
    valid_sample = None
    for s_img in sample_imgs:
        face_processed, orig_img = preprocess_image(s_img)
        if face_processed is not None:
            valid_sample = (face_processed, orig_img)
            break
            
    if valid_sample:
        face_processed, orig_img = valid_sample
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(face_processed, cmap='gray')
        ax2.set_title(f"Preprocessed Face\\n{FACE_SIZE}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No faces detected in the sample images for preprocessing preview.")
else:
    print("No training images found to demonstrate preprocessing.")
""")

# 12. Prepare Training Data
add_md("""## 12. Prepare Training Data
Now, we iterate through all images in `dataset/train/`, apply preprocessing, and associate each face with a numeric label. 

Machine learning models require numeric labels, so we also create a mapping between the numeric label (e.g., `0`) and the person's name (e.g., `"Loga"`).""")

add_code("""def prepare_training_data():
    faces_data = []
    labels_data = []
    
    name_to_label = {}
    label_to_name = {}
    current_label = 0
    
    total_imgs = 0
    valid_faces = 0
    no_faces = 0
    
    if not train_counts:
        print("Skipping training data preparation: No training data found.")
        return [], [], {}
        
    for person_name in sorted(train_counts.keys()):
        person_dir = TRAIN_DIR / person_name
        
        # Assign numeric label to person
        name_to_label[person_name] = current_label
        label_to_name[current_label] = person_name
        
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            total_imgs += 1
            face_processed, _ = preprocess_image(img_path)
            
            if face_processed is not None:
                faces_data.append(face_processed)
                labels_data.append(current_label)
                valid_faces += 1
            else:
                no_faces += 1
                
        current_label += 1
        
    print(f"--- Data Preparation Summary ---")
    print(f"Total images processed : {total_imgs}")
    print(f"Images with valid faces: {valid_faces}")
    print(f"Images with no face    : {no_faces}")
    
    print("\\n--- Label Mapping ---")
    for lbl, name in label_to_name.items():
        print(f"Label {lbl} -> {name}")
        
    return faces_data, labels_data, label_to_name

faces, labels, label_map = prepare_training_data()""")

# 13. Train the Model
add_md("""## 13. Train the Model
We train the **LBPH (Local Binary Pattern Histogram)** model using the extracted faces and numeric labels.

Once trained, the model is saved to disk so it can be reused without retraining.""")

add_code("""import pickle

model_path = MODEL_DIR / "facevault_lbph.yml"
label_map_path = MODEL_DIR / "label_map.pkl"

if len(faces) > 0:
    print("Initializing LBPH Face Recognizer...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    print("Training model... (this may take a few moments)")
    recognizer.train(faces, np.array(labels))
    
    # Save model and label mapping
    recognizer.save(str(model_path))
    with open(label_map_path, 'wb') as f:
        pickle.dump(label_map, f)
        
    print(f"\\n✅ Model trained successfully on {len(faces)} faces across {len(label_map)} people.")
    print(f"Model saved to: {model_path}")
    print(f"Label map saved to: {label_map_path}")
else:
    print("⚠️ Cannot train model: No valid training faces found.")
    recognizer = None""")

# 14. Training Visualization & Explanation
add_md("""## 14. Training Visualization & Explanation

**How LBPH Works (Local Binary Patterns Histograms):**

1. **Local Patterns:** The algorithm looks at each pixel and compares it to its surrounding neighbors. If the neighbor's intensity is greater than or equal to the center pixel, it writes `1`; otherwise, `0`. This creates a binary number (e.g., `11010011`) for that pixel.
2. **Histograms:** The face image is divided into a grid (e.g., 8x8). For each grid cell, it creates a histogram of the binary patterns present in that cell.
3. **Face Description:** All these histograms are concatenated into a single, long vector. This vector represents the texture and features of the face.
4. **Comparison:** When a new face is presented, its histogram vector is compared to the saved vectors of known faces. The distance between the vectors indicates similarity (lower distance = better match).""")

add_code("""# Simple Face Feature Histogram Chart
def plot_face_feature_histogram():
    if len(faces) == 0:
        print("Train model first to view face feature chart.")
        return
        
    sample_face = faces[0]
    sample_name = label_map.get(labels[0], "Person")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    
    # 1. Preprocessed Grayscale Face
    ax1.imshow(sample_face, cmap='gray')
    ax1.set_title(f"Face Sample: {sample_name}")
    ax1.axis('off')
    
    # 2. Pixel Intensity Distribution (Foundation of LBPH)
    ax2.hist(sample_face.ravel(), bins=32, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax2.set_title(f"Pixel Intensity Histogram ({sample_name})")
    ax2.set_xlabel("Pixel Intensity (0 - 255)")
    ax2.set_ylabel("Pixel Frequency")
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

plot_face_feature_histogram()""")

# 15. Single Image Recognition
add_md("""## 15. Single Image Recognition
The recognition function performs the following pipeline:
1. Detect faces in the target image.
2. Crop and preprocess each face.
3. Use the trained LBPH model to predict the numeric label and calculate the distance.
4. If the distance is below the `RECOGNITION_THRESHOLD`, we assign the known name. If it's higher, the face is marked as **"Unknown"**.
5. The distance is mapped to a friendly "Confidence %" score for display.""")

add_code("""def distance_to_confidence(distance, threshold=RECOGNITION_THRESHOLD):
    \"\"\"
    Converts LBPH distance to a percentage confidence score.
    Lower distance means higher similarity.
    \"\"\"
    if distance > threshold:
        # It's an unknown face, scale confidence from 0 to 49%
        conf = 50.0 - ((distance - threshold) / threshold) * 50.0
    else:
        # Known face, scale confidence from 50% to 100%
        conf = 100.0 - (distance / threshold) * 50.0
        
    return max(0.0, min(100.0, round(conf, 1)))

def recognize_faces_in_image(image_path, recognizer, label_map):
    \"\"\"
    Detects and recognizes all faces in an image.
    Returns the annotated image and a list of predictions.
    \"\"\"
    if recognizer is None:
        print("Error: Recognizer model not loaded.")
        return None, []
        
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None, []
        
    img_display = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces_detected = detect_faces(gray)
    predictions = []
    
    for (x, y, w, h) in faces_detected:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, FACE_SIZE)
        
        # Predict
        label, distance = recognizer.predict(face_resized)
        conf_percent = distance_to_confidence(distance)
        
        if distance <= RECOGNITION_THRESHOLD:
            name = label_map.get(label, "Unknown")
            color = (0, 255, 0) # Green for known
        else:
            name = "Unknown"
            color = (0, 0, 255) # Red for unknown
            
        predictions.append({"name": name, "confidence": conf_percent, "distance": distance})
        
        # Draw bounding box
        cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label_text = f"{name} ({conf_percent}%)"
        cv2.rectangle(img_display, (x, y+h), (x + len(label_text)*12, y+h+28), color, cv2.FILLED)
        cv2.putText(img_display, label_text, (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    return img_display, predictions""")

# 16 & 17. Test Image & Unknown Face Test
add_md("""## 16 & 17. Test Image & Unknown Face Test
Let's run the recognition on an actual test image. If you provide an image of someone not in the training set, they should be labeled as **"Unknown"**.""")

add_code("""# Find a sample test image
sample_test_img = None
if test_counts:
    first_test_person = list(test_counts.keys())[0]
    test_imgs = list((TEST_DIR / first_test_person).glob("*.*"))
    if test_imgs:
        sample_test_img = test_imgs[0]

if sample_test_img and recognizer is not None:
    print(f"Testing on image: {sample_test_img.name} (True Identity: {sample_test_img.parent.name})")
    
    annotated_img, preds = recognize_faces_in_image(sample_test_img, recognizer, label_map)
    
    if annotated_img is not None:
        # Display predictions text
        for p in preds:
            print(f"- Detected: {p['name']} | Confidence: {p['confidence']}% | Distance: {p['distance']:.2f}")
            
        # Show image
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Recognition Result")
        plt.show()
else:
    print("Cannot run test: Either no test images found or model not trained.")
""")

# 18. Multiple Face Test
add_md("""## 18. Multiple Face Test
The function `recognize_faces_in_image` uses `detectMultiScale`, which inherently detects multiple faces in a single frame. It loops over all bounding boxes, preprocessing and predicting the identity of each face independently.""")

# 19. Test Dataset Evaluation
add_md("""## 19. Test Dataset Evaluation
To formally evaluate the model, we run it against all images in the `dataset/test/` directory.

We compare the **Ground Truth** (the folder name) against the **Model Prediction**.
We will calculate standard machine learning classification metrics.""")

add_code("""def evaluate_model():
    if not test_counts or recognizer is None:
        print("Cannot evaluate: Model not trained or test dataset empty.")
        return None, None
        
    y_true = []
    y_pred = []
    total_imgs = 0
    no_faces = 0
    
    print("Running evaluation on test dataset...")
    
    for person_name in sorted(test_counts.keys()):
        person_dir = TEST_DIR / person_name
        
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            total_imgs += 1
            
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            
            predicted_name = "Unknown"
            
            if len(faces) > 0:
                # Pick largest face
                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces_sorted[0]
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, FACE_SIZE)
                
                label, distance = recognizer.predict(face_resized)
                if distance <= RECOGNITION_THRESHOLD:
                    predicted_name = label_map.get(label, "Unknown")
            else:
                no_faces += 1
                
            y_true.append(person_name)
            y_pred.append(predicted_name)
            
    print(f"Evaluated {total_imgs} images. ({no_faces} images had no detectable faces)")
    return y_true, y_pred

y_true, y_pred = evaluate_model()""")

# 20. False Positives and False Negatives
add_md("""## 20. False Positives and False Negatives
Understanding the errors in a biometric system is crucial:

* **False Positive (Misidentification)**: The model assigned a wrong specific person's name to a face. (e.g., Identifying "Person A" as "Person B", or identifying an unknown intruder as a registered employee).
* **False Negative (Missed Recognition)**: The model predicted "Unknown" for a person who is actually registered in the system.

Adjusting the `RECOGNITION_THRESHOLD` affects the balance: A stricter (lower) threshold increases False Negatives but reduces False Positives (more secure).""")

# 21. Confusion Matrix
add_md("""## 21. Confusion Matrix
The confusion matrix visually shows where the model gets confused between different classes.""")

add_code("""def plot_confusion_matrix_fn(y_true, y_pred):
    if not y_true or not y_pred:
        print("No evaluation data available to plot confusion matrix.")
        return
    
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(max(5, len(labels)*1.2), max(4, len(labels)*1.1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    
    ax.set_title("FaceVault - Confusion Matrix", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Identity")
    ax.set_ylabel("True Identity")
    plt.tight_layout()
    
    cm_path = EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")
    plt.show()

if y_true and y_pred:
    plot_confusion_matrix_fn(y_true, y_pred)""")

# 22 & 23. Performance Metrics & Results Table
add_md("""## 22 & 23. Performance Metrics & Results Table
We calculate Accuracy, Precision, Recall, and F1-Score using a macro average (averaging the unweighted mean per label) to treat all individuals equally.""")

add_code("""def calculate_and_display_metrics(y_true, y_pred):
    if not y_true or not y_pred:
        display(Markdown("### Evaluation cannot be performed until test images are added."))
        return
        
    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    md_table = f\"\"\"
### Model Performance Metrics

| Metric | Score |
|---|---|
| Accuracy | {acc:.2f}% |
| Precision (Macro) | {precision:.2f}% |
| Recall (Macro) | {recall:.2f}% |
| F1 Score (Macro) | {f1:.2f}% |
\"\"\"
    display(Markdown(md_table))
    
    # Classification report
    labels = sorted(list(set(y_true + y_pred)))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    
    with open(EVALUATION_DIR / "report.txt", "w") as f:
        f.write(report)
        
    print("Detailed Classification Report:\\n")
    print(report)
    
    # Plot Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [acc, precision, recall, f1]
    
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(metrics, scores, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'], width=0.55)
    plt.ylim(0, 110)
    plt.title("FaceVault Performance Metrics", fontsize=12, fontweight='bold')
    plt.ylabel("Percentage (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    plot_path = PLOTS_DIR / "performance_metrics.png"
    plt.savefig(plot_path, dpi=150)
    plt.tight_layout()
    plt.show()

if y_true and y_pred:
    calculate_and_display_metrics(y_true, y_pred)""")

# 24. Recognition Result Gallery
add_md("""## 24. Recognition Result Gallery
Let's visually verify the model on a few random test images.""")

add_code("""import random

def show_result_gallery():
    if not test_counts or recognizer is None:
        return
        
    # Gather all test images
    all_test_imgs = []
    for person_dir in TEST_DIR.iterdir():
        if person_dir.is_dir():
            all_test_imgs.extend(list(person_dir.glob("*.*")))
            
    if not all_test_imgs: return
    
    # Pick up to 4 random images
    num_samples = min(4, len(all_test_imgs))
    sample_imgs = random.sample(all_test_imgs, num_samples)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1: axes = [axes]
    
    for ax, img_path in zip(axes, sample_imgs):
        annotated_img, preds = recognize_faces_in_image(img_path, recognizer, label_map)
        
        if annotated_img is not None:
            # Save output
            out_path = RECOGNITION_RESULTS_DIR / f"result_{img_path.name}"
            cv2.imwrite(str(out_path), annotated_img)
            
            ax.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            
            true_name = img_path.parent.name
            pred_name = preds[0]['name'] if preds else "No face"
            ax.set_title(f"True: {true_name}\\nPred: {pred_name}")
        
    plt.tight_layout()
    plt.show()
    print(f"Results saved to {RECOGNITION_RESULTS_DIR}")

show_result_gallery()""")

# 25. Project Flow Diagram
add_md("""## 25. Project Flow Diagram

```mermaid
flowchart TD
    A[Training Images] --> B[Face Detection]
    B --> C[Face Preprocessing]
    C --> D[LBPH Feature Extraction]
    D --> E[Train LBPH Model]
    E --> F[Saved Model]

    G[Test Image / Webcam] --> H[Face Detection]
    H --> I[Face Preprocessing]
    I --> J[LBPH Prediction]
    F --> J
    J --> K{Recognition Threshold}
    K -->|Match| L[Known Person]
    K -->|No Match| M[Unknown]
```""")

# 26. Optional Webcam Demo
add_md("""## 26. Optional Webcam Demo
The core system is built entirely on the manually provided image datasets.
If your machine has a webcam attached and you are running this locally (not in a remote Jupyter environment), you can run the code below to test the recognition in real time.

*(Press the **Q** key while focusing the video window to quit)*""")

add_code("""def webcam_demo():
    if recognizer is None:
        print("Model not trained yet.")
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam detected or could not open webcam.")
        return
        
    print("Starting webcam demo. Press 'q' in the video window to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = detect_faces(gray)
        
        for (x, y, w, h) in faces_detected:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, FACE_SIZE)
            
            label, distance = recognizer.predict(face_resized)
            conf_percent = distance_to_confidence(distance)
            
            if distance <= RECOGNITION_THRESHOLD:
                name = label_map.get(label, "Unknown")
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label_text = f"{name} ({conf_percent}%)"
            cv2.rectangle(frame, (x, y+h), (x + len(label_text)*10, y+h+20), color, cv2.FILLED)
            cv2.putText(frame, label_text, (x+5, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        cv2.imshow("FaceVault Webcam Demo (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

# Uncomment the line below to run the webcam demo
# webcam_demo()""")

# 27. Model Limitations
add_md("""## 27. Model Limitations
* **Algorithm**: LBPH is great for educational use, local demonstrations, and constrained environments, but it is not a production-grade biometric security system. Deep learning methods (like ArcFace or FaceNet) are more robust but require significantly more computing power.
* **Lighting Dependency**: LBPH uses pixel intensity patterns, making it somewhat sensitive to dramatic changes in lighting (e.g., strong shadows).
* **Pose and Angles**: The model is trained on mostly frontal faces. Extreme side profiles may not be recognized.
* **Occlusion**: Masks, heavy glasses, or hands covering the face will degrade performance.
* **Threshold Selection**: Finding the perfect `RECOGNITION_THRESHOLD` is a trade-off between allowing false positives and causing false negatives.""")

# 28. Ethical and Privacy Section
add_md("""## 28. Ethical and Privacy Section
Facial recognition is a powerful technology that involves sensitive biometric data. Always adhere to the following principles:

* **Consent**: Always obtain clear, explicit consent before collecting and processing anyone's facial images.
* **Authorization**: Only use images that you are authorized to use.
* **Security**: Keep the facial datasets and the trained model secure and local. Do not upload them to public repositories.
* **Surveillance**: Do not use the system for unauthorized tracking or surveillance.
* **Transparency**: Clearly inform users when a facial recognition system is active (e.g., via a webcam).
* **Data Lifecycle**: Delete personal datasets when they are no longer needed for the project.""")

# 29. Final Project Summary
add_md("""## 29. Final Project Summary""")

add_code("""def generate_final_summary():
    train_people = len(train_counts) if train_counts else 0
    total_train = sum(train_counts.values()) if train_counts else 0
    total_test = sum(test_counts.values()) if test_counts else 0
    
    acc_str = precision_str = recall_str = f1_str = "N/A"
    
    if y_true and y_pred:
        acc_str = f"{accuracy_score(y_true, y_pred)*100:.2f}%"
        precision_str = f"{precision_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%"
        recall_str = f"{recall_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%"
        f1_str = f"{f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%"
        
    summary_md = f\"\"\"
### FaceVault Project Summary

* **Number of registered people:** {train_people}
* **Total training images:** {total_train}
* **Total test images:** {total_test}
* **Model:** OpenCV LBPHFaceRecognizer
* **Image size:** {FACE_SIZE}
* **Recognition threshold:** {RECOGNITION_THRESHOLD}

#### Performance
* **Accuracy:** {acc_str}
* **Precision:** {precision_str}
* **Recall:** {recall_str}
* **F1-score:** {f1_str}

#### Output Locations
* **Model saved at:** `{MODEL_DIR / 'facevault_lbph.yml'}`
* **Evaluation report saved at:** `{EVALUATION_DIR / 'report.txt'}`
* **Confusion matrix saved at:** `{EVALUATION_DIR / 'confusion_matrix.png'}`
\"\"\"
    display(Markdown(summary_md))

generate_final_summary()""")

with open('FaceVault_Facial_Recognition.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
