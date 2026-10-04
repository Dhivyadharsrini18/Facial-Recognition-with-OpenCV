# 🔐 FaceVault — AI-Based Facial Recognition System

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/LBPH-Face%20Recognition-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Offline-Privacy%20Focused-success?style=for-the-badge" />

</p>

<p align="center">
  <b>A lightweight, offline and privacy-focused facial recognition system built with Python and OpenCV.</b>
</p>

---

## 🧠 About FaceVault

**FaceVault** is an offline facial recognition system designed to detect faces, learn facial patterns from a custom dataset, recognize registered individuals, identify unknown faces, and evaluate recognition performance.

The project uses **OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer**, making it lightweight and suitable for small datasets and CPU-based execution.

The complete workflow is implemented through a **Jupyter Notebook**, making the project easy to understand, demonstrate, experiment with, and reproduce.

> 🔒 **Privacy First:** FaceVault is designed to work locally. Facial images and recognition data remain within the project environment unless the user chooses to share them.

---

# ✨ Key Features

- 👤 **Face Detection**
- 🧠 **LBPH-Based Face Recognition**
- 🖼️ **Image Preprocessing**
- 🔍 **Registered Person Identification**
- ❓ **Unknown Face Detection**
- 👥 **Multiple Face Recognition**
- 📊 **Accuracy Evaluation**
- 📈 **Precision, Recall & F1-Score**
- 🔢 **Confusion Matrix**
- 📉 **Performance Visualization**
- 🎯 **Configurable Recognition Threshold**
- 📓 **Complete Jupyter Notebook Workflow**
- 🔐 **Offline & Privacy-Focused**
- 💻 **CPU-Friendly Implementation**

---

# 🎯 Project Objectives

The main objectives of FaceVault are:

1. Detect human faces from input images.
2. Extract and preprocess facial regions.
3. Train a facial recognition model using LBPH.
4. Identify registered individuals.
5. Reject unregistered faces as **Unknown**.
6. Evaluate recognition performance using standard classification metrics.
7. Visualize model performance through charts and a confusion matrix.
8. Provide an understandable and reproducible AI/ML workflow.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Input Image/Data   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Face Detection    │
                    │     OpenCV Haar      │
                    │       Cascade        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Face Preprocessing │
                    │                      │
                    │ • Crop Face          │
                    │ • Grayscale          │
                    │ • Resize 200 × 200   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    LBPH Training     │
                    │                      │
                    │ Local Binary Patterns│
                    │     + Histograms     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Trained Model      │
                    │ facevault_lbph.yml   │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌──────────────────┐       ┌──────────────────┐
        │  Known Face      │       │   Unknown Face   │
        │                  │       │                  │
        │  Person Name     │       │    "Unknown"     │
        └──────────────────┘       └──────────────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌──────────────────────┐
                    │    Evaluation        │
                    │                      │
                    │ Accuracy             │
                    │ Precision            │
                    │ Recall               │
                    │ F1-Score             │
                    │ Confusion Matrix     │
                    └──────────────────────┘

