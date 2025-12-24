![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Contrib-green?logo=opencv)
![Computer Vision](https://img.shields.io/badge/Field-Computer%20Vision-purple)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

# Face Tracker & Recognition System

A real-time **face detection and face recognition** system built with **Python + OpenCV**. This project supports live webcam tracking, identity recognition using **LBPH**, dataset creation, model training, persistence, and an interactive command-driven UI.

> Built as a local, offline computer vision system for learning, experimentation, and research purposes.

---

##  Features

*  Real-time face detection via webcam
*  Face recognition using **LBPH (Local Binary Patterns Histograms)**
*  Automatic dataset handling (folder-per-person)
*  Add new faces directly from the camera
*  Train / re-train the recognizer at runtime
*  Persistent model & label storage
*  Screenshot capture
*  Interactive keyboard controls
*  Modular, readable, and extensible codebase

---

##  Tech Stack

* **Python 3**
* **OpenCV (opencv-contrib-python)**
* **NumPy**
* Haar Cascade (face detection)
* LBPH Face Recognizer

---

##  Project Structure

```text
.
├── face_data/                 # Face dataset (one folder per person)
│   ├── Alice/
│   ├── Bob/
│   └── ShadowRoot17/
├── face_recognizer_model.yml  # Trained LBPH model
├── label_mappings.pkl         # Label ↔ name mappings
├── camra-1.py                 # Main application script
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
https://github.com/Gl1tch0v3rr1d3/real-time-face-recagisation-.git
cd real-time-face-recagisation-
```

### 2. Install dependencies

```bash
pip install opencv-contrib-python numpy
```

>  `opencv-contrib-python` is required for `cv2.face` (LBPH).

---

## 🚀 Usage

Run the main script:

```bash
python camra-1.py 
```

On first launch, you will be guided through **face database setup**.

---

##  Controls (Live Mode)

| Key | Action                      |
| --: | --------------------------- |
| `q` | Quit application            |
| `s` | Save screenshot             |
| `a` | Add a new face              |
| `t` | Train / re-train recognizer |
| `r` | Toggle recognition ON / OFF |

---

## How Face Recognition Works

1. **Detection**: Faces are detected using Haar Cascade classifiers.
2. **Preprocessing**: Faces are converted to grayscale and resized.
3. **Training**: LBPH extracts texture-based features and learns per-identity patterns.
4. **Prediction**: Incoming faces are matched against the trained model using distance metrics.
5. **Thresholding**: Predictions below a confidence threshold are treated as recognized.

> Lower LBPH distance = better match.

---

## Why LBPH?

* Works well with **small datasets**
* Robust under varying lighting conditions
* Fast enough for real-time inference
* Ideal for offline and edge systems

---

## Privacy & Ethics Notice

This project performs **biometric identification**.

* ✔️ Intended for **local, offline, educational use only**
* ❌ Do NOT deploy publicly
* ❌ Do NOT collect faces without explicit consent
* ❌ Do NOT use for surveillance or identification of unaware individuals

You are responsible for complying with all applicable privacy and data protection laws.

---

## Possible Improvements

* Replace Haar Cascade with:

  * MediaPipe Face Detection
  * OpenCV DNN (SSD / ResNet)
* Switch from LBPH to **face embeddings** (FaceNet / ArcFace)
* Add face tracking across frames (IDs)
* Improve confidence normalization
* Add dataset augmentation
* GUI version (Qt / Tkinter)

---

## Status

✔️ Functional
✔️ Modular
✔️ Real-time
✔️ Offline

This project is **beyond demo-level** and suitable for portfolios, CV experiments, and further research.

---

## 📜 License

MIT License — use responsibly.

---

> Built with focus, curiosity, and zero tutorial-copy energy.
