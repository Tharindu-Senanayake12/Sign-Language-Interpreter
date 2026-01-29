# Sign Language Interpreter

A **real-time sign language interpreter** that recognizes hand gestures using computer vision and AI. This project leverages **MediaPipe** for hand landmark detection and **TensorFlow** for gesture classification, providing both static and dynamic gesture recognition with optimized **TensorFlow Lite models** for fast inference.

---

## Features

* **Real-time hand tracking** using webcam with MediaPipe Hands
* **Static gesture recognition** using hand keypoint landmarks
* **Dynamic gesture recognition** using point/motion history
* **Custom dataset logging** for new gestures (via `app.py`)
* **Neural network classification** with TensorFlow
* **Optimized for real-time inference** using TensorFlow Lite
* **Visualization** of hand landmarks, bounding boxes, and gesture info

---

## Demo

<img src="screenshots/dashboard.png" width="700"/>


---

## Technologies & Tools

* **Programming Language:** Python 3.10+
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** TensorFlow, TensorFlow Lite, NumPy, scikit-learn
* **Data Handling:** CSV
* **Visualization:** Matplotlib, Seaborn (for evaluation)

---

## Project Structure

```
sign-language-interpreter/
│
├─ app.py                       # Main application for real-time gesture recognition
├─ model/
│  ├─ keypoint_classifier/
│  │  ├─ keypoint_classifier.hdf5  # Trained Keras model
│  │  ├─ keypoint_classifier.tflite # Optimized TFLite model for inference
│  │  ├─ keypoint.csv              # Collected keypoint dataset
│  │  └─ keypoint_classifier_label.csv  # Gesture labels
│  └─ point_history_classifier/
│     ├─ point_history_classifier.hdf5
│     ├─ point_history_classifier.tflite
│     └─ point_history_classifier_label.csv
├─ keypoint_classification.ipynb  # Training and TFLite conversion notebook
├─ utils.py                       # Utility functions for FPS and visualization
├─ screenshots/                   # Screenshots or demo images
└─ requirements.txt               # Python dependencies
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/sign-language-interpreter.git
cd sign-language-interpreter
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Run the real-time interpreter:

```bash
python app.py
```

2. Controls:

* `ESC` – Exit the application
* `k` – Switch to keypoint logging mode
* `n` – Switch to point history logging mode
* `0-99` – Input gesture label for logging (can go for upto 100 different sign languages)
* `Enter` – Confirm label
* `p` – Save gesture sample for dataset (Capture)

---

## Training Your Own Model

1. Collect keypoint data via `app.py`. Make sure `mode=1` (keypoint logging).
2. Open `keypoint_classification.ipynb` and run the notebook to train the model.
3. The trained model will be saved as:

   * Keras: `keypoint_classifier.hdf5`
   * TensorFlow Lite: `keypoint_classifier.tflite`
4. Replace the models in `model/keypoint_classifier/` for inference.

---

## Evaluation

* Accuracy and confusion matrix are calculated using the notebook.
* Misclassified gestures can be improved by adding more samples or augmenting the dataset.

---

## Future Improvements

* Support **sentence-level interpretation** by combining gestures
* Build a **cross-platform GUI** or mobile application

---



